"""第 2 步（重做）EDL 生成：每个片段是不可分割单位，最多用 1 次、整段播放；播放时长 = 整数拍（4/6/7/8/9/10/12），
用匀速变速 0.85–1.15× 或击中点速度坡凑拍数；带宽受限 DP 同时决定弃用、局部顺序、拍数，并奖励 BGA 切换点落在片段边界。
输出 work/edl.csv、work/plan_v4_stats.json。覆盖项：overrides_v4.json {keep:[], drop:[]}。"""
import csv,json,pickle,os,sys,numpy as np
from collections import Counter,defaultdict
os.chdir(os.path.dirname(os.path.abspath(__file__)))
FPS=30; T=278.806
A=pickle.load(open('work/audio_an.pkl','rb')); per=A['per']; q=per/2; t0=A['t0']; AFPS=22050/256
PK=A['pk']; ENV=A['env']
VO=json.load(open('work/vocal_onsets.json'))
CLIPS={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}
FE={f['clip_id']:f for f in json.load(open('work/clip_feats.json'))}
OV=json.load(open('overrides_v4.json')); KEEP=set(OV.get('keep',[])); DROP=set(OV.get('drop',[]))-KEEP; SPECIAL={int(k) for k in OV.get('special',{})}&KEEP
NSET=[4,6,7,8,9,10,12,13,14,15,16]; VMIN,VMAX=0.85,1.15; TRIM=0.1
RAMP_G=0.2        # 速度坡：击中点附近 0.5 s（0.1 坡+0.3 平台+0.1 坡）降到 0.5×，等效多出 0.2 s 播放，其余略加速补偿
OUTRO_V=(0.55,0.85); OUTRO_N=(12,16); OUTRO_T=266.0
NB=830; LEAD=0.010; TOL=0.040
BGA_EV=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]
SEGS=[(0,16,'intro'),(16,79,'regular'),(79,100,'peak'),(100,188,'irregular'),(188,202,'sparse'),(202,219.5,'frag'),(219.5,230,'peak'),(230,266,'after'),(266,T,'outro')]
SEGN={n:i for i,n in enumerate(['intro','regular','peak','irregular','sparse','frag','after','outro'])}
def seg_of(t):
    for a,b,n in SEGS:
        if a<=t<b: return n
    return 'outro'
def smooth(a,s): w=max(1,int(s*AFPS)); return np.convolve(a,np.ones(w)/w,'same')
def n01(a): lo,hi=np.percentile(a,2),np.percentile(a,98); return np.clip((a-lo)/(hi-lo+1e-9),0,1)
E_=n01(smooth(A['ai'],1.0)); D_=n01(smooth(A['dens'],1.0)); INT=0.6*E_+0.4*D_; INT4=smooth(INT,4.0)
def at(a,t): return float(a[min(len(a)-1,max(0,int(t*AFPS)))])
# ---------------- 拍网格：全局网格 + 局部相位漂移（不规则区 100–188 用两端线性插值，不跟随其不稳定相位）
ts,dr=np.load('work/drift_curve.npy')
d100,d188=np.interp(100,ts,dr),np.interp(188,ts,dr)
dr2=dr.copy(); m=(ts>100)&(ts<188); dr2[m]=d100+(d188-d100)*(ts[m]-100)/88
def drift(t): return float(np.interp(t,ts,dr2))
BT=np.array([t0+k*per+drift(t0+k*per) for k in range(NB+1)])
# BGA 目标拍
bga_target={}; bga_info=[]
for e in BGA_EV:
    k=int(np.argmin(np.abs(BT-e))); dist=e-BT[k]
    bga_info.append(dict(event=e,beat=k,beat_time=round(float(BT[k]),3),event_minus_beat_ms=round(dist*1000)))
    if abs(dist)<=TOL: bga_target[k]=e
inside=np.zeros((NB+1,len(NSET)),bool)
for ni,n in enumerate(NSET):
    for k in bga_target:
        lo=max(0,k-n+1); inside[lo:k,ni]=True   # b<k<b+n  ⇔ k-n<b<k
# ---------------- 片段特征
ids=sorted(c for c in FE if c not in DROP)
rank=lambda v:(np.argsort(np.argsort(v))/(len(v)-1))
M=np.array([FE[i]['motion'] for i in ids]); Sa=np.array([FE[i]['sat'] for i in ids]); V=np.array([FE[i]['motion_std'] for i in ids])
S=dict(zip(ids,0.45*rank(M)+0.3*rank(Sa)+0.25*rank(V)))
LUM={i:FE[i]['lum'] for i in FE}; HUE={i:np.array(FE[i]['hue_hist']) for i in FE}
def similar(a,b):
    h=HUE[a]@HUE[b]/(np.linalg.norm(HUE[a])*np.linalg.norm(HUE[b])+1e-9); return bool(h>0.93 and abs(LUM[a]-LUM[b])<0.04)
def closure(c):
    ml=np.array(FE[c]['motion_local']); h=len(ml)//2
    return float((ml[:h].mean()-ml[h:].mean())/(ml.mean()+1e-6))
def hit_point(c):
    ml=np.array(FE[c]['motion_local']); D=float(CLIPS[c]['dur']); p=int(np.argmax(ml)); hs=p*0.5+0.25
    if ml[p]/(ml.mean()+1e-6)>=1.25 and 0.45<=hs<=D-0.45: return hs
    return None
# 可行拍数：每片段每个 n 的方案集合 {mode: dict}，mode ∈ uniform / ramp（仅峰值、不规则区）/ slow（仅收束池片段在收束区）
outro_pool=sorted([c for c in ids if S[c]<0.6],key=lambda c:-closure(c))[:4]
FEAS={}
for c in ids:
    D=float(CLIPS[c]['dur']); hp=hit_point(c)
    for n in NSET:
        L=n*per; opts={}
        lo,hi=max(VMIN*L,D-2*TRIM),min(VMAX*L,D)
        if lo<=hi: d=min(max(L,lo),hi); v=d/L; opts['uniform']=dict(mode='uniform',d=d,v=v,cost=3*abs(v-1))
        if hp is not None:
            Lr=L-RAMP_G; lo,hi=max(VMIN*Lr,D-2*TRIM),min(VMAX*Lr,D)
            if lo<=hi: d=min(max(Lr,lo),hi); v=d/Lr; opts['ramp']=dict(mode='ramp',d=d,v=v,cost=3*abs(v-1)-0.25,hit=hp)   # 峰值/不规则区优先
        if c in outro_pool and OUTRO_N[0]<=n<=OUTRO_N[1]:
            lo,hi=max(OUTRO_V[0]*L,D-2*TRIM),min(OUTRO_V[1]*L,D)
            if lo<=hi: d=min(max(0.7*L,lo),hi); v=d/L; opts['slow']=dict(mode='slow',d=d,v=v,cost=3*abs(v-0.7))
        if opts: FEAS[(c,n)]=opts
# ---------------- 目标时间与顺序 σ
IRb=rank(np.array([at(INT4,t) for t in BT[:NB]]))            # 每拍强度名次 0–1
rest=[c for c in ids if c not in outro_pool and c not in SPECIAL]
NS=len(rest); slot_t=BT[0]+(np.arange(NS)+0.5)*(266-BT[0])/NS; slot_I=np.array([at(INT4,t) for t in slot_t])
order_slots=np.argsort(slot_I); order_clips=sorted(rest,key=lambda c:S[c])
target={}
for s_idx,c in zip(order_slots,order_clips): target[c]=float(slot_t[s_idx])
for k,c in enumerate(outro_pool): target[c]=266+(k+0.5)*(BT[NB]-266)/4
for c,tt in zip(sorted(SPECIAL),(203.0,17.0)): target[c]=tt   # 97 在回落区之后（碎片区开头），109 在铺垫区之后
sigma=sorted(ids,key=lambda c:target[c]); N=len(sigma); pos={c:i for i,c in enumerate(sigma)}
SIM={c:{d:similar(c,d) for d in ids} for c in ids}
# ---------------- DP
K=4; LMIN=-6; NL=K-LMIN; INF=1e9
w_int,w_sim,w_bga=1.2,0.6,2.5
cdisc={c:(INF if c in KEEP else 1.2-0.6*S[c]) for c in ids}
segb=np.array([SEGN[seg_of(float(t))] for t in BT[:NB]]+[SEGN['outro']])
is_ri=np.isin(segb,[SEGN['peak'],SEGN['irregular']]); is_out=(segb==SEGN['outro']); is_low=np.isin(segb,[SEGN['intro'],SEGN['sparse'],SEGN['outro']])
NS_=len(NSET)
npref=np.zeros((8,len(NSET)),np.float32)
for ni,n in enumerate(NSET):
    npref[SEGN['intro'],ni]=0.12*max(0,9-n); npref[SEGN['peak'],ni]=0.12*max(0,n-7); npref[SEGN['sparse'],ni]=0.08*max(0,8-n)
F=np.full((N+1,8,NL,2,NB+1),INF,np.float32); BP=np.full((N+1,8,NL,2,NB+1),-1,np.int16)
F[N,0,:,:,NB]=0
def norm(i2,m2,l2):
    while m2&1: i2+=1; m2>>=1; l2-=1
    return i2,m2,max(l2,LMIN)
for i in range(N-1,-1,-1):
    for m in range(7,-1,-1):
        mask=m<<1
        for li in range(NL):
            last=li+LMIN; prev=sigma[i+last] if 0<=i+last<N else None
            for run in (1,2):
                best=np.full(NB+1,INF,np.float32); bp=np.full(NB+1,-1,np.int16)
                c=sigma[i]
                if cdisc[c]<INF:
                    i2,m2,l2=norm(i+1,mask>>1,last-1)
                    if i2<=N:
                        cand=cdisc[c]+F[i2,m2>>1,l2-LMIN,run-1]; upd=cand<best; best[upd]=cand[upd]; bp[upd]=0
                for j in range(K):
                    if i+j>=N or (mask>>j)&1: continue
                    c=sigma[i+j]; sim=prev is not None and SIM[c][prev]; run2=run+1 if sim else 1
                    if run2>2: continue
                    i2,m2,l2=norm(i,mask|(1<<j),j)
                    if i2>N: continue
                    base=w_int*np.abs(S[c]-IRb)+(w_sim if sim else 0)
                    for ni,n in enumerate(NSET):
                        opts=FEAS.get((c,n))
                        if opts is None: continue
                        cu=np.full(NB+1,INF,np.float32)
                        if 'uniform' in opts: cu[:]=opts['uniform']['cost']+0.8*is_out
                        if c in SPECIAL: cu=np.where(is_low,INF,cu)
                        if 'ramp' in opts: cu=np.where(is_ri,np.minimum(cu,opts['ramp']['cost']),cu)
                        if 'slow' in opts: cu=np.where(is_out,np.minimum(cu,opts['slow']['cost']),cu)
                        nxt=F[i2,m2>>1,l2-LMIN,run2-1]
                        tot=np.full(NB+1,INF,np.float32)
                        tot[:NB+1-n]=base[:NB+1-n]+cu[:NB+1-n]+npref[segb[:NB+1-n],ni]+w_bga*inside[:NB+1-n,ni]+nxt[n:]
                        upd=tot<best; best[upd]=tot[upd]; bp[upd]=1+j*len(NSET)+ni
                F[i,m,li,run-1]=best; BP[i,m,li,run-1]=bp
assert F[0,0,0,0,0]<INF,'DP 无解'
# 回溯
i,mask,last,run,b=0,0,LMIN,1,0; shots=[]; discarded=[]
while i<N:
    code=int(BP[i,mask>>1,last-LMIN,run-1,b])
    assert code>=0
    if code==0:
        discarded.append(sigma[i]); i,mask,last=norm(i+1,mask>>1,last-1)
    else:
        j=(code-1)//len(NSET); n=NSET[(code-1)%len(NSET)]; c=sigma[i+j]
        prev=sigma[i+last] if 0<=i+last<N else None; sim=prev is not None and SIM[c][prev]; run=run+1 if sim else 1
        opts=FEAS[(c,n)]; cand=[]
        if 'uniform' in opts: cand.append(opts['uniform'])
        if 'ramp' in opts and is_ri[b]: cand.append(opts['ramp'])
        if 'slow' in opts and is_out[b]: cand.append(opts['slow'])
        shots.append(dict(clip=c,b=b,n=n,**min(cand,key=lambda o:o['cost']))); b+=n; i,mask,last=norm(i,mask|(1<<j),j)
assert b==NB
# ---------------- 时间、速度、特效
CORP=[a for a,_ in VO['corp_100_188']]; FRAG=VO['frag_merged']
def ostr(o): return float(ENV[min(len(ENV)-1,int(o*AFPS))])
def win_pct(t,seg):
    e=at(E_,t)
    if seg=='intro': return 75+25*min(1,e/0.3)
    if seg=='regular': return 75+25*min(1,0.3+e)
    if seg in('peak','irregular','frag'): return 100
    if seg=='sparse': return 92
    if seg=='after': return 96
    return 100-25*min(1,(t-266)/9)
last_flash=-99; ascii_budget=[]
for k,s in enumerate(shots):
    k0,k1=s['b'],s['b']+s['n']
    s['t']=float(BT[k0])-(0 if k0 in bga_target else LEAD); s['te']=float(BT[k1])-(0 if k1 in bga_target else LEAD)
    s['dur']=s['te']-s['t']; s['seg']=seg_of(s['t']+0.01); s['n_idx']=k+1
    D=float(CLIPS[s['clip']]['dur']); tr=(D-s['d'])/2; s['src_in']=round(tr,3); s['src_out']=round(D-tr,3)
    Lr=s['dur']-(RAMP_G if s['mode']=='ramp' else 0); s['v']=s['d']/Lr; s['hit_rel']=round(s['hit']-s['src_in'],3) if s['mode']=='ramp' else None
    s['win']=round(win_pct(s['t'],s['seg'])); s['flash']=0; s['ascii']=0; s['bga']=''
    seg=s['seg']; t=s['t']; e=at(INT,t); fx=[]
    def ons_in(lst,pad=0.05): return [o for o in lst if s['t']+pad<=o<s['te']-pad]
    if seg=='intro': fx=[f'scan:{0.1+0.1*e:.2f}','vignette:0.3']
    elif seg=='regular':
        frac=(t-16)/63; fx=[f'scan:{0.15+0.2*e:.2f}','vignette:0.3']
        if frac>0.3: fx.append(f'beatpulse:{min(5,2+5*e):.1f}%:2')
        if frac>0.6: fx.append('rgb:2f@0')
    elif seg=='peak':
        fx=['scan:0.30',f'beatpulse:{min(6,3+4*e):.1f}%:1','rgb:3f@0',f'shake:{min(1.5,1.5*e):.1f}%','noise:0.15']
        o=sorted(ons_in(PK),key=lambda x:-ostr(x))[:max(1,int(1.5*s['dur']))]
        fx+=[f'rgb:2f@{x-s["t"]:.3f}' for x in sorted(o)]
        if e>0.7 and t-last_flash>=4: s['flash']=1; last_flash=t
    elif seg=='irregular':
        fx=[f'scan:{0.2+0.2*e:.2f}','noise:0.12',f'beatpulse:{min(5,2+4*e):.1f}%:2']
        cs=ons_in(CORP); fx+=[f'stutter:3f@{x-s["t"]:.3f} rgb:2f@{x-s["t"]:.3f}' for x in cs]
        others=[x for x in ons_in(PK) if all(abs(x-c)>0.12 for c in cs)]
        others=sorted(sorted(others,key=lambda x:-ostr(x))[:int(2*s['dur'])])
        for r_i,x in enumerate(others):   # 类型轮换 rgb/block/tear，帧数随能量 2–4
            ee=at(E_,x); nf=2 if ee<0.5 else (3 if ee<0.75 else 4); fx.append(f"{['rgb','block','tear'][r_i%3]}:{nf}f@{x-s['t']:.3f}")
    elif seg=='sparse': fx=['scan:0.25','vignette:0.6','noise:0.2','desat:0.2']
    elif seg=='frag':
        fx=['scan:0.3','noise:0.2','vignette:0.4']
        fr=ons_in(FRAG); thr=np.percentile([ostr(x) for x in FRAG],67)
        for i2,x in enumerate(fr):
            fx.append(('block:4f' if i2%2==0 else 'tear:4f')+f'@{x-s["t"]:.3f}')
            if ostr(x)>=thr: fx.append(f'stutter:2f@{x-s["t"]:.3f}')
    elif seg=='after': fx=['scan:0.25','vignette:0.45',f'beatpulse:{2+2*e:.1f}%:2','tone:cold']
    else: fx=[f'scan:{0.4+0.3*(t-266)/12:.2f}',f'noise:{0.25+0.3*(t-266)/12:.2f}','vignette:0.6','tone:cold']
    s['fx']=fx
# BGA 事件：命中边界 → 起始重音；否则在镜头内事件时刻放重音
hits=[]
for idx,ev in enumerate(BGA_EV):
    typ='rgb:4f' if idx%2==0 else 'block:4f'
    s=next((x for x in shots if x['t']-0.05<=ev<x['te']),None)
    if s is None: continue
    off=ev-s['t']
    if abs(off)<=TOL: s['fx']=[f'{typ}@0','bga_accent@0']+s['fx']; s['bga']=f'{ev} hit {round(1000*(s["t"]-ev)):+d}ms'; hits.append((ev,s['n_idx'],round(1000*(s['t']-ev))))
    else:
        s['fx']+= [f'{typ}@{off:.3f}',f'bga_accent@{off:.3f}']
        bd=min(abs(s['t']-ev),abs(s['te']-ev)); s['bga']=f'{ev} miss {round(1000*bd)}ms'
# ASCII 转场：碎片区前 10 个镜头尾 0.5s；两个峰值各 2 个 0.6s；144.3/187.2 所在镜头尾 0.6s
tot=0
for s in [x for x in shots if 202<=x['te']<=219.6][:10]: s['ascii']=0.5; tot+=0.5
for a,b in ((79,100),(219.5,230)):
    for s in [x for x in shots if a<=x['t']<b][1::3][:2]: s['ascii']=0.6; tot+=0.6
for ev in (144.31,187.21):
    s=next(x for x in shots if x['t']<=ev<x['te'])
    if not s['ascii']: s['ascii']=0.6; tot+=0.6
assert tot<=10
# ---------------- 特殊片段（97/109）：平均亮度 ≤1.3× 相邻段；闪烁检测（任意 1 s 内帧均亮度跳变 ≥5% 的次数 ≤3），超标压对比度
import subprocess
def lum_series(c):
    cache='work/flicker_%d.json'%c
    if os.path.exists(cache): return json.load(open(cache))
    r=CLIPS[c]; p=subprocess.run(['ffmpeg','-v','error','-i','input2/clips/'+r['file'],'-an','-vf','fps=15,crop=iw:ih*0.94:0:ih*0.06,scale=160:-2','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
    h=len(p)//160; hh=int(round(160*int(r['h'])*0.94/int(r['w'])/2))*2
    F=np.frombuffer(p,np.uint8).reshape(-1,hh,160).astype(np.float32)/255; L=F.mean((1,2)).tolist(); json.dump(L,open(cache,'w')); return L
def jumps_per_s(L,thr=0.05,fps=15):
    J=np.abs(np.diff(L)); idx=np.where(J>=thr)[0]; mx=0
    for i in idx: mx=max(mx,int(((idx>=i)&(idx<i+fps)).sum()))
    return mx,J
special_report={}
for s in shots:
    c=s['clip']
    if c not in SPECIAL: continue
    k=s['n_idx']-1; nb=[shots[j]['clip'] for j in (k-1,k+1) if 0<=j<len(shots)]
    nbl=float(np.mean([LUM[x] for x in nb])); gain=min(1.0,1.3*nbl/LUM[c])
    L=lum_series(c); mx,J=jumps_per_s(L); con=1.0
    if mx>3:
        # 以片段均亮度为枢轴压对比度：跳变幅度按 con 缩放，使最坏窗口内第 4 大跳变 < 0.05
        idx=np.where(J>=0.05)[0]; worst=max(idx,key=lambda i:((idx>=i)&(idx<i+15)).sum()); w=np.sort(J[idx[(idx>=worst)&(idx<worst+15)]])[::-1]
        con=min(1.0,0.049/w[3])
    s['fx']=[f'dim:{gain:.2f}']+([f'contrast:{con:.2f}'] if con<1 else [])+s['fx']
    special_report[c]=dict(shot=s['n_idx'],t=round(s['t'],2),seg=s['seg'],lum=round(LUM[c],3),neighbors=nb,neighbor_lum=round(nbl,3),dim_gain=round(gain,3),max_jumps_per_s=int(mx),contrast=round(con,3))
# BGA 重音与最近片段边界的距离
bga_dist=[]
for ev in BGA_EV:
    bd=min(min(abs(s['t']-ev),abs(s['te']-ev)) for s in shots); hit=any(h[0]==ev for h in hits)
    bga_dist.append(dict(event=ev,nearest_boundary_ms=round(bd*1000),hit=hit))
json.dump(dict(beats=[round(float(x),4) for x in BT],per=per,t0=t0,lead=LEAD,bga_target_beats=sorted(bga_target)),open('work/beats_v4.json','w'))
# ---------------- 输出 EDL
cols=['shot','seg','t_start','t_end','dur_ms','beats','beat_start','clip_id','src_in','src_out','speed','win_pct','bga_event','fx','transition','ascii_s','flash']
with open('work/edl.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(cols)
    for s in shots:
        sp=f"ramp:{s['v']:.4f}:hit={s['hit_rel']:.3f}:dip=0.5:w=0.5" if s['mode']=='ramp' else f"{s['v']:.4f}"
        w.writerow([s['n_idx'],s['seg'],f"{s['t']:.3f}",f"{s['te']:.3f}",round(s['dur']*1000),s['n'],s['b'],s['clip'],s['src_in'],s['src_out'],sp,s['win'],s['bga'],' '.join(s['fx']),
                    'ascii' if s['ascii'] else 'cut',s['ascii'],s['flash']])
# ---------------- 统计
vs=np.array([s['v'] for s in shots]); bins=[(0.55,0.85),(0.85,0.9),(0.9,0.95),(0.95,1.05),(1.05,1.1),(1.1,1.151)]
vdist={f'{a}-{b if b<1.15 else 1.15}':int(((vs>=a)&(vs<b)).sum()) for a,b in bins}
byseg=defaultdict(list)
for s in shots: byseg[s['seg']].append(s['n'])
run=1;mx=1
for a,b in zip(shots,shots[1:]):
    run=run+1 if SIM[a['clip']][b['clip']] else 1; mx=max(mx,run)
reasons={}
for c in discarded:
    nb=[sigma[pos[c]+d] for d in (-1,1) if 0<=pos[c]+d<N]
    reasons[c]=('强度分最低档' if S[c]<0.25 else '与相邻画面相似' if any(SIM[c][x] for x in nb) else '时间轴容量不足（速度坡与收束慢放多占约 12 s，DP 按强度错配弃用）')+f'（S={S[c]:.2f}）'
for c in sorted(DROP): reasons[c]=OV['drop_reason'].get(str(c),'指定弃用')
src_used=sum(s['d'] for s in shots)
stats=dict(fps=FPS,beats_total=NB,grid=dict(bpm=60/per,t0=t0,beat0=round(float(BT[0]),3),beat830=round(float(BT[NB]),3)),
    clips_used=len(shots),clips_discarded=sorted(discarded)+sorted(DROP),discard_reasons={str(k):v for k,v in reasons.items()},
    avg_beats=round(float(np.mean([s['n'] for s in shots])),2),beats_dist=dict(sorted(Counter(s['n'] for s in shots).items())),
    avg_beats_by_seg={k:round(float(np.mean(v)),2) for k,v in byseg.items()},n_by_seg={k:len(v) for k,v in byseg.items()},
    speed_dist=vdist,speed_mean=round(float(vs.mean()),3),ramps=sum(1 for s in shots if s['mode']=='ramp'),
    utilization=round(src_used/304.5,3),src_used_s=round(src_used,1),
    bga_hits=len(hits),bga_hit_list=hits,bga_beat_info=bga_info,bga_targets_on_grid=len(bga_target),bga_accent_to_boundary=bga_dist,special=special_report,
    ramps_by_seg=dict(Counter(s['seg'] for s in shots if s['mode']=='ramp')),outro=[(s['clip'],s['n'],round(s['v'],3)) for s in shots if s['mode']=='slow'],
    max_similar_run=mx,n_flash=sum(s['flash'] for s in shots),ascii_total=tot,
    first_clip_start=round(shots[0]['t'],3),last_clip_end=round(shots[-1]['te'],3),
    shot_len_s=dict(min=round(min(s['dur'] for s in shots),2),max=round(max(s['dur'] for s in shots),2),mean=round(float(np.mean([s['dur'] for s in shots])),2)))
json.dump(stats,open('work/plan_v4_stats.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(stats,ensure_ascii=False))
