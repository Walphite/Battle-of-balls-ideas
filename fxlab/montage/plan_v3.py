"""第 2 步方案生成（不渲染）：整首 278.8s，输出 work/edl.csv 与 ../docs/PLAN_MONTAGE_V3.md。
输入：work/clips.csv, work/clip_feats.json, work/audio_an.pkl, work/an4.pkl, work/vocal_onsets.json, work/bga_accent.json"""
import csv,json,pickle,os,sys,numpy as np
from collections import Counter,defaultdict
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(7)
IRR_TARGET=int(os.environ.get('IRR_TARGET','80'))
FPS=60; FR=1/FPS; T=278.806
A=pickle.load(open('work/audio_an.pkl','rb')); P=pickle.load(open('work/an4.pkl','rb'))
VO=json.load(open('work/vocal_onsets.json')); ACC=json.load(open('work/bga_accent.json'))
AFPS=22050/256
per=A['per']; q=per/2; t0=A['t0']
CLIPS={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}
FE={f['clip_id']:f for f in json.load(open('work/clip_feats.json'))}
INSET=0.15; GAP=float(os.environ.get('GAP','0.1')); LEAD=0.025   # 切点提前 25ms
BGA_EV=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]
SEGS=[(0,16,'intro'),(16,79,'regular'),(79,100,'peak'),(100,188,'irregular'),(188,202,'sparse'),(202,219.5,'frag'),(219.5,230,'peak'),(230,266,'after'),(266,T,'outro')]
DRIFT={'intro':0.044,'regular':0.039,'peak':0.031,'irregular':0.022,'sparse':0.003,'frag':0.003,'after':-0.004,'outro':-0.013}
def seg_of(t):
    for a,b,n in SEGS:
        if a<=t<b: return n
    return 'outro'
# ---------- 能量 / 起音密度（配乐时间轴）
def smooth(a,s): w=max(1,int(s*AFPS)); return np.convolve(a,np.ones(w)/w,'same')
def n01(a): lo,hi=np.percentile(a,2),np.percentile(a,98); return np.clip((a-lo)/(hi-lo+1e-9),0,1)
E_=n01(smooth(A['ai'],1.0)); D_=n01(smooth(A['dens'],1.0))
INT=0.6*E_+0.4*D_   # 特效强度驱动：平滑能量 + 起音密度
def at(a,t): return float(a[min(len(a)-1,max(0,int(t*AFPS)))])
PK=A['pk']  # 全部起音
CORP=[a for a,_ in VO['corp_100_188']]; FRAG=VO['frag_merged']; EARLY=[a for a,_ in VO['early_old']]
VOCAL=sorted(CORP+FRAG+EARLY)
# ---------- 网格
def grid(t,eighths=1):
    """返回 ≥t 的下一个八分拍（含段落相位偏移）"""
    d=DRIFT[seg_of(t)]; k=np.ceil((t-t0-d)/q-1e-6); return t0+d+k*q
# ---------- 切点生成
cuts=[]  # (time, kind, trigger)
def add(t,kind,trig=''):
    t=round(t,4)
    if cuts and t-cuts[-1][0]<3*FR-1e-6: return
    cuts.append((t,kind,trig))
add(0,'intro','start')
t=0.0
# intro 0–16：长镜头 12–16 个八分拍
while True:
    L=16 if at(E_,t)<0.15 else 12
    nt=grid(t+L*q-q/2)-LEAD
    if nt>=16: break
    add(nt,'intro','grid'); t=nt
# regular 16–79：随时间+能量 8→4→2（八分拍）
t=grid(16)-LEAD; add(t,'regular','seg')
while True:
    frac=(t-16)/63; lvl=0.55*frac+0.45*at(E_,t)
    L=8 if lvl<0.4 else (4 if lvl<0.8 else 2)
    nt=grid(t+L*q-q/2)-LEAD
    if nt>=79: break
    add(nt,'regular','grid'); t=nt
# peak 79–100：2/1 个八分拍，强起音处半个八分拍
def peak_cuts(a,b):
    t=grid(a)-LEAD; add(t,'peak','seg')
    while True:
        e=at(E_,t); d=at(D_,t)
        if e>0.92 and d>0.85 and rng.random()<0.12: L=0.5
        elif e>0.92 and rng.random()<0.5: L=1
        else: L=2
        nt=grid(t+L*q-q/2)-LEAD if L>=1 else t+q/2
        if nt>=b-0.05: break
        add(nt,'peak','grid' if L>=1 else 'half'); t=nt
peak_cuts(79,100)
# irregular 100–188：按起音切，3–12 帧；低密度（30 分位以下）≥0.5s；每 ≤12s 插一个 ≥0.5s 呼吸镜头
dens_reg=np.array([at(D_,x) for x in np.arange(100,188,0.1)]); p30=np.percentile(dens_reg,30)
ENV=A['env']; 
def ostr(o): return float(ENV[min(len(ENV)-1,int(o*AFPS))])
allo=PK[(PK>=100)&(PK<188)]; st=np.array([ostr(o) for o in allo])
thr=np.sort(st)[::-1][min(len(st)-1,IRR_TARGET)]          # 取最强 IRR_TARGET 个起音
ons=np.array(sorted(set(np.round(allo[st>=thr],3).tolist()+[round(c,3) for c in CORP])))
add(100,'irregular','seg'); t=100.0; last_breath=100.0
for o in ons:
    lo=at(D_,o)<p30
    minlen=0.5 if lo else 3*FR
    if o-t<minlen: continue
    if o-last_breath>11.0 and not lo:
        add(o-0.01,'irregular','breath'); t=o-0.01; last_breath=t; continue
    if cuts[-1][2]=='breath' and o-t<0.5: continue
    trig='corp' if any(abs(o-c)<0.02 for c in CORP) else 'onset'
    add(o-0.01,'irregular',trig); t=o-0.01
    if lo: last_breath=t
# sparse 188–202：8 个八分拍
t=grid(188)-LEAD; add(t,'sparse','seg')
while True:
    nt=grid(t+8*q-q/2)-LEAD
    if nt>=202: break
    add(nt,'sparse','grid'); t=nt
# frag 202–219.5：每个碎片起音硬切；碎片间长空档按 4 个八分拍补切
add(202.0,'frag','seg'); t=202.0
for o in FRAG:
    while o-t>4*q+0.05:
        nt=grid(t+4*q-q/2)-LEAD; 
        if nt>=o-0.1: break
        add(nt,'frag','grid'); t=nt
    if o-t>=3*FR: add(o-0.01,'frag','frag'); t=o-0.01
peak_cuts(219.5,230)
# after 230–266：4–8 个八分拍
t=grid(230)-LEAD; add(t,'after','seg')
while True:
    L=4 if at(E_,t)>0.55 else 8
    nt=grid(t+L*q-q/2)-LEAD
    if nt>=266: break
    add(nt,'after','grid'); t=nt
# outro 266–end：3 个慢放长镜头
for x in (266.0,270.5,275.0): add(x,'outro','seg')
# BGA 对位事件：强制硬切（若与既有切点差 >3 帧则插入）
for ev in BGA_EV:
    near=min(cuts,key=lambda c:abs(c[0]-ev))
    if abs(near[0]-ev)>3*FR: cuts.append((round(ev-LEAD,4),seg_of(ev),'bga'))
    else: cuts[cuts.index(near)]=(near[0],near[1],'bga')
cuts.sort()
# 合并 <3 帧
clean=[cuts[0]]
for c in cuts[1:]:
    if c[0]-clean[-1][0]<3*FR-1e-6:
        if c[2] in('bga','corp','frag'): clean[-1]=c
        continue
    clean.append(c)
cuts=clean
shots=[]
for i,(ts,kind,trig) in enumerate(cuts):
    te=cuts[i+1][0] if i+1<len(cuts) else T
    shots.append(dict(n=i+1,seg=kind,t=ts,te=te,dur=te-ts,trig=trig))
# ---------- 速度 / 取景窗 / 特效
def win_pct(t,seg):
    e=at(E_,t)
    if seg=='intro': return 75+25*min(1,e/0.3)
    if seg=='regular': return 75+25*min(1,0.3+e)
    if seg in('peak','irregular','frag'): return 100
    if seg=='sparse': return 92
    if seg=='after': return 96
    return 100-25*min(1,(t-266)/9)
ACCENT_ROT=['rgb','block']
ascii_budget=10.0; ascii_used=0.0
flash_times=[]
def fx_for(s,prev):
    seg=s['seg']; t=s['t']; e=at(INT,t); fx=[]; tr='cut'
    lvl=round(e,2)
    if seg=='intro': fx=[f'scan:{0.1+0.1*e:.2f}','vignette:0.3']; 
    elif seg=='regular':
        frac=(t-16)/63
        fx=[f'scan:{0.15+0.2*e:.2f}','vignette:0.3']
        if frac>0.3 or e>0.5: fx.append(f'zoompulse:{min(6,2+6*e):.1f}%')
        if frac>0.6 and e>0.45: fx.append('rgb:2f')
    elif seg=='peak':
        fx=[f'scan:{0.3:.2f}',f'zoompulse:{min(6,3+4*e):.1f}%','rgb:3f',f'shake:{min(1.5,1.5*e):.1f}%','noise:0.15']
        if s['trig']=='half': fx.append('block:3f')
    elif seg=='irregular':
        fx=[f'scan:{0.2+0.2*e:.2f}','noise:0.12']
        if s['trig']=='corp': fx.append('stutter:3f'); fx.append('rgb:2f')
        elif s['dur']<0.25: fx.append(rng.choice(['rgb:2f','block:3f','shake:1%']))
        if s['trig']=='breath' or s['dur']>=0.5: fx.append('speed:0.5x' if e<0.5 else 'speed:0.75x')
    elif seg=='sparse': fx=[f'scan:{0.25:.2f}','vignette:0.6','noise:0.2',f'desat:0.2']
    elif seg=='frag':
        fx=['scan:0.3','noise:0.2','vignette:0.4']
        if s['trig']=='frag': fx.append(rng.choice(['block:4f','tear:4f']))
    elif seg=='after': fx=['scan:0.25','vignette:0.45',f'zoompulse:{2+2*e:.1f}%','tone:cold']
    else: fx=[f'scan:{0.4+0.3*(t-266)/12:.2f}',f'noise:{0.25+0.3*(t-266)/12:.2f}','vignette:0.6','speed:0.5x','tone:cold']
    if s['trig']=='bga': fx.append(ACCENT_ROT[BGA_EV.index(min(BGA_EV,key=lambda v:abs(v-t)))%2]+':4f'); fx.append('bga_accent')
    return fx,lvl
# ASCII 转场：frag 区 10 次 ×0.5s；峰值各 2 次 ×0.6；BGA 事件 144.3/187.2 各 0.6
ascii=[]
fr=[s for s in shots if s['seg']=='frag' and s['trig']=='frag' and s['dur']>=0.5]
for s in fr[:10]: ascii.append((s['n'],0.5))
for seg,a,b in (('peak',79,100),('peak',219.5,230)):
    c=[s for s in shots if s['seg']==seg and a<=s['t']<b and s['dur']>=0.6][::6][:2]
    for s in c: ascii.append((s['n'],0.6))
for ev in (144.31,187.21):
    s=min(shots,key=lambda s:abs(s['t']-ev)); ascii.append((s['n'],0.6))
ascii_map={n:d for n,d in ascii}; ascii_total=sum(d for _,d in ascii)
assert ascii_total<=10
# ---------- 闪白：仅峰值区小节首拍，每 2 小节 1 次，峰值 ≤60%
flash=set()
for s in shots:
    if s['seg']=='peak' and s['trig']=='grid':
        k=round((s['t']+LEAD-t0-DRIFT['peak'])/q)
        if k%16==0 and at(E_,s['t'])>0.7: flash.add(s['n'])
# ---------- 素材分配
rank=lambda v:(np.argsort(np.argsort(v))/(len(v)-1))
ids=sorted(FE); M=np.array([FE[i]['motion'] for i in ids]); S=np.array([FE[i]['sat'] for i in ids]); V=np.array([FE[i]['motion_std'] for i in ids])
strength=dict(zip(ids,0.45*rank(M)+0.3*rank(S)+0.25*rank(V)))
LUM={i:FE[i]['lum'] for i in ids}; HUE={i:np.array(FE[i]['hue_hist']) for i in ids}
def similar(a,b):
    h=HUE[a]@HUE[b]/(np.linalg.norm(HUE[a])*np.linalg.norm(HUE[b])+1e-9); return h>0.93 and abs(LUM[a]-LUM[b])<0.04
used=defaultdict(list)   # clip -> [(in,out)]
last_t=defaultdict(lambda:-99); uses=Counter()
def free_window(c,need,prefer_motion):
    d=float(CLIPS[c]['dur']); lo,hi=INSET,d-INSET
    iv=sorted(used[c]); free=[]; cur=lo
    for a,b in iv:
        if a-GAP-cur>=need: free.append((cur,a-GAP))
        cur=max(cur,b+GAP)
    if hi-cur>=need: free.append((cur,hi))
    if not free: return None
    ml=FE[c]['motion_local']; best=None
    for a,b in free:
        for st in {a,b-need}:
            k=slice(int(st*2),max(int(st*2)+1,int((st+need)*2))); m=np.mean(ml[k]) if ml[k] else 0
            sc=(m if prefer_motion else -m)-0.5*(b-a-need)   # 优先填满短空闲区
            if best is None or sc>best[0]: best=(sc,st)
    return (best[1],best[1]+need)
ONCE={97,109}
SLOW_AFTER=os.environ.get('SLOW_AFTER','1')=='1'
appear=defaultdict(list)  # clip -> [(t,te)]
MINGAP=float(os.environ.get('MINGAP','3.0'))
def gap_ok(c,s): return all(s['t']>=b+MINGAP or s['te']<=a-MINGAP for a,b in appear[c])
for s in shots:
    s['fx'],s['lvl']=fx_for(s,None)
    if os.environ.get('INTRO_SLOW','1')=='1' and s['seg']=='intro': s['fx'].append('speed:0.75x')
    if os.environ.get('REG_SLOW','1')=='1' and s['seg']=='regular' and s['dur']>=1.2: s['fx'].append('speed:0.75x')
    if s['seg']=='after' and s['t']>=256: s['fx']=[f for f in s['fx'] if not f.startswith('speed')]+['speed:0.5x']   # 收束前 10s 速度坡
    if SLOW_AFTER and s['seg'] in('sparse','after') and s['dur']>=0.6 and not any(f.startswith('speed') for f in s['fx']): s['fx'].append('speed:0.75x')
    s['speed']=0.5 if 'speed:0.5x' in s['fx'] else (0.75 if 'speed:0.75x' in s['fx'] else 1.0)
    s['need']=s['dur']*s['speed']+0.02
def assign(s,maxu):
    seg=s['seg']; need=s['need']; want=at(INT,s['t']); short=s['dur']<0.25
    prev=[x['clip'] for x in shots[max(0,s['n']-3):s['n']-1] if x.get('clip') is not None]
    nxt=[x['clip'] for x in shots[s['n']:s['n']+2] if x.get('clip') is not None]
    cands=[]
    for c in ids:
        u=uses[c]
        if c in ONCE and (u>=1 or s['dur']>0.4 or seg in('intro','outro') or s['n'] in flash): continue
        if u>=maxu: continue
        if not gap_ok(c,s): continue
        if short and prev and abs(LUM[c]-LUM[prev[-1]])>0.25*max(LUM[prev[-1]],0.2): continue
        if len(prev)>=2 and similar(prev[-1],prev[-2]) and similar(c,prev[-1]): continue
        if len(nxt)>=2 and similar(nxt[0],nxt[1]) and similar(c,nxt[0]): continue
        w=free_window(c,need,want>0.5)
        if w is None: continue
        score=abs(strength[c]-want)+1.0*u+(0.3 if any(similar(c,p) for p in prev+nxt) else 0)
        cands.append((score,c,w))
    if not cands: return False
    cands.sort(key=lambda x:x[0]); sc,c,w=cands[0]
    s['clip']=c; s['src_in'],s['src_out']=w; used[c].append(w); uses[c]+=1; appear[c].append((s['t'],s['te']))
    s['variant']='' if uses[c]==1 else f"zoom{int(rng.integers(115,141))}%@{rng.choice(['TL','TR','BL','BR','C'])}"
    return True
FRAGSEG=('peak','irregular','frag')
for s in shots:
    s['win']=round(win_pct(s['t'],s['seg'])); s['ascii']=ascii_map.get(s['n'],0); s['flash']=int(s['n'] in flash); s['clip']=None
# 第零遍：长窗口（≥1.5s 源素材）先占位
for s in shots:
    if s['need']>=1.5: assign(s,2)
# 第一遍：非碎切段 + 碎切段里 ≥0.5s 的镜头，窗口 ≤2
for s in shots:
    if s['clip'] is not None: continue

    if s['seg'] not in FRAGSEG or s['dur']>=0.5: assign(s,2)
# 第二遍：碎切段其余镜头，窗口 ≤4
for s in shots:
    if s['clip'] is None and s['seg'] in FRAGSEG: assign(s,4)
# 第三遍：仍空的，放宽到 4
MAXW=int(os.environ.get('MAXW','5'))
for s in shots:
    if s['clip'] is None: assign(s,MAXW)
# 第四遍（超预算）：仍空的镜头允许第 6 个窗口，并在 EDL 标记 over_budget
OVER=[]
for s in shots:
    if s['clip'] is None and assign(s,6): OVER.append(s['n']); s['over']=1

# 97/109：前后 0.5s 无闪白
for s in shots:
    if s.get('clip') in ONCE:
        for x in shots:
            if abs(x['t']-s['t'])<0.5+s['dur'] and x.get('flash'): x['flash']=0
# ---------- 输出 EDL
cols=['over_budget','shot','seg','t_start','t_end','dur_ms','frames','eighths','clip_id','src_in','src_out','speed','win_pct','variant','trigger','fx','transition','ascii_s','flash']
with open('work/edl.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(cols)
    for s in shots:
        w.writerow([s.get('over',0),s['n'],s['seg'],f"{s['t']:.3f}",f"{s['te']:.3f}",round(s['dur']*1000),round(s['dur']*FPS),round(s['dur']/q,2),s.get('clip'),
                    f"{s.get('src_in',0):.3f}",f"{s.get('src_out',0):.3f}",s.get('speed',1),s.get('win',100),s.get('variant',''),s['trig'],' '.join(s['fx']),
                    'ascii' if s.get('ascii') else ('fadein' if s['n']==1 else 'cut'),s.get('ascii',0),s.get('flash',0)])
# ---------- 统计
unassigned=[s['n'] for s in shots if s.get('clip') is None]
byseg=defaultdict(list)
for s in shots: byseg[s['seg']].append(s['dur'])
src_used=sum(b-a for c in used for a,b in used[c])
dist=Counter(uses[c] for c in ids)
unused=[c for c in ids if uses[c]==0]
# 闪白/秒
ft=sorted(s['t'] for s in shots if s.get('flash'))
maxf=max((sum(1 for x in ft if t<=x<t+1) for t in ft),default=0)
# 相似连续
run=1;mx=1
for a,b in zip(shots,shots[1:]):
    if a.get('clip') and b.get('clip') and similar(a['clip'],b['clip']): run+=1; mx=max(mx,run)
    else: run=1
# 短镜头亮度差
viol=[s['n'] for a,s in zip(shots,shots[1:]) if s['dur']<0.25 and a.get('clip') and s.get('clip') and abs(LUM[s['clip']]-LUM[a['clip']])>0.25*max(LUM[a['clip']],0.2)]
corp_in=sum(1 for s in shots if s['trig']=='corp'); frag_in=sum(1 for s in shots if s['trig']=='frag')
need_total=round(sum(s['need'] for s in shots),1); supply=round(sum(float(CLIPS[c]['dur'])-2*INSET for c in ids),1)
stats=dict(need_total=need_total,supply_raw=supply,gaps_est=round((len(shots)-len(ids))*GAP,1),over_budget=OVER,shots=len(shots),unassigned=unassigned,avg_len={k:round(np.mean(v),3) for k,v in byseg.items()},n_seg={k:len(v) for k,v in byseg.items()},
           utilization=round(src_used/304.5,3),src_used=round(src_used,1),reuse_dist=dict(sorted(dist.items())),unused=unused,max_flash_per_s=maxf,n_flash=len(ft),
           max_similar_run=mx,short_lum_viol=viol,ascii_total=ascii_total,n_ascii=len(ascii),corp_cuts=corp_in,frag_cuts=frag_in,
           bga_events=[(ev,next(s['n'] for s in shots if s['trig']=='bga' and abs(s['t']+LEAD-ev)<0.06)) for ev in BGA_EV],
           min_shot_frames=min(round(s['dur']*FPS) for s in shots),max_shot=round(max(s['dur'] for s in shots),2))
json.dump(stats,open('work/plan_v3_stats.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(stats,ensure_ascii=False))
