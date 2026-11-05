"""第 2 步方案生成（不渲染）：整首 278.8s，输出 work/edl.csv 与 work/plan_v3_stats.json。
输入：work/clips.csv, work/clip_feats.json, work/audio_an.pkl, work/an4.pkl, work/vocal_onsets.json, work/bga_accent.json
规则见 ../docs/PLAN_MONTAGE_V3.md。"""
import csv,json,pickle,os,sys,subprocess,numpy as np
from collections import Counter,defaultdict
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(7)
FPS=30; FR=1/FPS; T=278.806; MINF=3*FR            # 最短镜头 3 帧 @30fps
A=pickle.load(open('work/audio_an.pkl','rb')); P=pickle.load(open('work/an4.pkl','rb'))
VO=json.load(open('work/vocal_onsets.json'))
AFPS=22050/256
per=A['per']; q=per/2; t0=A['t0']
CLIPS={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}
FE={f['clip_id']:f for f in json.load(open('work/clip_feats.json'))}
INSET=0.15; GAP=0.1; LEAD=0.025; MAXW=5; MINGAP=3.0; CROP_TOP=0.06; IRR_TARGET=80
BGA_EV=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]
SEGS=[(0,16,'intro'),(16,79,'regular'),(79,100,'peak'),(100,188,'irregular'),(188,202,'sparse'),(202,219.5,'frag'),(219.5,230,'peak'),(230,266,'after'),(266,T,'outro')]
DRIFT={'intro':0.044,'regular':0.039,'peak':0.031,'irregular':0.022,'sparse':0.003,'frag':0.003,'after':-0.004,'outro':-0.013}
def seg_of(t):
    for a,b,n in SEGS:
        if a<=t<b: return n
    return 'outro'
def smooth(a,s): w=max(1,int(s*AFPS)); return np.convolve(a,np.ones(w)/w,'same')
def n01(a): lo,hi=np.percentile(a,2),np.percentile(a,98); return np.clip((a-lo)/(hi-lo+1e-9),0,1)
E_=n01(smooth(A['ai'],1.0)); D_=n01(smooth(A['dens'],1.0))
INT=0.6*E_+0.4*D_
def at(a,t): return float(a[min(len(a)-1,max(0,int(t*AFPS)))])
PK=A['pk']; ENV=A['env']
def ostr(o): return float(ENV[min(len(ENV)-1,int(o*AFPS))])
CORP=[a for a,_ in VO['corp_100_188']]; FRAG=VO['frag_merged']
def grid(t):
    d=DRIFT[seg_of(t)]; k=np.ceil((t-t0-d)/q-1e-6); return t0+d+k*q
# ---------------- 切点
cuts=[]
def add(t,kind,trig=''):
    t=round(t,4)
    if cuts and t-cuts[-1][0]<MINF-1e-6: return
    cuts.append((t,kind,trig))
add(0,'intro','start'); t=0.0
while True:
    L=16 if at(E_,t)<0.15 else 12
    nt=grid(t+L*q-q/2)-LEAD
    if nt>=16: break
    add(nt,'intro','grid'); t=nt
t=grid(16)-LEAD; add(t,'regular','seg')
while True:
    frac=(t-16)/63; lvl=0.55*frac+0.45*at(E_,t)
    L=8 if lvl<0.4 else (4 if lvl<0.8 else 2)
    nt=grid(t+L*q-q/2)-LEAD
    if nt>=79: break
    add(nt,'regular','grid'); t=nt
def peak_cuts(a,b):
    t=grid(a)-LEAD; add(t,'peak','seg')
    while True:
        e=at(E_,t)
        L=1 if (e>0.92 and rng.random()<0.5) else 2      # 30fps 下不用半个八分拍（84ms<3帧）
        nt=grid(t+L*q-q/2)-LEAD
        if nt>=b-0.05: break
        add(nt,'peak','grid'); t=nt
peak_cuts(79,100)
dens_reg=np.array([at(D_,x) for x in np.arange(100,188,0.1)]); p30=np.percentile(dens_reg,30)
allo=PK[(PK>=100)&(PK<188)]; st=np.array([ostr(o) for o in allo])
thr=np.sort(st)[::-1][min(len(st)-1,IRR_TARGET)]
ons=np.array(sorted(set(np.round(allo[st>=thr],3).tolist()+[round(c,3) for c in CORP])))
add(100,'irregular','seg'); t=100.0; last_breath=100.0
for o in ons:
    lo=at(D_,o)<p30
    minlen=0.5 if lo else MINF
    if o-t<minlen: continue
    if o-last_breath>11.0 and not lo:
        add(o-0.01,'irregular','breath'); t=o-0.01; last_breath=t; continue
    if cuts[-1][2]=='breath' and o-t<0.5: continue
    trig='corp' if any(abs(o-c)<0.02 for c in CORP) else 'onset'
    add(o-0.01,'irregular',trig); t=o-0.01
    if lo: last_breath=t
# 不规则区低密度长空档（>2.0s）按 8 个八分拍补切（慢放长镜头，不算碎切）
extra=[]
for (ta,ka,ga),(tb,kb,gb) in zip(cuts,cuts[1:]):
    if ka=='irregular' and tb-ta>2.0:
        x=grid(ta+8*q-q/2)-LEAD
        while tb-x>=1.0:
            extra.append((round(x,4),'irregular','grid')); x=grid(x+8*q-q/2)-LEAD
cuts+=extra; cuts.sort()
t=grid(188)-LEAD; add(t,'sparse','seg')
while True:
    nt=grid(t+8*q-q/2)-LEAD
    if nt>=202: break
    add(nt,'sparse','grid'); t=nt
add(202.0,'frag','seg'); t=202.0
for o in FRAG:
    while o-t>4*q+0.05:
        nt=grid(t+4*q-q/2)-LEAD
        if nt>=o-0.1: break
        add(nt,'frag','grid'); t=nt
    if o-t>=MINF: add(o-0.01,'frag','frag'); t=o-0.01
peak_cuts(219.5,230)
t=grid(230)-LEAD; add(t,'after','seg')
while True:
    L=4 if at(E_,t)>0.55 else 8
    nt=grid(t+L*q-q/2)-LEAD
    if nt>=266: break
    add(nt,'after','grid'); t=nt
for x in (266.0,270.5,275.0): add(x,'outro','seg')
for ev in BGA_EV:
    near=min(cuts,key=lambda c:abs(c[0]-ev))
    if abs(near[0]-ev)>3*FR: cuts.append((round(ev-LEAD,4),seg_of(ev),'bga'))
    else: cuts[cuts.index(near)]=(near[0],near[1],'bga')
cuts.sort()
clean=[cuts[0]]
for c in cuts[1:]:
    if c[0]-clean[-1][0]<MINF-1e-6:
        if c[2] in('bga','corp','frag'): clean[-1]=c
        continue
    clean.append(c)
cuts=clean
shots=[]
for i,(ts,kind,trig) in enumerate(cuts):
    te=cuts[i+1][0] if i+1<len(cuts) else T
    shots.append(dict(n=i+1,seg=kind,t=ts,te=te,dur=te-ts,trig=trig))
# ---------------- 速度 / 取景窗 / 特效
def win_pct(t,seg):
    e=at(E_,t)
    if seg=='intro': return 75+25*min(1,e/0.3)
    if seg=='regular': return 75+25*min(1,0.3+e)
    if seg in('peak','irregular','frag'): return 100
    if seg=='sparse': return 92
    if seg=='after': return 96
    return 100-25*min(1,(t-266)/9)
ACCENT_ROT=['rgb','block']
# 余波速度坡：每 4 小节（16 拍=5.36s）最多 1 次，≤1s 镜头，0.5→1→0.5
ramp_ids=set(); last_ramp=-99
for s in shots:
    if s['seg']=='after' and s['dur']<=1.0 and s['t']-last_ramp>=16*per and at(E_,s['t'])>0.5:
        ramp_ids.add(s['n']); last_ramp=s['t']
def fx_for(s):
    seg=s['seg']; t=s['t']; e=at(INT,t); fx=[]
    if seg=='intro': fx=[f'scan:{0.1+0.1*e:.2f}','vignette:0.3','speed:0.75x']
    elif seg=='regular':
        frac=(t-16)/63
        fx=[f'scan:{0.15+0.2*e:.2f}','vignette:0.3']
        if frac>0.3 or e>0.5: fx.append(f'zoompulse:{min(6,2+6*e):.1f}%')
        if frac>0.6 and e>0.45: fx.append('rgb:2f')
        if s['dur']>=1.2: fx.append('speed:0.75x')
    elif seg=='peak':
        fx=['scan:0.30',f'zoompulse:{min(6,3+4*e):.1f}%','rgb:3f',f'shake:{min(1.5,1.5*e):.1f}%','noise:0.15']
    elif seg=='irregular':
        fx=[f'scan:{0.2+0.2*e:.2f}','noise:0.12']
        if s['trig']=='corp': fx+=['stutter:3f','rgb:2f']
        elif s['dur']<0.25: fx.append(str(rng.choice(['rgb:2f','block:3f','shake:1%'])))
        else:
            # 簇外镜头：在镜头内最强起音处加缩放脉冲或色差（镜头内偏移秒）
            m=PK[(PK>=s['t']+0.05)&(PK<s['te']-0.05)]
            if len(m):
                o=m[np.argmax([ostr(x) for x in m])]; off=o-s['t']
                fx.append((f'zoompulse:{min(6,3+4*e):.1f}%' if rng.random()<0.5 else 'rgb:3f')+f'@{off:.3f}')
        if s['trig'] in('breath','grid') or s['dur']>=0.5: fx.append('speed:0.5x' if (e<0.5 or s['trig']=='grid') else 'speed:0.75x')
    elif seg=='sparse': fx=['scan:0.25','vignette:0.6','noise:0.2','desat:0.2']+(['speed:0.75x'] if s['dur']>=0.6 else [])
    elif seg=='frag':
        fx=['scan:0.3','noise:0.2','vignette:0.4']
        if s['trig']=='frag': fx.append(str(rng.choice(['block:4f','tear:4f'])))
    elif seg=='after':
        fx=['scan:0.25','vignette:0.45',f'zoompulse:{2+2*e:.1f}%','tone:cold']
        fx.append('speed:ramp0.5-1-0.5' if s['n'] in ramp_ids else 'speed:0.5x')
    else: fx=[f'scan:{0.4+0.3*(t-266)/12:.2f}',f'noise:{0.25+0.3*(t-266)/12:.2f}','vignette:0.6','speed:0.5x','tone:cold']
    if s['trig']=='bga': fx+=[ACCENT_ROT[BGA_EV.index(min(BGA_EV,key=lambda v:abs(v-t)))%2]+':4f','bga_accent']
    return fx,round(e,2)
ascii=[]
fr=[s for s in shots if s['seg']=='frag' and s['trig']=='frag' and s['dur']>=0.5]
for s in fr[:10]: ascii.append((s['n'],0.5))
for seg,a,b in (('peak',79,100),('peak',219.5,230)):
    c=[s for s in shots if s['seg']==seg and a<=s['t']<b and s['dur']>=0.6][::6][:2]
    for s in c: ascii.append((s['n'],0.6))
for ev in (144.31,187.21):
    s=min(shots,key=lambda s:abs(s['t']-ev)); ascii.append((s['n'],0.6))
ascii_map={n:d for n,d in ascii}; ascii_total=sum(d for _,d in ascii); assert ascii_total<=10
flash=set()
for s in shots:
    if s['seg']=='peak' and s['trig']=='grid':
        k=round((s['t']+LEAD-t0-DRIFT['peak'])/q)
        if k%16==0 and at(E_,s['t'])>0.7: flash.add(s['n'])
for s in shots:
    s['fx'],s['lvl']=fx_for(s)
    sp=[f for f in s['fx'] if f.startswith('speed')]
    s['speed']=('ramp' if 'ramp' in sp[0] else float(sp[0][6:-1])) if sp else 1.0
    s['need']=s['dur']*(0.75 if s['speed']=='ramp' else s['speed'])+0.02
    s['win']=round(win_pct(s['t'],s['seg'])); s['ascii']=ascii_map.get(s['n'],0); s['flash']=int(s['n'] in flash); s['clip']=None
# ---------------- 素材分配
rank=lambda v:(np.argsort(np.argsort(v))/(len(v)-1))
ids=sorted(FE); M=np.array([FE[i]['motion'] for i in ids]); S=np.array([FE[i]['sat'] for i in ids]); V=np.array([FE[i]['motion_std'] for i in ids])
strength=dict(zip(ids,0.45*rank(M)+0.3*rank(S)+0.25*rank(V)))
LUM={i:FE[i]['lum'] for i in ids}; HUE={i:np.array(FE[i]['hue_hist']) for i in ids}
def similar(a,b):
    h=HUE[a]@HUE[b]/(np.linalg.norm(HUE[a])*np.linalg.norm(HUE[b])+1e-9); return h>0.93 and abs(LUM[a]-LUM[b])<0.04
used=defaultdict(list); uses=Counter(); appear=defaultdict(list); ONCE={97,109}
def gap_ok(c,s): return all(s['t']>=b+MINGAP or s['te']<=a-MINGAP for a,b in appear[c])
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
            sc=(m if prefer_motion else -m)-0.5*(b-a-need)
            if best is None or sc>best[0]: best=(sc,st)
    return (best[1],best[1]+need)
def neighbors(s):
    prev=[x['clip'] for x in shots[max(0,s['n']-3):s['n']-1] if x.get('clip') is not None]
    nxt=[x['clip'] for x in shots[s['n']:s['n']+2] if x.get('clip') is not None]
    return prev,nxt
def compatible(c,s,prev,nxt):
    short=s['dur']<0.25
    if short and prev and abs(LUM[c]-LUM[prev[-1]])>0.25*max(LUM[prev[-1]],0.2): return False
    if len(prev)>=2 and similar(prev[-1],prev[-2]) and similar(c,prev[-1]): return False
    if len(nxt)>=2 and similar(nxt[0],nxt[1]) and similar(c,nxt[0]): return False
    return True
def assign(s,maxu):
    seg=s['seg']; need=s['need']; want=at(INT,s['t']); prev,nxt=neighbors(s); cands=[]
    for c in ids:
        u=uses[c]
        if c in ONCE and (u>=1 or s['dur']>0.4 or seg in('intro','outro') or s['flash']): continue
        if u>=maxu or not gap_ok(c,s) or not compatible(c,s,prev,nxt): continue
        w=free_window(c,need,want>0.5)
        if w is None: continue
        cands.append((abs(strength[c]-want)+1.0*u+(0.3 if any(similar(c,p) for p in prev+nxt) else 0),c,w))
    if not cands: return False
    cands.sort(key=lambda x:x[0]); _,c,w=cands[0]
    s['clip']=c; s['src_in'],s['src_out']=w; used[c].append(w); uses[c]+=1; appear[c].append((s['t'],s['te']))
    s['variant']='' if uses[c]==1 else f"zoom{int(rng.integers(115,141))}%@{rng.choice(['TL','TR','BL','BR','C'])}"
    return True
FRAGSEG=('peak','irregular','frag')
for s in shots:
    if s['need']>=1.5: assign(s,2)
for s in shots:
    if s['clip'] is None and (s['seg'] not in FRAGSEG or s['dur']>=0.5): assign(s,2)
for s in shots:
    if s['clip'] is None and s['seg'] in FRAGSEG: assign(s,4)
for s in shots:
    if s['clip'] is None: assign(s,MAXW)
# ---------------- 取景变体填充：复用已用窗口（次数最少优先），放大 120–180%，裁切位取运动中心附近不同区域
def motion_center(c,a,b):
    """低分辨率解码该窗口，帧差加权质心（归一化 0–1，已排除顶部裁切区）"""
    p=subprocess.run(['ffmpeg','-v','error','-ss',f'{a:.3f}','-t',f'{max(b-a,0.3):.3f}','-i','input2/clips/'+CLIPS[c]['file'],'-an','-vf','fps=10,scale=96:-2','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
    hh=int(round(96*int(CLIPS[c]['h'])/int(CLIPS[c]['w'])/2))*2
    F=np.frombuffer(p,np.uint8).reshape(-1,hh,96).astype(np.float32) if len(p)>=2*hh*96 else None
    if F is None: return (0.5,0.55)
    hh=F.shape[1]; F=F[:,int(hh*CROP_TOP):]; d=np.abs(np.diff(F,axis=0)).sum(0)+1e-3
    ys,xs=np.mgrid[0:d.shape[0],0:d.shape[1]]
    return (float((d*xs).sum()/d.sum()/d.shape[1]),float((d*ys).sum()/d.sum()/d.shape[0]))
win_reuse=Counter(); mc_cache={}
FILLED=[]
DIRS=[(0,0),(-0.12,-0.1),(0.12,-0.1),(-0.12,0.1),(0.12,0.1)]
for s in shots:
    if s['clip'] is not None: continue
    prev,nxt=neighbors(s); cands=[]
    for c in ids:
        if c in ONCE or not gap_ok(c,s) or not compatible(c,s,prev,nxt): continue
        for (a,b) in used[c]:
            if b-a>=s['need']: cands.append((win_reuse[(c,a)],uses[c],abs(strength[c]-at(INT,s['t'])),c,a,b))
    if not cands: continue
    cands.sort(); _,_,_,c,a,b=cands[0]
    k=win_reuse[(c,a)]; win_reuse[(c,a)]+=1
    if (c,a) not in mc_cache: mc_cache[(c,a)]=motion_center(c,a,b)
    cx,cy=mc_cache[(c,a)]; dx,dy=DIRS[k%len(DIRS)]
    z=int(rng.integers(120,181)); cx=min(max(cx+dx,0.5/z*100),1-0.5/z*100); cy=min(max(cy+dy,0.5/z*100),1-0.5/z*100)
    s['clip']=c; s['src_in'],s['src_out']=a,a+s['need']; uses[c]+=1; appear[c].append((s['t'],s['te']))
    s['variant']=f"fill:zoom{z}%@({cx:.2f},{cy:.2f})"; FILLED.append(s['n'])
for s in shots:
    if s.get('clip') in ONCE:
        for x in shots:
            if abs(x['t']-s['t'])<0.5+s['dur'] and x.get('flash'): x['flash']=0
# ---------------- 输出
cols=['shot','seg','t_start','t_end','dur_ms','frames','eighths','clip_id','src_in','src_out','speed','win_pct','variant','trigger','fx','transition','ascii_s','flash']
with open('work/edl.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(cols)
    for s in shots:
        w.writerow([s['n'],s['seg'],f"{s['t']:.3f}",f"{s['te']:.3f}",round(s['dur']*1000),round(s['dur']*FPS),round(s['dur']/q,2),s.get('clip'),
                    f"{s.get('src_in',0):.3f}",f"{s.get('src_out',0):.3f}",s['speed'],s['win'],s.get('variant',''),s['trig'],' '.join(s['fx']),
                    'ascii' if s['ascii'] else ('fadein' if s['n']==1 else 'cut'),s['ascii'],s['flash']])
unassigned=[s['n'] for s in shots if s.get('clip') is None]
byseg=defaultdict(list)
for s in shots: byseg[s['seg']].append(s['dur'])
src_used=sum(b-a for c in used for a,b in used[c])
ft=sorted(s['t'] for s in shots if s['flash']); maxf=max((sum(1 for x in ft if t<=x<t+1) for t in ft),default=0)
run=1;mx=1
for a,b in zip(shots,shots[1:]):
    if a.get('clip') and b.get('clip') and similar(a['clip'],b['clip']): run+=1; mx=max(mx,run)
    else: run=1
viol=[s['n'] for a,s in zip(shots,shots[1:]) if s['dur']<0.25 and a.get('clip') and s.get('clip') and abs(LUM[s['clip']]-LUM[a['clip']])>0.25*max(LUM[a['clip']],0.2)]
# 不规则区连续碎切（<0.3s）最长时长
irr=[s for s in shots if s['seg']=='irregular']; cur=0;mxfrag=0
for s in irr:
    cur=cur+s['dur'] if s['dur']<0.3 else 0; mxfrag=max(mxfrag,cur)
stats=dict(fps=FPS,crop_top=CROP_TOP,need_total=round(sum(s['need'] for s in shots),1),supply_raw=round(sum(float(CLIPS[c]['dur'])-2*INSET for c in ids),1),
           shots=len(shots),unassigned=unassigned,variant_filled=FILLED,avg_len={k:round(np.mean(v),3) for k,v in byseg.items()},n_seg={k:len(v) for k,v in byseg.items()},
           utilization=round(src_used/304.5,3),src_used=round(src_used,1),reuse_dist={str(k):v for k,v in sorted(Counter(uses[c] for c in ids).items())},unused=[c for c in ids if uses[c]==0],
           max_flash_per_s=maxf,n_flash=len(ft),max_similar_run=mx,short_lum_viol=viol,ascii_total=ascii_total,n_ascii=len(ascii),
           corp_cuts=sum(1 for s in shots if s['trig']=='corp'),frag_cuts=sum(1 for s in shots if s['trig']=='frag'),n_ramps=len(ramp_ids),
           bga_events=[(ev,next(s['n'] for s in shots if s['trig']=='bga' and abs(s['t']+LEAD-ev)<0.06)) for ev in BGA_EV],
           min_shot_frames=min(round(s['dur']*FPS) for s in shots),max_shot=round(max(s['dur'] for s in shots),2),max_irregular_frag_run_s=round(mxfrag,1))
json.dump(stats,open('work/plan_v3_stats.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(stats,ensure_ascii=False))
