"""V9（终版）EDL：LEAD=0（画面不预先提前，成片时把音频整体延迟 25ms 实现提前量）；LOW_CONF=0；只用强起音（一/二/三级 = ≥P50），不再用弱起音；区段内匈牙利指派（强度分差阈值 0.25，跨区段不换）；局部速度坡（吞噬前 0.4s 最低 0.6×、吞噬后 0.6s 最高 1.5×，片段平均速度仍 0.8–1.25×）。
V8 EDL：在 plan_v7 的基础上扩大吞噬对位覆盖率。
 a) 边界偏移 ≤1 个八分拍（相邻片段各自调整，规则区已落强起音的边界不动）；b) 强度分差 ≤0.1 的槽位间互换片段（匈牙利指派最大化落位数）；
 c) 速度严格 0.8–1.25×（不放宽）；窗口 ±0.3s；d) 落不了位的片段列原因。
 低置信度（<0.5）片段保持原做法（LOW_CONF 可改）。输出 work/edl_v10.csv、work/plan_v10_stats.json"""
import csv,json,os,re,sys,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
LOW_CONF=float(os.environ.get('LOW_CONF','0')); FPS=30; LEAD=0.0; STR_THR=0.25; LAMBDA=50.0; DIP=0.8; PEAK=1.2; RAMPW=0.25; VLO,VHI=0.8,1.25
AN=json.load(open('work/anchors.json')); anchors=AN['anchors']; PER=AN['per']; Q=PER/2
BT=np.array(json.load(open('work/beats_v4.json'))['beats']); FT=np.load('work/onsets_fine.npz'); ft,fs=FT['t'],FT['s']
strong2=np.sort(ft[fs>=AN['threshold_tier2_z']]); strong=np.sort(ft[fs>=AN['threshold_z']])
atm={a['t_ms']/1000:a for a in anchors}
for _t in strong2: atm.setdefault(float(_t),dict(id=-1,tier=3))
T4=float(os.environ.get('TIER4_PCT','0'))
if T4>0:
    for _t in ft[fs>=np.percentile(fs,T4)]: atm.setdefault(float(_t),dict(id=-1,tier=4))      # 四级：较弱起音（仅在更强的都不可用时作为落位点，已标注）
at_all=np.array(sorted(atm))
CL={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}; HIT=json.load(open('work/hits.json')); FE={f['clip_id']:f for f in json.load(open('work/clip_feats.json'))}
OLD=list(csv.DictReader(open('work/edl_v6.csv',encoding='utf-8'))); SCHED=json.load(open('work/glyph_schedule_v10.json'))['events']
RULE=[(16,79),(188,208),(230,266)]
def nearest(arr,t): j=int(np.clip(np.searchsorted(arr,t),1,len(arr)-1)); a,b=arr[j-1],arr[j]; return a if abs(t-a)<abs(t-b) else b
def in_rule(t): return any(a<=t<b for a,b in RULE)
ids=sorted(FE); M=np.array([FE[i]['motion'] for i in ids]); Sa=np.array([FE[i]['sat'] for i in ids]); V=np.array([FE[i]['motion_std'] for i in ids])
rank=lambda v:(np.argsort(np.argsort(v))/(len(v)-1)); STR=dict(zip(ids,0.45*rank(M)+0.3*rank(Sa)+0.25*rank(V)))
HUE={i:np.array(FE[i]['hue_hist']) for i in ids}
def similar(a,b): return HUE[a]@HUE[b]/(np.linalg.norm(HUE[a])*np.linalg.norm(HUE[b])+1e-9)>0.93 and abs(FE[a]['lum']-FE[b]['lum'])<0.04
n=len(OLD); b0=[int(r['beat_start']) for r in OLD]; nb=[int(r['beats']) for r in OLD]
K=[b0[0]]+[b0[i]+nb[i] for i in range(n)]; t_old=[float(OLD[0]['t_start'])]+[float(r['t_end']) for r in OLD]
anchored={j for j in range(n+1) if (j>0 and OLD[j-1]['anchor_id_end']) or (j<n and OLD[j]['anchor_id'])}
full=[e for e in SCHED if e['kind'] in('out','in','scan','final','f')]; locs=[(e['t'],e['t']+e['dur']) for e in SCHED if e['kind']=='local']
def ev_span(e):
    k=e['kind']; return (e['A0']-0.5 if k=='out' else (e['B0'] if k=='in' else e['A0']), e['A2'] if k in('out','scan') else (e['A1'] if k in('in','f') else e['t_end']))
OV={int(k):v for k,v in json.load(open('work/zone_overrides.json')).items()}
fixed={0,n}|anchored|set(OV)
for e in full:
    a,b=ev_span(e)
    for j in range(n):
        if t_old[j]-0.01<=a<t_old[j+1]+0.01 or t_old[j]-0.01<=b<=t_old[j+1]+0.01: fixed|={j,j+1}
# ---------- 规则区 Viterbi（同 v7）
def feas_u(c,O,lo=0.8,hi=1.25):
    D=float(CL[c]['dur']); return max(D-0.2,0.5)/O<=hi and D/O>=lo
cost=lambda k:abs(float(BT[k])-nearest(strong2,float(BT[k])))
cand=[[0] if (j in fixed or not in_rule(t_old[j])) else [-1,0,1] for j in range(n+1)]
dp=[{d:0.0 for d in cand[0]}]; bp=[{}]
for j in range(1,n+1):
    cur={};back={}
    for d in cand[j]:
        kj=K[j]+d; cj=(cost(kj) if (j not in fixed and in_rule(t_old[j])) else 0.0)+0.03*abs(d); best=1e9;arg=None
        for dp_,v in dp[j-1].items():
            nbk=kj-(K[j-1]+dp_)
            if nbk<5 or nbk>14: continue
            if nbk!=(K[j]-K[j-1]) and OLD[j-1]['seg']!='outro' and not feas_u(int(OLD[j-1]['clip_id']),PER*nbk): continue
            if v+cj<best: best=v+cj;arg=dp_
        if arg is not None: cur[d]=best;back[d]=arg
    dp.append(cur);bp.append(back)
dl=[0]*(n+1); dl[n]=min(dp[n],key=dp[n].get)
for j in range(n,0,-1): dl[j-1]=bp[j][dl[j]]
K2=[K[j]+dl[j] for j in range(n+1)]
T=[]
for j in range(n+1):
    if j in fixed or (K2[j]==K[j] and j in anchored): T.append(t_old[j])
    else: T.append(round((float(BT[K2[j]])-LEAD)*FPS)/FPS)
T[0]=t_old[0]; T[n]=t_old[n]
for _j,_t in OV.items(): T[_j]=_t        # 人声减弱区事件的边界吸附到锚点精确时刻
T=np.array(T)
good_rule={j for j in range(1,n) if in_rule(T[j]) and abs(float(BT[K2[j]])-nearest(strong2,float(BT[K2[j]])))<=0.04}   # 已落强起音的规则区边界：不动
# ---------- 槽位与片段
slot_clip=[int(r['clip_id']) for r in OLD]; special={97,109}
slot_fixed=[(r['seg']=='outro') or slot_clip[i] in special for i,r in enumerate(OLD)]
def feas_uniform(c,O,lo=VLO,hi=VHI):
    D=float(CL[c]['dur']); return max(D-0.2,0.5)/O<=hi and D/O>=lo
def in_loc(t0,t1): return any(not(l1<t0-2/FPS or l0>t1+2/FPS) for l0,l1 in locs)
def m_profile(tau,Oh):
    """局部速度包络 m(τ)：吞噬前 0.4s 内平滑降到 DIP（慢），吞噬后 0.12s 内升到 PEAK（快）再在 0.6s 内回到 1"""
    sm=lambda x:(lambda y:y*y*(3-2*y))(np.clip(x,0,1))
    pre=1-(1-DIP)*sm((tau-(Oh-0.4))/0.3)*(tau<=Oh)
    up=DIP+(PEAK-DIP)*sm((tau-Oh)/0.12); down=PEAK+(1-PEAK)*sm((tau-(Oh+0.12))/0.48)
    post=np.where(tau<Oh+0.12,up,down)
    return np.where(tau<=Oh,pre,post)
def solve_hit(c,t0,t1,win=0.3):
    """返回 (cost,A,va,vb,w,si,so,tier)；基线速度 va/vb ∈ [0.8,1.25]，局部包络使最低 0.64×、最高 1.5×"""
    D=float(CL[c]['dur']); th=HIT[str(c)]['t_hit']; O=t1-t0; d_nat=min(max(D-0.1,max(D-0.2,0.5)),D); v_nat=float(np.clip(d_nat/O,0.85,1.15)); si_n=(D-d_nat)/2; A_nat=t0+(th-si_n)/v_nat
    best=None; dt=1/300; tau=np.arange(0,O+dt,dt)
    for A in at_all[(at_all>t0+0.25)&(at_all<t1-0.25)&(np.abs(at_all-A_nat)<=win)]:
        Oh=A-t0; Or=t1-A; w=min(RAMPW,0.8*Oh,0.8*Or); sm=np.clip((tau-(Oh-w))/(2*w),0,1); sm=sm*sm*(3-2*sm); m=m_profile(tau,Oh); pre=tau<=Oh
        I11=float((m*(1-sm))[pre].sum()*dt); I12=float((m*sm)[pre].sum()*dt); I21=float((m*(1-sm))[~pre].sum()*dt); I22=float((m*sm)[~pre].sum()*dt)
        Mx=np.array([[I11,I12],[I21,I22]])
        for si in (0.0,0.05,0.1):
            for so in (D-0.1,D-0.05,D):
                Lh=th-si; Rh=so-th
                if Lh<=0.1 or Rh<=0.1: continue
                try: va,vb=np.linalg.solve(Mx,np.array([Lh,Rh]))
                except np.linalg.LinAlgError: continue
                if not(VLO<=va<=VHI and VLO<=vb<=VHI): continue
                tier=atm[float(A)]['tier']; cst=abs(va-1)+abs(vb-1)+0.5*abs(va-vb)+(0.1 if tier==2 else (0.25 if tier==3 else 0))+0.3*abs(A-A_nat)
                if best is None or cst<best[0]: best=(cst,float(A),float(va),float(vb),w,si,so,tier)
    return best
def shift_ok(j,Tcur,frozen):
    return j not in fixed and j not in good_rule and j not in frozen and 0<j<n
def options(i,Tcur,frozen):
    out=[(0,0)]
    for ds in (0,-1,1):
        for de in (0,-1,1):
            if (ds,de)==(0,0): continue
            if ds and not shift_ok(i,Tcur,frozen): continue
            if de and not shift_ok(i+1,Tcur,frozen): continue
            out.append((ds,de))
    return sorted(out,key=lambda x:abs(x[0])+abs(x[1]))
def eval_slot(i,c,Tcur,frozen,assign):
    """(cost,(ds,de),sol) 最优落位方案；考虑相邻槽位在当前（假定）片段下的时长可行性与局部溶解窗口"""
    if HIT[str(c)]['conf']<LOW_CONF: return None
    best=None
    for ds,de in options(i,Tcur,frozen):
        t0=round((Tcur[i]+ds*Q)*FPS)/FPS; t1=round((Tcur[i+1]+de*Q)*FPS)/FPS
        if t1-t0<1.6: continue
        if (ds and in_loc(min(Tcur[i],t0),max(Tcur[i],t0))) or (de and in_loc(min(Tcur[i+1],t1),max(Tcur[i+1],t1))): continue
        if ds and i>0 and not feas_uniform(assign[i-1],(Tcur[i]+ds*Q)-Tcur[i-1]) and not slot_fixed[i-1]==False and False: pass
        if ds and i>0 and not slot_fixed[i-1] and not feas_uniform(assign[i-1],(t0)-Tcur[i-1]): continue
        if de and i+1<n and not slot_fixed[i+1] and not feas_uniform(assign[i+1],Tcur[i+2]-t1): continue
        s=solve_hit(c,t0,t1)
        if s is None: continue
        cst=s[0]+0.25*(abs(ds)+abs(de))
        if best is None or cst<best[0]: best=(cst,(ds,de),s,t0,t1)
    return best
Tcur=T.copy(); frozen=set(); assign=list(slot_clip)
# ---------- 阶段 1：区段内全局最优指派（匈牙利）。区段 = 连续同 seg 的槽位；跨区段不换
import time; T0=time.time(); NS=n
def hungarian(C):
    nn=C.shape[0]; u=np.zeros(nn+1); v=np.zeros(nn+1); p=np.zeros(nn+1,int); way=np.zeros(nn+1,int)
    for i in range(1,nn+1):
        p[0]=i; j0=0; minv=np.full(nn+1,np.inf); used=np.zeros(nn+1,bool)
        while True:
            used[j0]=True; i0=p[j0]; delta=np.inf; j1=0
            for j in range(1,nn+1):
                if not used[j]:
                    cur=C[i0-1,j-1]-u[i0]-v[j]
                    if cur<minv[j]: minv[j]=cur; way[j]=j0
                    if minv[j]<delta: delta=minv[j]; j1=j
            for j in range(nn+1):
                if used[j]: u[p[j]]+=delta; v[j]-=delta
                else: minv[j]-=delta
            j0=j1
            if p[j0]==0: break
        while True:
            j1=way[j0]; p[j0]=p[j1]; j0=j1
            if j0==0: break
    res=np.zeros(nn,int)
    for j in range(1,nn+1): res[p[j]-1]=j-1
    return res
regions=[]; i=0
while i<n:
    j=i
    while j+1<n and OLD[j+1]['seg']==OLD[i]['seg']: j+=1
    regions.append(list(range(i,j+1))); i=j+1
Tcur=T.copy(); frozen=set(); assign=list(slot_clip); greedy_used=False; BIG=1e6
for reg in regions:
    sl=[k for k in reg if not slot_fixed[k]]; cl=[slot_clip[k] for k in sl]; m_=len(sl)
    if m_==0: continue
    W=np.full((m_,m_),BIG)
    for a_,k in enumerate(sl):
        for b_,c in enumerate(cl):
            if abs(STR[c]-STR[slot_clip[k]])>STR_THR+1e-9: continue
            e=eval_slot(k,c,Tcur,frozen,assign); arc=LAMBDA*abs(STR[c]-STR[slot_clip[k]])
            W[a_,b_]=(-(1000-100*e[0])+arc) if e else (((0.0 if c==slot_clip[k] else 0.5)+arc) if (c==slot_clip[k] or feas_u(c,Tcur[k+1]-Tcur[k])) else BIG)
    if time.time()-T0>900:            # 超过 15 分钟：退回贪心
        greedy_used=True; col=np.arange(m_)
    else: col=hungarian(W)
    for a_,k in enumerate(sl): assign[k]=cl[col[a_]]
# 邻接相似连续检查（≤2）：违反则撤销该槽的互换
def sim_run(a):
    r=1;m=1
    for x,y in zip(a,a[1:]): r=r+1 if similar(x,y) else 1; m=max(m,r)
    return m
swapped=[i for i in range(NS) if assign[i]!=slot_clip[i]]
while sim_run(assign)>2 and swapped:
    # 撤销互换：把涉及的槽位恢复为原片段（并同时恢复其对偶槽位，保持排列有效）
    i=swapped.pop(); c=assign[i]; k=[x for x in range(NS) if assign[x]==slot_clip[i]][0]; assign[k]=c; assign[i]=slot_clip[i]; swapped=[x for x in range(NS) if assign[x]!=slot_clip[x]]
# ---------- 阶段 2：按收益从大到小落位并冻结边界
order=list(range(NS))
sol={}
for i in order:
    c=assign[i]; e=eval_slot(i,c,Tcur,frozen,assign)
    if e is None: continue
    cst,(ds,de),s,t0,t1=e; Tcur[i]=t0; Tcur[i+1]=t1; frozen|={i,i+1}; sol[i]=(c,s,ds,de)
# ---------- 输出 EDL
BGA=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]
rows=[];unmapped=[]
for i,r in enumerate(OLD):
    c=assign[i]; D=float(CL[c]['dur']); t0=float(Tcur[i]); t1=float(Tcur[i+1]); O=t1-t0; new=dict(r)
    new.update(clip_id=c,t_start=f'{t0:.4f}',t_end=f'{t1:.4f}',dur_ms=round(O*1000))
    ks=int(round((t0+LEAD-float(BT[0]))/PER+0)); kb0=int(np.argmin(np.abs(BT-(t0+LEAD)))); kb1=int(np.argmin(np.abs(BT-(t1+LEAD)))); new['beat_start']=kb0; new['beats']=max(1,kb1-kb0)
    if i in sol:
        c,s,ds,de=sol[i]; cst,A,va,vb,w,si,so,tier=s
        new.update(src_in=f'{si:.3f}',src_out=f'{so:.3f}',speed=f'hit2:{va:.4f}:{vb:.4f}:{A-t0:.4f}:{w:.3f}:{DIP}:{PEAK}',hit_t=round(A,4),hit_conf=HIT[str(c)]['conf'],hit_anchor=atm[A]['id'],hit_tier=tier,hit_shift=f'{ds},{de}')
    else:
        old_sp=r['speed']; ramp=old_sp.startswith('ramp') and c==slot_clip[i]; seg=r['seg']
        if seg=='outro': new.update(src_in=r['src_in'],src_out=r['src_out'],speed=f"{(float(r['src_out'])-float(r['src_in']))/O:.4f}")
        else:
            d=min(max(O,max(D-0.2,0.5)),D); tr=(D-d)/2          # d∈[D−0.2,D]：首尾各修剪 ≤0.1 s，速度 d/O 由可行性保证落在 0.8–1.25
            new.update(src_in=f'{tr:.3f}',src_out=f'{D-tr:.3f}',speed=f'{d/O:.4f}')
        unmapped.append((i,c))
    rows.append(new)
def clean(fx):
    out=[];i=0
    while i<len(fx):
        if i+1<len(fx) and fx[i+1].startswith('bga_accent@') and fx[i].split('@')[0] in('rgb:4f','block:4f'): i+=2; continue
        out.append(fx[i]); i+=1
    return out
for r in rows: r['fx_l']=clean(r['fx'].split())
for idx,e in enumerate(BGA):
    s=next(r for r in rows if float(r['t_start'])-0.05<=e<float(r['t_end'])); off=max(0.0,e-float(s['t_start'])); s['fx_l']=[f"{'rgb' if idx%2==0 else 'block'}:4f@{off:.3f}",f'bga_accent@{off:.3f}']+s['fx_l']
fulls=[ev_span(e) for e in full]; dropped=[]
for r in rows:
    r['hit_acc']=''
    if 'hit_t' not in r: continue
    tt=r['hit_t']; a,b=tt-3/FPS,tt+7/FPS; t0=float(r['t_start']); keep=[]
    for tok in r['fx_l']:
        m=re.match(r'stutter:(\d+)f@([\d.]+)',tok)
        if m and not((t0+float(m.group(2))+int(m.group(1))/FPS)<a or (t0+float(m.group(2)))>b): dropped.append(('stutter',round(t0+float(m.group(2)),3),r['shot'])); continue
        keep.append(tok)
    r['fx_l']=keep
    if any(not(l1<a or l0>b) for l0,l1 in locs) or any(not(f1<a or f0>b) for f0,f1 in fulls) or abs(tt-t0)<3/FPS or abs(tt-float(r['t_end']))<7/FPS: dropped.append(('hit_accent',round(tt,3),r['shot'])); r['hit_acc']=0
    else: r['fx_l'].append(f"hit:4f@{tt-t0:.4f}"); r['hit_acc']=1
cols=list(OLD[0].keys())+['hit_t','hit_conf','hit_anchor','hit_tier','hit_shift','hit_acc']
with open('work/edl_v10.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader()
    for r in rows: r['fx']=' '.join(r['fx_l']); w.writerow(r)
# ---------- 统计与原因
def reason(i,c):
    h=HIT[str(c)]; D=float(CL[c]['dur']); t0,t1=float(Tcur[i]),float(Tcur[i+1]); th=h['t_hit']
    if h['conf']<LOW_CONF: return '置信度<%.1f（保持原做法）'%LOW_CONF
    d_nat=min(max(D-0.1,max(D-0.2,0.5)),D); v=float(np.clip(d_nat/(t1-t0),0.85,1.15)); A_nat=t0+(th-(D-d_nat)/2)/v
    near=at_all[np.abs(at_all-A_nat)<=0.45]
    if len(near)==0: return '±0.45 秒内无起音'
    if th<0.25 or D-th<0.25 or th-0.1<=0 or D-0.1-th<=0.1: return '吞噬点太靠头/尾'
    return '速度越界（0.8–1.25×）'
un=[dict(clip_id=c,shot=i+1,reason=reason(i,c)) for i,c in unmapped]
from collections import Counter
mapped=len(sol); total=NS
def dev_disp(): 
    d=[]
    for i,(c,s,ds,de) in sol.items(): A=s[1]; k=int(np.ceil(A*FPS-0.5)); d.append((k/FPS-A)*1000)
    return np.array(d)
dv=dev_disp()
st=dict(total=total,mapped=mapped,coverage=round(mapped/total,3),low_conf_threshold=LOW_CONF,swapped_slots=int(sum(1 for i in range(NS) if assign[i]!=slot_clip[i])),boundary_shifted=int(sum(1 for i,(c,s,ds,de) in sol.items() if ds or de)),
        shifts={str(k):v for k,v in Counter((ds,de) for i,(c,s,ds,de) in sol.items()).items()},tier=dict(Counter(s[7] for i,(c,s,ds,de) in sol.items())),unmapped=un,reasons=dict(Counter(u['reason'] for u in un)),
        disp_dev_ms=dict(max=round(float(np.abs(dv).max()),1),median=round(float(np.median(np.abs(dv))),1)),sim_run=sim_run(assign),dropped=dropped,greedy_fallback=greedy_used,solve_seconds=round(time.time()-T0,1))
json.dump(st,open('work/plan_v10_stats.json','w'),ensure_ascii=False,indent=1,default=str); print(json.dumps({k:v for k,v in st.items() if k not in('unmapped','dropped')},ensure_ascii=False,default=str)); print('未落位',[(u['clip_id'],u['reason']) for u in un])
