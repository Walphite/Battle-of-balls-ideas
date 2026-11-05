"""V7 EDL 重排：
 1) 普通切点：拍点 −25ms 提前量并取最近帧（k/30）；锚点吸附的切点保持精确锚点时刻；
 2) 规则区（16–79、188–208、230–266 s）边界允许 ±1 拍，Viterbi 最小化「边界到最近强起音的距离」（重音得分）；
 3) 吞噬时刻 t_hit（置信度≥0.5）：映射到 ±0.3s 内最近的强起音锚点，片段内两段变速（吞噬前后不同速度，平滑缓动，不拆分素材）；
 4) 吞噬处加重音 hit:4f（缩放脉冲≤4% + RGB 色差 2 帧），与字符溶解/定格重复/硬切重叠的重音被丢弃。
 输出 work/edl_v7.csv、work/plan_v7_stats.json"""
import csv,json,os,re,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
FPS=30; LEAD=0.025; RAMPW=0.25
AN=json.load(open('work/anchors.json')); anchors=AN['anchors']; PER=AN['per']; Q=PER/2
BT=np.array(json.load(open('work/beats_v4.json'))['beats']); FT=np.load('work/onsets_fine.npz'); ft,fs=FT['t'],FT['s']
strong=np.sort(ft[fs>=AN['threshold_z']]); strong2=np.sort(ft[fs>=AN['threshold_tier2_z']]); atm={a['t_ms']/1000:a for a in anchors}; at_all=np.array(sorted(atm)); at_t1=np.array(sorted(t for t,a in atm.items() if a['tier']==1))
for _t in strong2:
    if all(abs(_t-x)>0.002 for x in at_all[np.clip(np.searchsorted(at_all,_t),0,len(at_all)-1):][:1]) : atm.setdefault(float(_t),dict(id=-1,tier=3))
at_all=np.array(sorted(atm))
CL={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}; HIT=json.load(open('work/hits.json'))
OLD=list(csv.DictReader(open('work/edl_v6.csv',encoding='utf-8'))); SCHED=json.load(open('work/glyph_schedule_v5.json'))['events']
RULE=[(16,79),(188,208),(230,266)]
def nearest(arr,t): j=int(np.clip(np.searchsorted(arr,t),1,len(arr)-1)); a,b=arr[j-1],arr[j]; return a if abs(t-a)<abs(t-b) else b
def in_rule(t): return any(a<=t<b for a,b in RULE)
n=len(OLD); b0=[int(r['beat_start']) for r in OLD]; nb=[int(r['beats']) for r in OLD]
K=[b0[0]]+[b0[i]+nb[i] for i in range(n)]                      # 边界的拍序号（n+1 个）
t_old=[float(OLD[0]['t_start'])]+[float(r['t_end']) for r in OLD]
anchored={j for j in range(n+1) if (j>0 and OLD[j-1]['anchor_id_end']) or (j<n and OLD[j]['anchor_id'])}
# 含完整溶解事件的镜头：其起止边界固定
full=[e for e in SCHED if e['kind'] in('out','in','scan','final')]
def ev_span(e):
    k=e['kind']; return (e['A0']-0.5 if k=='out' else (e['B0'] if k=='in' else e['A0']), e['A2'] if k in('out','scan') else (e['A1'] if k=='in' else e['t_end']))
fixed={0,n}|anchored
for e in full:
    a,b=ev_span(e)
    for j in range(n):
        if t_old[j]-0.01<=a<t_old[j+1]+0.01 or t_old[j]-0.01<=b<=t_old[j+1]+0.01: fixed|={j,j+1}
# ---------- 2) Viterbi：规则区边界 ±1 拍
cost=lambda k:abs(float(BT[k])-nearest(strong2,float(BT[k])))   # 重音得分：到最近强起音(≥P50)的距离
DEL=(-1,0,1); INF=1e9
cand=[[0] if (j in fixed or not in_rule(t_old[j])) else list(DEL) for j in range(n+1)]
dp=[{d:(0.0 if True else 0) for d in cand[0]}]; bp=[{}]
for j in range(1,n+1):
    cur={}; back={}
    for d in cand[j]:
        kj=K[j]+d; cj=(cost(kj) if d!=0 or (j not in fixed and in_rule(t_old[j])) else 0.0)+0.03*abs(d)
        best=INF; arg=None
        for dp_,v in dp[j-1].items():
            nbeats=kj-(K[j-1]+dp_)
            if nbeats<5 or nbeats>14: continue
            if v+cj<best: best=v+cj; arg=dp_
        if arg is not None: cur[d]=best; back[d]=arg
    dp.append(cur); bp.append(back)
d_end=min(dp[n],key=dp[n].get); dl=[0]*(n+1); dl[n]=d_end
for j in range(n,0,-1): dl[j-1]=bp[j][dl[j]]
K2=[K[j]+dl[j] for j in range(n+1)]
# 新边界时间
T=[]
for j in range(n+1):
    if j in fixed or K2[j]==K[j] and j in anchored: T.append(t_old[j])
    else: T.append(round((float(BT[K2[j]])-LEAD)*FPS)/FPS)
T[0]=t_old[0]; T[n]=t_old[n]
# ---------- 3) 每个镜头：速度模型
rows=[]; low=[]; unplaced=[]; stat_hit=[]; hitmapped=0
for i,r in enumerate(OLD):
    c=int(r['clip_id']); D=float(CL[c]['dur']); t0,t1=T[i],T[i+1]; O=t1-t0; seg=r['seg']
    old_sp=r['speed']; ramp=old_sp.startswith('ramp'); slow=(seg=='outro')
    h=HIT[str(c)]; conf=h['conf']; th=h['t_hit']; si0=float(r['src_in']); so0=float(r['src_out']); d0=so0-si0
    new=dict(r); new['t_start']=f'{t0:.4f}'; new['t_end']=f'{t1:.4f}'; new['dur_ms']=round(O*1000); new['beat_start']=K2[i]; new['beats']=K2[i+1]-K2[i]
    best=None
    if not slow:
        v_nat=d0/O if not ramp else d0/(O-0.2); A_nat=t0+(th-si0)/max(v_nat,1e-6)
        for lim,win in (((0.8,1.25),0.3),((0.7,1.4),0.3),((0.7,1.4),0.45)):
            for si in (0.0,0.05,0.1):
                for so in (D-0.1,D-0.05,D):
                    Lh=th-si; Rh=so-th
                    if Lh<=0.1 or Rh<=0.1: continue
                    for A in at_all[(at_all>t0+0.25)&(at_all<t1-0.25)&(np.abs(at_all-A_nat)<=win)]:
                        Oh=A-t0; Or=t1-A; w=min(RAMPW,0.8*Oh,0.8*Or); q=0.25*w
                        M=np.array([[Oh-q,q],[q,Or-q]]); va,vb=np.linalg.solve(M,np.array([Lh,Rh]))
                        if not(lim[0]<=va<=lim[1] and lim[0]<=vb<=lim[1]): continue
                        tier=atm[float(A)]['tier']; cst=abs(va-1)+abs(vb-1)+0.5*abs(va-vb)+(0.1 if tier==2 else (0.25 if tier==3 else 0))+0.3*abs(A-A_nat)+(0.5 if lim[1]>1.25 else 0)+(0.3 if win>0.3 else 0)
                        if best is None or cst<best[0]: best=(cst,float(A),va,vb,w,si,so,(lim[0],lim[1],win))
            if best: break
    if conf>=0.5 and best:
        cst,A,va,vb,w,si,so,lim=best; new['src_in']=f'{si:.3f}'; new['src_out']=f'{so:.3f}'
        new['speed']=f'hit:{va:.4f}:{vb:.4f}:{A-t0:.4f}:{w:.3f}'; new['hit_t']=round(A,4); new['hit_conf']=conf; new['hit_anchor']=atm[A]['id']; new['hit_tier']=atm[A]['tier']; hitmapped+=1
        if lim[1]>1.25 or lim[2]>0.3: new['hit_note']='relaxed'+(' speed 0.7-1.4' if lim[1]>1.25 else '')+(' window 0.45s' if lim[2]>0.3 else '')
        stat_hit.append((i+1,c,0.0))
    else:
        # 保持原做法：按新时长重算匀速/速度坡/慢放
        if slow: new['speed']=f'{d0/O:.4f}'
        elif ramp: new['speed']=re.sub(r'ramp:[\d.]+',f'ramp:{d0/(O-0.2):.4f}',old_sp)
        else:
            d=min(max(d0,max(D-0.2,0.5)),D); lim=(0.85,1.15); v=np.clip(d/O,*lim); d=v*O; d=min(d,D); tr=(D-d)/2; new['src_in']=f'{tr:.3f}'; new['src_out']=f'{D-tr:.3f}'; new['speed']=f'{d/O:.4f}'
        A_nat_=None
        if not slow:
            v=float(new['speed'].split(':')[1]) if new['speed'].startswith('ramp') else float(new['speed']); A_nat_=t0+(th-float(new['src_in']))/max(v,1e-6)
            dist=abs(A_nat_-nearest(at_all,A_nat_)); stat_hit.append((i+1,c,round(dist,4)))
        (low if conf<0.5 else unplaced).append(c)
    rows.append(new)
# ---------- 4) fx：BGA 重音重排 + 吞噬重音
BGA=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]
def clean_bga(fx):
    out=[];i=0
    while i<len(fx):
        if i+1<len(fx) and fx[i+1].startswith('bga_accent@') and fx[i].split('@')[0] in('rgb:4f','block:4f'): i+=2; continue
        out.append(fx[i]); i+=1
    return out
for r in rows: r['fx_l']=clean_bga(r['fx'].split())
for idx,e in enumerate(BGA):
    s=next(r for r in rows if float(r['t_start'])-0.05<=e<float(r['t_end'])); off=max(0.0,e-float(s['t_start'])); typ='rgb:4f' if idx%2==0 else 'block:4f'
    s['fx_l']=[f'{typ}@{off:.3f}',f'bga_accent@{off:.3f}']+s['fx_l']
loc=[(e['t'],e['t']+e['dur']) for e in SCHED if e['kind']=='local']; fulls=[ev_span(e) for e in full]
dropped=[]
for r in rows:
    if 'hit_t' not in r: continue
    tt=r['hit_t']; a,b=tt-3/FPS,tt+4/FPS+3/FPS
    # 定格重复冲突：去掉 stutter 记号（保留吞噬重音，因其对位到锚点）
    t0=float(r['t_start']); keep=[]
    for tok in r['fx_l']:
        m=re.match(r'stutter:(\d+)f@([\d.]+)',tok)
        if m:
            s0=t0+float(m.group(2)); s1=s0+int(m.group(1))/FPS
            if not(s1<a or s0>b): dropped.append(('stutter',round(s0,3),r['shot'])); continue
        keep.append(tok)
    r['fx_l']=keep
    conflict=any(not(l1<a or l0>b) for l0,l1 in loc) or any(not(f1<a or f0>b) for f0,f1 in fulls) or abs(tt-float(r['t_start']))<3/FPS or abs(tt-float(r['t_end']))<7/FPS
    if conflict: dropped.append(('hit_accent',round(tt,3),r['shot'])); r['hit_acc']=0
    else: r['fx_l'].append(f"hit:4f@{tt-t0:.4f}"); r['hit_acc']=1
cols=list(OLD[0].keys())+['hit_t','hit_conf','hit_anchor','hit_tier','hit_acc']
with open('work/edl_v7.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader()
    for r in rows: r['fx']=' '.join(r['fx_l']); w.writerow(r)
# ---------- 统计
bd=[(T[j],K2[j],j) for j in range(1,n)]
def reg_stats(lo,hi):
    xs=[(t,k) for t,k,j in bd if lo<=t<hi]
    if not xs: return None
    dg=np.array([(t-(float(BT[k])-LEAD))*1000 for t,k in xs]); dgrid=np.array([(t-float(BT[k]))*1000 for t,k in xs]); don=np.array([abs(t+LEAD-nearest(strong,float(BT[k])))*1000 for t,k in xs]); don2=np.array([abs(float(BT[k])-nearest(strong2,float(BT[k])))*1000 for t,k in xs]); don70=np.array([abs(float(BT[k])-nearest(strong,float(BT[k])))*1000 for t,k in xs])
    return dict(n=len(xs),grid_median_ms=round(float(np.median(dgrid)),1),grid_mean_ms=round(float(dgrid.mean()),1),onset_median_ms=round(float(np.median(don2)),1),onset_le40=round(float((don2<=40).mean()),2),onset70_median_ms=round(float(np.median(don70)),1),onset70_le40=round(float((don70<=40).mean()),2))
st=dict(shifted_boundaries=sum(1 for d in dl if d),by_region={f'{a}-{b}':reg_stats(a,b) for a,b in RULE+[(0,16),(79,188),(100,188),(208,230),(266,279)]},overall=reg_stats(0,300),
        hit_mapped=hitmapped,low_conf_clips=sorted(set(low)),unplaced_clips=sorted(set(unplaced)),dropped=dropped,
        hit_tier=dict(__import__('collections').Counter(r.get('hit_tier') for r in rows if 'hit_tier' in r)),hit_dist=dict(max_ms=round(max(d for _,_,d in stat_hit)*1000,1),median_ms=round(float(np.median([d for _,_,d in stat_hit]))*1000,1),over17=sum(1 for _,_,d in stat_hit if d>0.017),n=len(stat_hit)))
json.dump(st,open('work/plan_v7_stats.json','w'),ensure_ascii=False,indent=1); print(json.dumps(st,ensure_ascii=False))
