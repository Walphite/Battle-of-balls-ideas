"""字符溶解事件与锚点对位（V5）：
 1) 把全片 EDL 按校正后的拍网格重定时（旧网格取自粗起音，比真实起音早 ≈60ms）；
 2) 为溶出/凝出/扫描式选择「边界=锚点」的方案，必要时在 ±0.25s 外移动事件或把边界吸附到锚点（相邻片段调速≤上限）；
 3) 局部溶解取起音精确时刻；
 输出：work/edl.csv（增 anchor_id/anchor_id_end）、work/glyph_schedule_v5.json、work/glyph_events.csv、work/glyph_schedule_E.json（样 E）。"""
import json,csv,os,sys,re,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
FPS=30; T=278.806; LEAD=0.010; RAMP_G=0.2
AN=json.load(open('work/anchors.json')); DELTA=AN['delta_ms']/1000; PER=AN['per']; Q=PER/2
anch={a['grid_idx']:a for a in AN['anchors']}
BT=np.array(json.load(open('work/beats_v4.json'))['beats'])
FINE=np.load('work/onsets_fine.npz'); FT=FINE['t']; FS=FINE['s']
VO=json.load(open('work/vocal_onsets.json')); BGA_EV=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]
def G(g): return float(BT[g//2]) if g%2==0 else float((BT[g//2]+BT[g//2+1])/2)
def anc(g,tier): a=anch.get(g); return a if (a and a['tier']<=tier) else None
OLD=list(csv.DictReader(open('work/edl_v4_pre_anchor.csv',encoding='utf-8')))
shots=[]
for r in OLD:
    b=int(r['beat_start']); k=int(r['beats']); sp=r['speed']
    s=dict(n=int(r['shot']),seg=r['seg'],b=b,k=k,clip=int(r['clip_id']),src_in=float(r['src_in']),src_out=float(r['src_out']),win=r['win_pct'],fx=r['fx'].split(),
           bga_old=r['bga_event'],trans=r['transition'],flash=r['flash'],asc=r['ascii_s'],speed_old=sp,
           t0=float(BT[b])-LEAD,t1=float(BT[b+k])-LEAD)
    s['ramp']=sp.startswith('ramp'); s['hit_rel']=float(re.search(r'hit=([\d.]+)',sp).group(1)) if s['ramp'] else None
    s['d']=s['src_out']-s['src_in']; shots.append(s)
bounds=[dict(i=i,k=s['b']+s['k'],t=s['t1']) for i,s in enumerate(shots)]   # 第 i 个镜头的结束边界
used_bound=set(); events=[]; override={}      # override[(i,'end')]=(t,anchor_id)
def shot_start_t(i): return override.get((i-1,'end'),(shots[i]['t0'],None))[0] if i>0 else shots[0]['t0']
def shot_end_t(i): return override.get((i,'end'),(shots[i]['t1'],None))[0]
# ---------- 求解器
def cands_boundary(exp,wide,allowed=None):
    out=[]
    for bd in bounds:
        if allowed is not None and bd['i'] not in allowed: continue
        if bd['i'] in used_bound or bd['i']>=len(shots)-1: continue
        d=abs(bd['t']-exp)
        if d<=wide: out.append((d,bd))
    return sorted(out,key=lambda x:x[0])
def solve(kind,exp,wide=4.0,allowed=None,spec=None):
    best=None
    for dist,bd in cands_boundary(exp,wide,allowed):
        i=bd['i']; k=bd['k']; st=shot_start_t(i)
        for shift in (0,-1,1):
            g2=2*k+shift
            for tier in (1,2):
                a2=anc(g2,tier)
                if a2 is None: continue
                t2=a2['t_ms']/1000
                if kind=='out':
                    for m in (2,3):
                        a1=anc(g2-m,tier)
                        if a1 is None: continue
                        for n in (6,4):
                            a0=anc(g2-m-n,tier)
                            if a0 is None: continue
                            if a0['t_ms']/1000-0.5<st+0.02: continue
                            if spec and spec.get('n') and n!=spec['n']: continue
                            if spec and spec.get('m') and m!=spec['m']: continue
                            cost=10*dist+(0 if tier==1 else 4)+(6 if shift else 0)+(0 if a2['downbeat'] else 0.4)+0.1*(m==3)+0.1*(n==4)
                            if best is None or cost<best[0]: best=(cost,dict(kind='out',i=i,k=k,shift=shift,tier=tier,A=[a0,a1,a2],n=n,m=m,dist=dist,exp=exp))
                elif kind=='scan':
                    for n in (3,4):
                        a0=anc(g2-n,tier)
                        if a0 is None or a0['t_ms']/1000<st+0.02: continue
                        cost=10*dist+(0 if tier==1 else 4)+(6 if shift else 0)+(0 if a2['downbeat'] else 0.4)+0.1*(n==4)
                        if best is None or cost<best[0]: best=(cost,dict(kind='scan',i=i,k=k,shift=shift,tier=tier,A=[a0,a2],n=n,dist=dist,exp=exp))
    return best[1] if best else None
def solve_in(exp,wide=4.0,allowed=None):
    """凝出：起点=某镜头开始边界（锚点）；A0=+h 个八分拍的锚点；A1=A0+2 个八分拍的锚点"""
    best=None
    for i in range(1,len(shots)):
        if allowed is not None and i not in allowed: continue
        t=shot_start_t(i); dist=abs(t-exp)
        if dist>wide: continue
        g0=2*shots[i]['b']
        for shift in (0,-1,1):
            for tier in (1,2):
                a=anc(g0+shift,tier)
                if a is None: continue
                prev=override.get((i-1,'end'))
                if prev is not None and prev[1]!=a['id']: continue
                for h in (2,3,4):
                    for tier2 in (tier,2):
                        a0=anc(g0+shift+h,tier2); a1=anc(g0+shift+h+2,tier2)
                        if a0 is None or a1 is None: continue
                        if a1['t_ms']/1000>shot_end_t(i)-0.25: continue
                        cost=10*dist+(0 if max(tier,tier2)==1 else 4)+(6 if shift else 0)+(0 if a['downbeat'] else 0.4)+0.1*(h-2)
                        if best is None or cost<best[0]: best=(cost,dict(kind='in',i=i,shift=shift,tier=max(tier,tier2),A=[a,a0,a1],h=h,dist=dist,exp=exp))
    return best[1] if best else None
def commit(ev):
    """把边界吸附到锚点（精确毫秒），登记占用"""
    A=ev['A']
    if ev['kind'] in('out','scan'):
        a2=A[-1]; override[(ev['i'],'end')]=(a2['t_ms']/1000,a2['id']); used_bound.add(ev['i'])
    elif ev['kind']=='in':
        a=A[0]; override[(ev['i']-1,'end')]=(a['t_ms']/1000,a['id']); used_bound.add(ev['i']-1)
    events.append(ev)
def first_boundary_after(e,tol=0.05):
    return min((bd for bd in bounds if bd['t']>=e-tol),key=lambda b:b['t'])['t']
# ---------- 全片事件
exps=[(29.77,'out',dict()),(44.57,'scan',dict()),(47.91,'out',dict()),(144.31,'out',dict()),(187.21,'scan',dict()),(233.14,'out',dict(n=6))]
full=[]
# 开场凝出：第一个镜头（起点=影片开头，无锚点；A0/A1 取锚点或网格）
a0=None
for h in (2,3,4):
    for tier in (1,2):
        x=anc(2*shots[0]['b']+h,tier); y=anc(2*shots[0]['b']+h+2,tier)
        if x and y and a0 is None: a0=(x,y,h,tier)
if a0: A0,A1,note=a0[0]['t_ms']/1000,a0[1]['t_ms']/1000,'A0/A1 为锚点'
else: g=2*shots[0]['b']+2; A0,A1=G(g),G(g+2); note='首镜头内无锚点：A0/A1 取八分拍网格（未锚定）'
events.append(dict(kind='in',i=0,opening=True,A=[None,None,None],t_B0=shots[0]['t0'],tA0=A0,tA1=A1,note=note,dist=0,exp=shots[0]['t0'],tier=0,shift=0))
import itertools,copy
def try_plan(pts,types):
    """按给定类型依次求解并占用边界；返回 (总移动量, 事件列表, 状态)；无解返回 None"""
    global used_bound,override
    ub0,ov0=set(used_bound),dict(override); evs=[]; cost=0
    for (e,spec),k in zip(pts,types):
        sp=dict(spec); 
        ev=solve(k,first_boundary_after(e),wide=12.0,spec=sp)
        if ev is None: used_bound,override=ub0,ov0; return None
        ev['bga']=e; ev['spec']=sp; commit_state(ev); evs.append(ev); cost+=ev['dist']
    st=(set(used_bound),dict(override)); used_bound,override=ub0,ov0; return cost,evs,st
def commit_state(ev):
    A=ev['A']; a2=A[-1]; override[(ev['i'],'end')]=(a2['t_ms']/1000,a2['id']); used_bound.add(ev['i'])
# 规则区三个 BGA 点：类型交替（溶出/扫描式），取总移动量最小的组合；其余点固定类型
trio=[(29.77,{}),(44.57,{}),(47.91,{})]; best=None; cmb=[]
for types in itertools.product(('out','scan'),repeat=3):
    r=try_plan(trio,types)
    if not r: continue
    seq=[ev['kind'] for ev in sorted(r[1],key=lambda e:e['A'][-1]['t_ms'])]
    if any(a==b for a,b in zip(seq,seq[1:])): continue         # 时间顺序上不连用同一种
    tt=sorted(ev['A'][-1]['t_ms'] for ev in r[1])
    if any(b-a<3000 for a,b in zip(tt,tt[1:])): continue        # 两次完整溶解间隔 ≥3 s
    cmb.append((round(r[0],2),types,[round(e['dist'],2) for e in r[1]]))
    if best is None or r[0]<best[0]: best=(r[0],r[1],r[2],types)
print('规则区三点候选（总移动量, 类型(按 29.8/44.6/47.9), 各移动量）:',sorted(cmb)[:5])
assert best,'规则区无可行组合'
for ev in best[1]: events.append(ev)
used_bound.clear(); used_bound.update(best[2][0]); override.clear(); override.update(best[2][1])
for e,k,spec in [(144.31,'out',{}),(187.21,'scan',{}),(233.14,'out',dict(n=6))]:
    ev=solve(k,first_boundary_after(e),wide=12.0,spec=spec)
    if ev is None: print('!! 无解',e,k); continue
    ev['bga']=e; ev['spec']=spec; commit(ev)
# 收束：最后一个镜头
li=len(shots)-1; st=shot_start_t(li); en=shots[li]['t1']
cand=[(abs(G(g)-(en-3.6)),g) for g in range(2*shots[li]['b'],2*(shots[li]['b']+shots[li]['k'])) if G(g)>=st+0.02 and G(g)+1.0+2.1+0.4<=en]
fa=None
for d,g in sorted(cand):
    for tier in (1,2):
        a=anc(g,tier)
        if a: fa=(a,tier); break
    if fa: break
if fa: events.append(dict(kind='final',i=li,A0=fa[0]['t_ms']/1000,anchor=fa[0],tier=fa[1],t_end=en,note='A0 为锚点'))
else:
    g=min(cand)[1]; events.append(dict(kind='final',i=li,A0=G(g),anchor=None,tier=0,t_end=en,note='末镜头内无可用锚点：A0 取八分拍网格（未锚定）'))
# ---------- 局部溶解：起音精确时刻
def refine(x):
    tgt=x+DELTA; j=int(np.argmin(np.abs(FT-tgt))); return float(FT[j]) if abs(FT[j]-tgt)<=0.045 else float(tgt)
corp=sorted({round(refine(a),4) for a,_ in VO['corp_100_188'] if 100<=a<=113})
frag=[]; 
for a in VO['frag_merged']:
    r=refine(a)
    if 202<=a<=219.5 and (not frag or r-frag[-1]>=0.30): frag.append(round(r,4))
rng=np.random.default_rng(11); used_rect=[]
def rect():
    for _ in range(600):
        w=rng.uniform(0.18,0.38); h=rng.uniform(0.15,0.35); u=rng.uniform(0,1-w); v=rng.uniform(0,1-h); r=(u,v,u+w,v+h)
        if all(max(0,min(r[2],q[2])-max(r[0],q[0]))*max(0,min(r[3],q[3])-max(r[1],q[1]))<=0.0 for q in used_rect[-6:]): used_rect.append(r); return tuple(round(x,3) for x in r)
    used_rect.append(r); return tuple(round(x,3) for x in r)
def next_grid(t):
    k=int(np.searchsorted(BT,t)); gs=[G(g) for g in range(max(0,2*k-3),2*k+3)]; return min(x for x in gs if x>t+0.03)
def local_events(times,nblocks):
    out=[]
    for i,t in enumerate(times):
        nxt=times[i+1] if i+1<len(times) else 1e9
        dur=min(next_grid(t)-t,nxt-t,0.25); dur=max(dur,2/FPS)
        # 不跨镜头边界
        end=next(shot_end_t(j) for j in range(len(shots)) if shots[j]['t0']-1e-6<=t<shot_end_t(j)); dur=min(dur,end-t-0.01)
        out.append(dict(kind='local',t=t,dur=round(dur,4),rects=[rect() for _ in range(nblocks[i%len(nblocks)])],next_trigger=None if nxt>1e8 else nxt))
    return out
loc=local_events(corp,[1,2,3])+local_events(frag,[1,2])

def stutter_windows():
    out=[]
    for i,sh in enumerate(shots):
        for tok in sh['fx']:
            m=re.match(r'stutter:(\d+)f@([\d.]+)',tok)
            if m: t=shot_start_t(i)+float(m.group(2)); out.append((t,t+int(m.group(1))/FPS,i,tok))
    return out
def all_boundaries(): return sorted({shot_start_t(i) for i in range(len(shots))}|{shot_end_t(i) for i in range(len(shots))})
ANCH_T=sorted(a['t_ms']/1000 for a in AN['anchors'])
dropped_stutter=[]
def resolve_conflicts(evs):
    """局部溶解 vs 定格重复/硬切（±2 帧内不得同组）。
    定格重复冲突：局部溶解必须在起音精确时刻出现（锚点对位规则），所以保留局部溶解、去掉该处的定格重复（rgb 色差保留）；
    硬切冲突：局部溶解推迟到下一个锚点（任一级，且与其它局部事件间隔 ≥0.3s）。"""
    pad=2/FPS; moved=[]; bd=all_boundaries()
    taken=[e['t'] for e in evs]
    for e in evs:
        a,b=e['t']-pad,e['t']+e['dur']+pad
        for (w0,w1,i,tok) in stutter_windows():
            if not(w1<a or w0>b):
                shots[i]['fx'].remove(tok); dropped_stutter.append((round(w0,3),tok,shots[i]['n']))
        if any(a<=x<=b for x in bd):
            t0=e['t']; taken=[x for x in taken if x!=t0]
            for an in [x for x in ANCH_T if x>t0]:
                if any(abs(an-x)<0.3 for x in taken): continue
                end=next(shot_end_t(j) for j in range(len(shots)) if shot_start_t(j)-1e-6<=an<shot_end_t(j)); d=min(next_grid(an)-an,0.25,end-an-0.01)
                if d>=2/FPS and not any(an-pad<=x<=an+d+pad for x in bd):
                    e['t_orig']=t0; e['t']=round(an,4); e['dur']=round(d,4); e['pushed']=True; moved.append((round(t0,3),round(an,3))); taken.append(an); break
    return moved
# ---------- 重定时 EDL（先把锚点边界吸附，再写出）
def shot_time(i): return shot_start_t(i),shot_end_t(i)
warn=[]
for i,s in enumerate(shots):
    t0,t1=shot_time(i); dur=t1-t0
    if s['ramp']: v=s['d']/(dur-RAMP_G)
    else: v=s['d']/dur if s['speed_old'] and float(s['speed_old'])>0 else 1
    s['t0n'],s['t1n']=t0,t1; s['v']=v
    lim=(0.55,1.15) if s['seg']=='outro' else (0.85,1.15)
    if not(lim[0]-1e-6<=v<=lim[1]+1e-6):
        # 允许再多修剪/少修剪 ≤0.1s（每端）：d 在 [D-0.2, D]
        D=float(next(r['dur'] for r in csv.DictReader(open('work/clips.csv',encoding='utf-8')) if int(r['clip_id'])==s['clip']))
        L=dur-(RAMP_G if s['ramp'] else 0); d=min(max(L*np.clip(v,*lim),D-0.2),D); v2=d/L
        if lim[0]-1e-6<=v2<=lim[1]+1e-6: s['d']=d; s['src_in']=round((D-d)/2,3); s['src_out']=round(D-(D-d)/2,3); s['v']=v2
        else: warn.append((s['n'],round(v,3)))
# BGA 重音：去掉旧的 (typ:4f@off, bga_accent@off) 成对记号，按新时间在含事件的镜头里重新加
for s in shots:
    fx=s['fx']; out=[]; i=0
    while i<len(fx):
        if i+1<len(fx) and fx[i+1].startswith('bga_accent@') and fx[i].split('@')[0] in('rgb:4f','block:4f'): i+=2; continue
        out.append(fx[i]); i+=1
    s['fx']=out; s['bga_new']=''
bga_rows=[]
for idx,e in enumerate(BGA_EV):
    si=next(i for i,s in enumerate(shots) if s['t0n']-0.05<=e<s['t1n']); s=shots[si]; off=e-s['t0n']; typ='rgb:4f' if idx%2==0 else 'block:4f'
    if off<0: off=0.0
    s['fx']=[f'{typ}@{off:.3f}',f'bga_accent@{off:.3f}']+s['fx']
    bd=min(abs(s['t0n']-e),abs(s['t1n']-e)); s['bga_new']=f'{e} 重音距边界 {round(bd*1000)}ms'; bga_rows.append((e,si+1,round(bd*1000)))
# 写 EDL
cols=['shot','seg','t_start','t_end','dur_ms','beats','beat_start','clip_id','src_in','src_out','speed','win_pct','bga_event','fx','transition','ascii_s','flash','anchor_id','anchor_id_end']
with open('work/edl.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(cols)
    for i,s in enumerate(shots):
        sp=f"ramp:{s['v']:.4f}:hit={s['hit_rel']:.3f}:dip=0.5:w=0.5" if s['ramp'] else f"{s['v']:.4f}"
        a_s=override.get((i-1,'end'),(None,''))[1] if i>0 else ''; a_e=override.get((i,'end'),(None,''))[1]
        w.writerow([s['n'],s['seg'],f"{s['t0n']:.3f}",f"{s['t1n']:.3f}",round((s['t1n']-s['t0n'])*1000),s['k'],s['b'],s['clip'],s['src_in'],s['src_out'],sp,s['win'],s['bga_new'],' '.join(s['fx']),s['trans'],'0',s['flash'],a_s if a_s is not None else '',a_e if a_e is not None else ''])
# ---------- 事件序列化
def ser(ev):
    k=ev['kind']; A=ev.get('A')
    if k=='out':
        a0,a1,a2=A; return dict(kind='out',variant='溶出',A0=a0['t_ms']/1000,A1=a1['t_ms']/1000,A2=a2['t_ms']/1000,anchor_ids=[a0['id'],a1['id'],a2['id']],downbeat_cut=a2['downbeat'],n_eighths=ev['n'],hold_eighths=ev['m'],tier=ev['tier'],
                                  t=a0['t_ms']/1000-0.5,dur=a2['t_ms']/1000-(a0['t_ms']/1000-0.5),shot=ev['i']+1,moved_s=round(ev['dist'],3),boundary_shift_eighths=ev['shift'],bga=ev.get('bga'))
    if k=='scan':
        a0,a2=A; return dict(kind='scan',variant='扫描式',A0=a0['t_ms']/1000,A2=a2['t_ms']/1000,anchor_ids=[a0['id'],a2['id']],downbeat_cut=a2['downbeat'],n_eighths=ev['n'],tier=ev['tier'],t=a0['t_ms']/1000,dur=a2['t_ms']/1000-a0['t_ms']/1000,
                             direction='h' if ev['i']%2==0 else 'v',shot=ev['i']+1,moved_s=round(ev['dist'],3),boundary_shift_eighths=ev['shift'],bga=ev.get('bga'))
    if k=='in' and ev.get('opening'):
        return dict(kind='in',variant='凝出',opening=True,B0=ev['t_B0'],A0=ev['tA0'],A1=ev['tA1'],anchor_ids=[None,None,None],t=ev['t_B0'],dur=ev['tA1']-ev['t_B0'],shot=1,tier=0,note=ev['note'],moved_s=0.0)
    if k=='in':
        a,a0,a1=A; return dict(kind='in',variant='凝出',B0=a['t_ms']/1000,A0=a0['t_ms']/1000,A1=a1['t_ms']/1000,anchor_ids=[a['id'],a0['id'],a1['id']],hold_eighths=ev['h'],tier=ev['tier'],t=a['t_ms']/1000,dur=a1['t_ms']/1000-a['t_ms']/1000,shot=ev['i']+1,moved_s=round(ev['dist'],3),boundary_shift_eighths=ev['shift'])
    if k=='final':
        return dict(kind='final',variant='收束',A0=ev['A0'],anchor_ids=[ev['anchor']['id'] if ev.get('anchor') else None],t=ev['A0']-0.5,dur=ev['t_end']-(ev['A0']-0.5),t_end=ev['t_end'],shot=ev['i']+1,tier=ev['tier'],note=ev['note'])
    return ev
moved_loc=resolve_conflicts(loc); print('局部溶解因硬切推迟到下一锚点：',moved_loc,'；去掉重叠的定格重复',len(dropped_stutter),'处')
sched=[ser(e) for e in events]+loc; sched.sort(key=lambda e:e['t'])
json.dump(dict(events=sched,bga_accent=bga_rows,warn=warn,delta_ms=DELTA*1000),open('work/glyph_schedule_v5.json','w'),ensure_ascii=False,indent=0)
# ---------- 样 E：与 sample_D 同一段 EDL 真实镜头 25–29；在这段里求解 4 个变体
shots_E=[24,25,26,27,28]     # 镜头 25–29 的 0 基索引
E_used=set(); events_save=list(events); ub_save=set(used_bound); ov_save=dict(override)
events.clear(); used_bound.clear(); override.clear()
for k,v in ov_save.items():                       # E 独立求解：保留全片的吸附（同边界已吸附则沿用）
    override[k]=v
evE=[]
allowed_out=[24,25,26]; 
e_out=solve('out',shot_end_t(24),wide=4.0,allowed=allowed_out); 
if e_out: commit(e_out); evE.append(e_out)
e_scan=solve('scan',shot_end_t(26),wide=4.0,allowed=[25,26,27]);
if e_scan: commit(e_scan); evE.append(e_scan)
e_in=solve_in(shot_start_t(27),wide=4.0,allowed=[26,27,28])
if e_in: commit(e_in); evE.append(e_in)
t_lo,t_hi=shot_start_t(28),shot_end_t(28)
cand=[(float(t),float(s)) for t,s in zip(FT,FS) if t_lo+0.3<=t<=t_hi-0.4 and s>=1.6]
cand.sort(key=lambda x:-x[1]); pick=[]
for t,s in cand:
    if all(abs(t-p)>=0.95 for p in pick): pick.append(t)
    if len(pick)==3: break
pick.sort(); used_rect.clear()
locE=local_events(pick,[1,2,3]); 
moved_E=resolve_conflicts(locE); print('E 内冲突：',moved_E)
schedE=[ser(e) for e in evE]+locE; schedE.sort(key=lambda e:e['t'])
# E 的边界吸附也要写入 EDL（E 求解后覆盖已写出的 edl.csv 前，需重新写）
def write_edl():
    for i,s in enumerate(shots):
        t0,t1=shot_start_t(i),shot_end_t(i); dur=t1-t0; s['t0n'],s['t1n']=t0,t1
        s['v']=s['d']/(dur-RAMP_G) if s['ramp'] else s['d']/dur
        lim=(0.55,1.15) if s['seg']=='outro' else (0.85,1.15)
        if not(lim[0]-1e-6<=s['v']<=lim[1]+1e-6): warn.append((s['n'],round(s['v'],3)))
    with open('work/edl.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(cols)
        for i,s in enumerate(shots):
            sp=f"ramp:{s['v']:.4f}:hit={s['hit_rel']:.3f}:dip=0.5:w=0.5" if s['ramp'] else f"{s['v']:.4f}"
            a_s=override.get((i-1,'end'),(None,''))[1] if i>0 else ''; a_e=override.get((i,'end'),(None,''))[1]
            w.writerow([s['n'],s['seg'],f"{s['t0n']:.3f}",f"{s['t1n']:.3f}",round((s['t1n']-s['t0n'])*1000),s['k'],s['b'],s['clip'],s['src_in'],s['src_out'],sp,s['win'],s['bga_new'],' '.join(s['fx']),s['trans'],'0',s['flash'],a_s if a_s is not None else '',a_e if a_e is not None else ''])
write_edl()
json.dump(dict(t0=shot_start_t(24),t1=shot_end_t(28),shots=[25,26,27,28,29],events=schedE),open('work/glyph_schedule_E.json','w'),ensure_ascii=False,indent=0)
print('警告(调速越界)：',warn)
print('E: 段',round(shot_start_t(24),3),round(shot_end_t(28),3))
for e in schedE: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in e.items() if k!='rects'})
print('--- 全片完整溶解')
for e in sched:
    if e['kind']!='local': print({k:(round(v,3) if isinstance(v,float) else v) for k,v in e.items()})
print('局部溶解',len(loc),'次，累计 %.2f s，块 %d'%(sum(e['dur'] for e in loc),sum(len(e['rects']) for e in loc)))
