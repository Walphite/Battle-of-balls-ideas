"""V10 字符溶解调度：人声减弱区 [123,188] 内复用溶出/扫描式/凝出并新增交叉溶解 f；覆盖峰值与时长随区内进度递增；
 区外沿用 glyph_schedule_v5.json（去掉落在区内的旧事件）。输出 work/glyph_schedule_v10.json、work/zone_overrides.json（锚点吸附的边界时间）、work/zone_info.json"""
import json,csv,os,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ZS,ZE=123.0,188.0
AN=json.load(open('work/anchors.json')); anch={a['grid_idx']:a for a in AN['anchors'] if a['tier']<=2}
def anc(g): return anch.get(g)
R=list(csv.DictReader(open('work/edl_v9.csv',encoding='utf-8')))
OLD=json.load(open('work/glyph_schedule_v5.json'))['events']
bounds=[dict(i=i,t=float(R[i]['t_end']),g=2*(int(R[i]['beat_start'])+int(R[i]['beats'])),st=float(R[i]['t_start']),en=float(R[i+1]['t_end'])) for i in range(len(R)-1)]
def T(a): return a['t_ms']/1000
def opt_f(b):
    best=None
    for sh in (0,-1,1):
        for n in (3,2,4):
            for n2 in (3,2,4):
                a0,m,a1=anc(b['g']+sh-n),anc(b['g']+sh),anc(b['g']+sh+n2)
                if a0 and m and a1:
                    if T(a0)<b['st']+0.05 or T(a1)>b['en']-0.05: continue
                    c=3*abs(sh)+abs(n-3)+abs(n2-3)
                    if best is None or c<best[0]: best=(c,sh,n,n2,a0,m,a1)
    return best
def opt_scan(b,u):
    for n in ((3,4) if u<0.5 else (4,3)):
        a0,a2=anc(b['g']-n),anc(b['g'])
        if a0 and a2 and T(a0)>b['st']+0.05: return (n,a0,a2)
def opt_out(b,u):
    for n in ((4,6) if u<0.5 else (6,4)):
        for m in (2,3):
            a0,a1,a2=anc(b['g']-m-n),anc(b['g']-m),anc(b['g'])
            if a0 and a1 and a2 and T(a0)-0.5>=b['st']+0.02: return (n,m,a0,a1,a2)
def opt_in(b,u):
    # 凝出：起点=下一镜头开头(本边界)，A0=+h，A1=A0+2 个八分拍
    for h in (2,3,4):
        a=anc(b['g']); a0=anc(b['g']+h); a1=anc(b['g']+h+2)
        if a and a0 and a1 and T(a1)<b['en']-0.2: return (h,a,a0,a1)
events=[]; spans=[]; over={}
def free(t0,t1): return all(t1<s0-0.15 or t0>s1+0.15 for s0,s1 in spans)
types=[]
for b in [x for x in bounds if ZS<=x['t']<=ZE]:
    u=(b['t']-ZS)/(ZE-ZS); pmax=round(min(1.0,0.4+0.6*u),2); last2=types[-2:]
    cands={}
    f=opt_f(b)
    if f:
        c,sh,n,n2,a0,m,a1=f
        if free(T(a0),T(a1)): cands['f']=(f,)
    s=opt_scan(b,u)
    if s and free(T(s[1]),T(s[2])): cands['scan']=(s,)
    o=opt_out(b,u)
    if o and free(T(o[2])-0.5,T(o[4])): cands['out']=(o,)
    i_=opt_in(b,u)
    if i_ and free(T(i_[1]),T(i_[3])): cands['in']=(i_,)
    if not cands: continue
    if len(types)>=12: continue                                           # 区内至多 12 个完整溶解（全片完整溶解合计 ≤18）
    pick=None
    if len(last2)==2 and last2[0]==last2[1]=='f':                         # 两个交叉溶解之后必须换一种变体（同一种不得连续超过 2 次）
        rot=[t_ for t_ in types if t_!='f']; seq=['out','in','scan']
        if rot and rot[-1] in seq: i0=seq.index(rot[-1]); seq=seq[i0+1:]+seq[:i0+1]      # 分隔用的变体轮换，尽量用到溶出/凝出/扫描式
        for ty in seq:
            if ty in cands: pick=ty; break
    elif 'f' in cands: pick='f'
    elif last2[-1:]==['f'] and False: pass
    if pick is None: continue
    v=cands[pick][0]; types.append(pick)
    if pick=='f':
        c,sh,n,n2,a0,m,a1=v; ev=dict(kind='f',variant='交叉溶解',A0=T(a0),M=T(m),A1=T(a1),anchor_ids=[a0['id'],m['id'],a1['id']],n_before=n,n_after=n2,boundary_shift_eighths=sh,shot=b['i']+1,t=T(a0),dur=T(a1)-T(a0),pmax=1.0)
        spans.append((T(a0),T(a1))); over[b['i']+1]=T(m)           # 边界索引 = 第 i 个镜头之后的边界 → j=i+1
    elif pick=='scan':
        n,a0,a2=v; ev=dict(kind='scan',variant='扫描式',A0=T(a0),A2=T(a2),anchor_ids=[a0['id'],a2['id']],n_eighths=n,direction='h' if len(events)%2==0 else 'v',shot=b['i']+1,t=T(a0),dur=T(a2)-T(a0),pmax=pmax)
        spans.append((T(a0),T(a2))); over[b['i']+1]=T(a2)
    elif pick=='out':
        n,m_,a0,a1,a2=v; ev=dict(kind='out',variant='溶出',A0=T(a0),A1=T(a1),A2=T(a2),anchor_ids=[a0['id'],a1['id'],a2['id']],n_eighths=n,hold_eighths=m_,shot=b['i']+1,t=T(a0)-0.5,dur=T(a2)-T(a0)+0.5,pmax=pmax)
        spans.append((T(a0)-0.5,T(a2))); over[b['i']+1]=T(a2)
    else:
        h,a,a0,a1=v; ev=dict(kind='in',variant='凝出',B0=T(a),A0=T(a0),A1=T(a1),anchor_ids=[a['id'],a0['id'],a1['id']],hold_eighths=h,shot=b['i']+2,t=T(a),dur=T(a1)-T(a),pmax=pmax)
        spans.append((T(a),T(a1))); over[b['i']+1]=T(a)
    ev['zone_progress']=round(u,2); events.append(ev)
# 区外事件沿用；区内旧事件（143 溶出、185 扫描）去掉
keep=[e for e in OLD if not(e['kind']!='local' and ZS-1<=e['t']<=ZE+1)]
sched=sorted(keep+events,key=lambda e:e['t'])
def span(e):
    k=e['kind']; return (e['A0']-0.5 if k=='out' else (e['B0'] if k=='in' else e['A0']), e['A2'] if k in('out','scan') else (e['A1'] if k in('in','f') else e['t_end']))
full=[e for e in sched if e['kind']!='local']; tot=sum(span(e)[1]-span(e)[0] for e in full)
info=dict(zone=[ZS,ZE],n_zone_events=len(events),types=types,n_f=types.count('f'),full_total=len(full),full_seconds=round(tot,2),local_seconds=round(sum(e['dur'] for e in sched if e['kind']=='local'),2),
          f_params=[(round(e['A0'],3),e['n_before'],e['n_after'],round(e['A1']-e['A0'],3)) for e in events if e['kind']=='f'])
json.dump(dict(events=sched,bga_accent=[],warn=[],delta_ms=AN['delta_ms']),open('work/glyph_schedule_v10.json','w'),ensure_ascii=False,indent=0)
json.dump({str(k):v for k,v in over.items()},open('work/zone_overrides.json','w')); json.dump(info,open('work/zone_info.json','w'),ensure_ascii=False)
print(json.dumps(info,ensure_ascii=False)); print('事件',[(e['kind'],round(e['t'],2),round(e['dur'],2),e.get('pmax')) for e in events])
