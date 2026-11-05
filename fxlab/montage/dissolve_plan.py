"""字符溶解调度：从 work/edl.csv、采样起音、BGA 事件、拍点生成 work/dissolve_schedule.json/.csv（全片）与 work/dissolve_demo.json（样 D）。不改 EDL。"""
import csv,json,os,sys,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
from effects.glyph_dissolve import GlyphParams
P=GlyphParams(); T=278.806
shots=[dict(n=int(r['shot']),t=float(r['t_start']),te=float(r['t_end']),seg=r['seg']) for r in csv.DictReader(open('work/edl.csv',encoding='utf-8'))]
BEATS=np.array(json.load(open('work/beats_v4.json'))['beats']); VO=json.load(open('work/vocal_onsets.json'))
BGA_EV=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]
def shot_of(t): return next(s for s in shots if s['t']-1e-6<=t<s['te'])
def boundary_after(E): return min((s['te'] for s in shots if s['te']>=E-0.05),key=lambda x:x)
def total(kind,scale=1.0): return GlyphGlyph.total(kind,scale)
class GlyphGlyph:
    @staticmethod
    def total(kind,scale=1.0):
        p=P
        return {'out':(p.tear+p.still+p.cover+p.hold)*scale,'in':(p.hold+p.restore)*scale,'scan':p.sweep+p.sweep_hold,'final':p.final_cover+p.final_fade+p.final_black}[kind]
ev=[]
def add(kind,t0,dur,trig,**kw): ev.append(dict(kind=kind,t=round(t0,3),dur=round(dur,3),trigger=trig,**kw))
# 1) 开场：第一个镜头「凝出」
add('in',shots[0]['t'],total('in'),'开场信号接入',scale=1.0)
# 2) 规则区 BGA 点：溶出 / 扫描式 / 溶出（不连用同一种）；其余全片完整溶解：144.3 溶出、187.2 扫描式、233.1 慢速溶出
plan=[(29.77,'out',1.0,'h'),(44.57,'scan',1.0,'h'),(47.91,'out',1.0,'h'),(144.31,'out',1.0,'h'),(187.21,'scan',1.0,'v'),(233.14,'out',1.6,'h')]
for e,k,sc,d in plan:
    B=boundary_after(e); dur=total(k,sc); add(k,B-dur,dur,f'BGA {e}',scale=sc,direction=d,bga=e)
# 3) 收束：最后一个镜头完整溶解成字符点阵并熄灭
last=shots[-1]; dur=total('final'); add('final',last['te']-dur,dur,'收束·信号断开')
assert last['te']-dur>=last['t'],'最后镜头容不下收束溶解'
# 4) 局部溶解：100–113 采样簇、202–219.5 碎片点
rng=np.random.default_rng(11); used=[]
def rect():
    for _ in range(400):
        w=rng.uniform(0.18,0.38); h=rng.uniform(0.15,0.35); u=rng.uniform(0,1-w); v=rng.uniform(0,1-h); r=(u,v,u+w,v+h)
        ok=True
        for q in used[-8:]:
            iw=max(0,min(r[2],q[2])-max(r[0],q[0])); ih=max(0,min(r[3],q[3])-max(r[1],q[1])); inter=iw*ih
            if inter/((r[2]-r[0])*(r[3]-r[1])+(q[2]-q[0])*(q[3]-q[1])-inter)>0.08: ok=False;break
        if ok: used.append(r); return tuple(round(x,3) for x in r)
    used.append(r); return tuple(round(x,3) for x in r)
corp=[a for a,_ in VO['corp_100_188'] if 100<=a<=113]
frag=[]; 
for a in VO['frag_merged']:
    if 202<=a<=219.5 and (not frag or a-frag[-1]>=0.3): frag.append(a)
loc=[]
for i,a in enumerate(corp): loc.append((a,0.20,1+i%3,'采样簇 Corp-ra-corp'))
for i,a in enumerate(frag): loc.append((a,0.15 if i%3 else 0.20,1+(i%2),'碎片采样点'))
for a,dur,nb,why in loc:
    s=shot_of(a); dur=min(dur,s['te']-a-0.02)
    add('local',a,dur,why,rects=[rect() for _ in range(nb)])
ev.sort(key=lambda e:e['t'])
# 检查：无重叠、完整溶解不在峰值区 79–100 / 208–230
for a,b in zip(ev,ev[1:]):
    if a['t']+a['dur']>b['t']+1e-6 and not (a['kind']=='local' and b['kind']=='local'): print('WARN overlap',a['kind'],a['t'],b['kind'],b['t'])
for e in ev:
    if e['kind']!='local' and (79<=e['t']<100 or 208<=e['t']<230 or 79<=e['t']+e['dur']<100 or 208<=e['t']+e['dur']<230): print('WARN full dissolve in peak',e)
# 记录：触发点与最近镜头边界 / BGA 事件 / 拍点的差 (ms)
rows=[]
for e in ev:
    trig=e['t'] if e['kind'] in('local','in') else e['t']+e['dur']      # 溶出/扫描/收束以「硬切点」为参照；局部以起音、凝出以起点为参照
    ref={'local':'onset','in':'start'}.get(e['kind'],'cut-end')
    bd=min(min(abs(s['t']-trig),abs(s['te']-trig)) for s in shots); bg=min(BGA_EV,key=lambda x:abs(x-trig)); bt=BEATS[np.argmin(np.abs(BEATS-trig))]
    e['ref']=ref; e['d_boundary_ms']=round((trig-min([s['t'] for s in shots]+[s['te'] for s in shots],key=lambda x:abs(x-trig)))*1000)
    e['d_bga_ms']=round((trig-bg)*1000); e['nearest_bga']=bg; e['d_beat_ms']=round((trig-bt)*1000)
    rows.append(e)
json.dump(rows,open('work/dissolve_schedule.json','w'),ensure_ascii=False,indent=0)
with open('work/dissolve_schedule.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['kind','t_start','dur','trigger','ref','d_boundary_ms','nearest_bga','d_bga_ms','d_beat_ms'])
    for e in rows: w.writerow([e['kind'],e['t'],e['dur'],e['trigger'],e['ref'],e['d_boundary_ms'],e['nearest_bga'],e['d_bga_ms'],e['d_beat_ms']])
full=[e for e in rows if e['kind']!='local']; lo=[e for e in rows if e['kind']=='local']
from collections import Counter
st=dict(full_count=len(full),full_total_s=round(sum(e['dur'] for e in full),2),by_kind=dict(Counter(e['kind'] for e in full)),local_count=len(lo),local_total_s=round(sum(e['dur'] for e in lo),2),local_blocks=sum(len(e['rects']) for e in lo))
json.dump(st,open('work/dissolve_stats.json','w')); print(st)
for e in full: print(e['kind'],e['t'],e['dur'],e['trigger'],'cut@',round(e['t']+e['dur'],3),'dBound',e['d_boundary_ms'],'dBGA',e['d_bga_ms'],'dBeat',e['d_beat_ms'])
# ---------- 样 D 演示：5 个连续真实镜头（规则区 ~60 s 起），依次 a 溶出 / b 凝出 / c 扫描 / d 局部
i0=next(i for i,s in enumerate(shots) if s['t']>=60.0); S=shots[i0:i0+5]; demo=[]
dur=total('out'); demo.append(dict(kind='out',t=round(S[0]['te']-dur,3),dur=round(dur,3),scale=1.0,trigger='D:a 溶出（镜头尾硬切）'))
demo.append(dict(kind='in',t=round(S[1]['t'],3),dur=round(total('in'),3),scale=1.0,trigger='D:b 凝出（镜头头）'))
dur=total('scan'); demo.append(dict(kind='scan',t=round(S[2]['te']-dur,3),dur=round(dur,3),direction='h',scale=1.0,trigger='D:c 扫描式（横向，镜头尾硬切）'))
used.clear(); t4=S[3]['t']
for off,du,nb in ((0.55,0.12,1),(1.05,0.18,2),(1.55,0.25,3)):
    demo.append(dict(kind='local',t=round(t4+off,3),dur=du,rects=[rect() for _ in range(nb)],trigger=f'D:d 局部溶解 {nb} 块'))
json.dump(dict(t0=S[0]['t'],t1=S[4]['te'],shots=[s['n'] for s in S],events=demo),open('work/dissolve_demo.json','w'),ensure_ascii=False,indent=0)
print('demo shots',[s['n'] for s in S],S[0]['t'],S[4]['te'])
