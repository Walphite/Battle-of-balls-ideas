import os,sys,json,csv
os.environ['EDL']='work/edl_v10.csv'; os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import numpy as np, render_c as RC, render_v4 as R4
sched=json.load(open('work/glyph_schedule_v10.json'))['events']; comp=RC.Composer(960,540,sched); shots=R4.load_edl()
bw=json.load(open('work/bw_plan.json')); N=8364; S=np.ones(N); cause=['']*N
for k in range(N):
    t=(k+0.5)/30; s=next((x for x in shots if x['t']<=t<x['te']),None)
    if s is None: S[k]=0; continue
    v=comp.sat_shot(s['n'],t)
    for e in comp.sched:
        st=comp.state(e,t)
        if st is None: continue
        if e['kind'] in('out','in','f','scan'):
            pp=st.get('p',st.get('front',0)) or 0; v=min(v,1-pp)
    # 局部黑白：窗口局部，不计入整帧
    S[k]=v
low=S<=0.05; sec=low.sum()/30
def rng(a,b): i,j=int(a*30),int(b*30); return low[i:j].sum()/30, (j-i)/30
print('整片 sat<=0.05: %.1fs (%.1f%%)'%(sec,100*sec/278.8))
for a,b in((100,188),(123,188),(0,100),(188,278.8),(266,278.8)): x,d=rng(a,b); print(a,b,'%.1fs / %.1fs = %.0f%%'%(x,d,100*x/d))
# 每秒表
persec=[float(low[i*30:(i+1)*30].mean()) for i in range(278)]
json.dump(dict(total_s=sec,persec=persec),open('work/v10/bw_metrics.json','w'))
# 黑白镜头列表
rows=[]
H={e['shot']:e for e in sched if e['kind']=='f'}
for n,v in sorted(bw['bw_shots'].items(),key=lambda kv:int(kv[0])):
    s=next(x for x in shots if x['n']==int(n)); rows.append((int(n),round(s['t'],2),round(s['te'],2),bw['why'][n]))
for e in sched:
    if e['kind']=='f': rows.append((e['shot'],round(e['A0'],2),round(e['A1'],2),'f 交叉溶解覆盖期（游戏层黑白）+ 之后0.4s褪回'))
    if e['kind'] in('out','scan'): rows.append((e['shot'],round(e['A0'],2),round(e['A2'],2),e['variant']+'覆盖期黑白 + 硬切后0.4s褪回'))
    if e['kind']=='local' and 202<=e['t']<219.6: pass
print(len([e for e in sched if e['kind']=='local' and 202<=e['t']<219.6]),'个局部黑白矩形事件(202–219.5)')
rows.sort(key=lambda r:r[1]); 
with open('../docs/bw_shots_v10.md','w') as f:
    f.write('# V10 黑白镜头清单\n\n整片 sat≤0.05 累计 %.1f s (%.1f%%)\n\n| 镜头 | 起 | 止 | 原因 |\n|--|--|--|--|\n'%(sec,100*sec/278.8))
    for r in rows: f.write('| %d | %.2f | %.2f | %s |\n'%r)
    f.write('\n局部黑白(202–219.5)：%d 个矩形事件，0.08–0.17 s 各\n'%len([e for e in sched if e['kind']=='local' and 202<=e['t']<219.6]))
