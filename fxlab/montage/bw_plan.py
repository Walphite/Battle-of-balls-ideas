"""事件式黑白计划 → work/bw_plan.json（仅游戏层；BGA 文字保持原色）"""
import csv,json,numpy as np
R=list(csv.DictReader(open('work/edl_v10.csv',encoding='utf-8')))
SC=json.load(open('work/glyph_schedule_v10.json'))['events']
TOT=278.8; sh={int(r['shot']):(float(r['t_start']),float(r['t_end'])) for r in R}
bw={}; why={}; fade=[]
# (a) 完整溶解(out/scan/f)覆盖 100% 后硬切 → 下一镜头 0.4 s 由 0 褪回（in 无硬切，不加）
for e in SC:
    if e['kind'] in('out','scan'): fade.append([round(e['A2'] if e['kind']=='out' else e['A2'],4),0.4]);
    if e['kind']=='f': fade.append([round(e['M'],4),0.4])
# (b) 人声减弱区 123–188：整镜黑白，比例 15%→60% 随镜头序号上升
zone=[n for n,(a,b) in sh.items() if 100<=a<188]; zone.sort(); N=len(zone)
rng=np.random.default_rng(10)
for i,n in enumerate(zone):
    u=i/max(1,N-1); p=0.15+0.45*u
    # 确定性：累积误差扩散
    acc=getattr(bw_plan:=None,'x',0) if False else 0
cum=0.3; chosen=[]
for i,n in enumerate(zone):
    u=i/max(1,N-1); cum+=0.17+0.50*u
    if cum>=1: cum-=1; chosen.append(n)
# 避免 f 事件所在镜头冲突（f 本身覆盖期已黑白）；
for n in chosen: bw[str(n)]=0.03; why[n]='人声减弱区整镜黑白(按序号比例)'
# (d) 收束：最后 4 个镜头逐个黑白，最后一镜溶入黑
last=sorted(sh)[-4:]
for n in last: bw[str(n)]=0.0; why[n]='收束逐镜黑白'
json.dump(dict(bw_shots=bw,fadeback=fade,why={str(k):v for k,v in why.items()}),open('work/bw_plan.json','w'),ensure_ascii=False)
# 统计
dur=lambda n: sh[n][1]-sh[n][0]
z0=sum(dur(n) for n in zone); zb=sum(dur(n) for n in chosen)
print('zone shots',N,'chosen',len(chosen),'chosen s %.1f / zone s %.1f'%(zb,z0))
print('total bw-shots s %.1f'%sum(dur(int(n)) for n in bw))
