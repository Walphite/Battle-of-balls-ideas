"""锚点：强度 ≥70 分位、与（校正后的）八分拍网格偏差 ≤30ms 的精细起音；输出 work/anchors.json。
校正：旧网格取自粗起音（STFT 窗起点约定），相对真实起音系统性早约 60ms —— δ 由强起音相对旧八分拍网格的偏差分布测得。"""
import numpy as np,json,os,sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
Z=np.load('work/onsets_fine.npz'); t=Z['t']; s=Z['s']
OLD=json.load(open('work/beats_v4_pre_anchor.json')); BT_old=np.array(OLD['beats']); per=float(np.diff(BT_old).mean()); q=per/2
E8_old=np.sort(np.r_[BT_old,(BT_old[:-1]+BT_old[1:])/2])
def dev_to(grid,x):
    j=np.clip(np.searchsorted(grid,x),1,len(grid)-1); a=grid[j-1]; b=grid[j]; return np.where(np.abs(x-a)<np.abs(x-b),x-a,x-b)
thr=float(np.percentile(s,70)); thr2=float(np.percentile(s,50)); hi=s>=thr; dv=dev_to(E8_old,t[hi])
cl=dv[(dv>0.03)&(dv<0.09)]; DELTA=float(np.median(cl)); print('强度 70 分位阈值 z=%.2f；δ（真实起音相对旧网格）=%.1f ms（样本 %d / 强起音 %d）'%(thr,DELTA*1000,len(cl),hi.sum()))
BT=BT_old+DELTA; E8=E8_old+DELTA
# 锚点
ids=[]; an=[]
for i in np.where(s>=thr2)[0]:
    g=int(np.argmin(np.abs(E8-t[i]))); d=t[i]-E8[g]
    if abs(d)<=0.030: an.append(dict(t=float(t[i]),strength=float(s[i]),g=g,dev_ms=float(d*1000),tier=1 if s[i]>=thr else 2))
# 同一网格点多个起音：保留最强
best={}
for a in an:
    if a['g'] not in best or a['strength']>best[a['g']]['strength']: best[a['g']]=a
an=sorted(best.values(),key=lambda a:a['t'])
# 重拍相位（小节 = 4 拍）：偶数 g（拍点）上的强度和按 (g//2)%4 分组
sc=np.zeros(4)
for a in an:
    if a['g']%2==0 and a['tier']==1: sc[(a['g']//2)%4]+=a['strength']
phase=int(np.argmax(sc)); print('小节相位得分',sc.round(1),'→ 重拍 = 拍序号%4=='+str(phase),'（最强/次强 %.2f）'%(sc.max()/np.sort(sc)[-2]))
an1=[a for a in an if a['tier']==1]
for k,a in enumerate(an):
    a['id']=k; a['t_ms']=round(a['t']*1000,1); a['downbeat']=bool(a['g']%2==0 and (a['g']//2)%4==phase); a['on_beat']=bool(a['g']%2==0)
json.dump(dict(delta_ms=DELTA*1000,threshold_z=thr,threshold_tier2_z=thr2,bar_phase=phase,per=per,anchors=[dict(id=a['id'],t_ms=a['t_ms'],strength=round(a['strength'],2),grid_idx=a['g'],downbeat=a['downbeat'],on_beat=a['on_beat'],dev_ms=round(a['dev_ms'],1),tier=a['tier']) for a in an]),open('work/anchors.json','w'),ensure_ascii=False)
json.dump(dict(beats=[round(float(x),4) for x in BT],per=per,t0=float(BT[0]),lead=0.010,delta=DELTA,note='校正后：旧网格 + δ'),open('work/beats_v4.json','w'))
print('锚点(一级)数',len(an1),'二级补充',len(an)-len(an1),'；一级中拍点上',sum(a['on_beat'] for a in an1),'重拍',sum(a['downbeat'] for a in an1),'占全部八分拍点 %.0f%%'%(100*len(an1)/len(E8)))
for lo,hi_ in ((0,16),(16,79),(79,100),(100,188),(188,202),(202,230),(230,279)):
    m=[a for a in an1 if lo<=a['t']<hi_]; m2=[a for a in an if lo<=a['t']<hi_]; print(' ',lo,hi_,'一级锚点',len(m),'（%.1f/s）'%(len(m)/(hi_-lo)),'含二级',len(m2))
# 旧 EDL 边界（对应拍序号）处是否有锚点
import csv
R=list(csv.DictReader(open('work/edl_v4_pre_anchor.csv',encoding='utf-8'))); gs={a['g']:a for a in an if a['tier']==1}
bs=sorted({int(r['beat_start']) for r in R}|{int(r['beat_start'])+int(r['beats']) for r in R})
have=[b for b in bs if 2*b in gs]; print('EDL 边界 %d 条，其拍点上有锚点 %d 条（%.0f%%），其中重拍 %d'%(len(bs),len(have),100*len(have)/len(bs),sum(gs[2*b]['downbeat'] for b in have)))
