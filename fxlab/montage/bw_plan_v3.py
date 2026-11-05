"""bw_keyframes.csv → work/bw_pulses.json（吸附锚点、整数八分拍）。可手改 csv 后重跑：python3 -I bw_plan_v3.py
对齐目标：#1 → BGA「<CONNECTION LOST>」打完的时刻（实测 ≈124.8 s）；#2 → BGA 切换点 144.30 s；#4 → BGA 切换点 187.20 s。"""
import csv,json,os,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
TARGET={1:124.80,2:144.30,4:187.20}          # 实测（见 README）
A=[a for a in json.load(open('work/anchors.json'))['anchors'] if a['tier']<=2]; AT=np.array([a['t_ms']/1000 for a in A]); TI=[a['tier'] for a in A]
BT=np.array(json.load(open('work/beats_v4.json'))['beats']); log=[]; P=[]
def near(t,lim):
    i=int(np.argmin(np.abs(AT-t))); return (AT[i],A[i]['id'],A[i]['tier']) if abs(AT[i]-t)<=lim else None
for r in csv.DictReader(open('bw_keyframes.csv')):
    pid=int(r['pulse_id']); nom=float(r['peak_time_s']); dn=int(r['down_eighths']); hold=int(r['hold_eighths']); up=int(r['up_eighths'])
    tg=TARGET.get(pid,nom); j=int(np.searchsorted(BT,tg)); j=min(max(j,1),len(BT)-1); e=(BT[j]-BT[j-1])/2
    a=near(tg,0.5)
    if a: pk,aid,tier=a; note='吸附锚点 #%d(tier%d)，相对%s %.0f ms'%(aid,tier,'对齐目标' if pid in TARGET else '名义时刻',(pk-tg)*1000)
    else: pk=tg; aid=None; note='±0.5 s 内无合格锚点，保持 %.3f'%tg
    t0=pk-dn*e
    if hold<0: r0=None; r1=None
    else:
        want=pk+hold*e; b=near(want,1.5*e); r0=b[0] if b else want; r1=r0+up*e
    P.append(dict(id=pid,nominal=nom,target=tg,t_fall0=round(t0,4),t_peak=round(pk,4),t_rise0=None if r0 is None else round(r0,4),t_rise1=None if r1 is None else round(r1,4),eighth=round(e,4),peak_anchor=aid,note=note,rise_anchor=bool(hold>=0 and b),hold_s=None if r0 is None else round(r0-pk,3)))
json.dump(dict(pulses=P),open('work/bw_pulses.json','w'),ensure_ascii=False,indent=1)
for p in P: print(p)
tot=sum(((p['t_rise1'] if p['t_rise1'] else 278.8)-p['t_fall0']) for p in P); print('S<1 区间总长 %.1f s (%.1f%%)'%(tot,tot/2.788))
