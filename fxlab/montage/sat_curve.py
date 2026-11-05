"""崩坏滤镜控制点（时间 s, 饱和度 %）→ work/sat_curve.csv。100–188 s 的下降由「混乱度」累计驱动（起音密度 + 节拍相干度）。"""
import numpy as np,pickle,json,os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
A=pickle.load(open('work/audio_an.pkl','rb')); P=pickle.load(open('work/an4.pkl','rb')); AF=22050/256
R8=P['R8s']; dens=A['dens']
t=np.arange(100,188.01,0.5); i=np.clip((t*AF).astype(int),0,len(R8)-1)
chaos=np.clip(0.5*(1-np.clip(R8[i]/0.6,0,1))+0.5*np.clip(dens[i]/9.0,0,1),0,1); chaos=np.convolve(chaos,np.ones(9)/9,'same')
Pc=np.cumsum(chaos); Pc=(Pc-Pc[0])/(Pc[-1]-Pc[0])
pts=[(0,100),(100,100)]
for tt in range(110,189,11): pts.append((float(tt),round(float(100-35*np.interp(tt,t,Pc)),1)))
if pts[-1][0]!=188: pts.append((188.0,round(float(100-35*Pc[-1]),1)))
AN=json.load(open('work/anchors.json'))['anchors']; a208=min((a['t_ms']/1000 for a in AN if a['tier']==1 and a['t_ms']/1000>=207.6),key=lambda x:abs(x-208.0))
pts+= [(197.0,76.0),(a208-0.8,85.0),(round(a208-0.3,3),85.0),(round(a208,3),100.0),(230.0,100.0),(242.0,82.0),(254.0,68.0),(266.0,60.0),(272.0,34.0),(277.0,6.0),(278.806,0.0)]
pts=sorted(set(pts)); 
with open('work/sat_curve.csv','w') as f:
    f.write('time_s,saturation_pct\n'); f.writelines(f'{a:.3f},{b:.1f}\n' for a,b in pts)
tt=np.arange(0,278.8,1/30); v=np.interp(tt,[p[0] for p in pts],[p[1] for p in pts]); print('控制点',pts); print('饱和度<70%% 累计 %.1f s（≤%.1f s）'%((v<70).sum()/30,0.35*278.8)); print('弹回锚点 %.3f s'%a208)
