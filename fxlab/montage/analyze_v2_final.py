import numpy as np,sys,os,pickle,json,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as V
import beat as B
FPS=V.FPS; A=pickle.load(open('work/audio_an.pkl','rb')); P=pickle.load(open('work/an4.pkl','rb'))
per=A['per'];t0=A['t0'];T=A['T']
# ---- 偏移精修：22.05k 原始波形互相关（lag 0.1625±40ms，取 0.1s–60s 内 48 秒窗）
a=B.load_mono(V.MUS,22050); b=B.load_mono(V.BGA,22050)
seg=slice(int(10*22050),int(58*22050))
best=None
for k in range(int(0.12*22050),int(0.21*22050)):
    s=float(np.dot(a[seg],b[seg.start+k:seg.stop+k])); 
    if best is None or s>best[0]: best=(s,k)
lag=best[1]/22050; nrm=best[0]/np.linalg.norm(a[seg])/np.linalg.norm(b[seg.start+best[1]:seg.stop+best[1]])
print('refined lag s',lag,'wave corr',nrm)
# ---- 重新分段
R8s=P['R8s'];ai=A['ai'];ai4=V.smooth(ai,4.0);n=len(ai);N=int(T)
lab=[]
for s in range(N):
    i=int((s+.5)*FPS);r=R8s[i];x=ai4[i]
    if s<6 or x<0.12:L='intro'
    elif r<0.30:L='irregular'
    elif x>=0.72:L='peak'
    elif r>=0.45:L='regular'
    else:L='mixed'
    lab.append(L)
from collections import Counter
for s in range(N-12,N):lab[s]='outro'
sm=[Counter(lab[max(0,s-3):s+4]).most_common(1)[0][0] for s in range(N)]
for s in range(N-12,N):sm[s]='outro'
sg=[];st=0
for s in range(1,N+1):
    if s==N or sm[s]!=sm[st]:sg.append([st,s,sm[st]]);st=s
m=[]
for x,y,L in sg:
    if m and (y-x<8):m[-1][1]=y
    elif m and m[-1][2]==L:m[-1][1]=y
    else:m.append([x,y,L])
m[-1][1]=T
pk=A['pk'];q=per/2
rows=[]
for x,y,L in m:
    p=pk[(pk>=x)&(pk<y)]; mu=np.angle(np.exp(2j*np.pi*(p-t0)/q).mean())/(2*np.pi)*q
    d=(p-t0-mu)/q; d=(d-np.round(d))*q*1000
    rows.append(dict(start=round(float(x),1),end=round(float(y),1),type=L,resid_ms=round(float(np.median(np.abs(d))),1),drift_ms=round(float(mu*1000)),R8=round(float(R8s[int(x*FPS):int(y*FPS)].mean()),2),ai=round(float(ai[int(x*FPS):int(y*FPS)].mean()),2),onsets_per_s=round(len(p)/(y-x),1)))
for r in rows:print(r)
# ---- BGA 强度 -> 配乐时间轴，10Hz
d,lum,dens=P['d'],P['lum'],P['dens']
def sm_(a,s,f=30):w=int(s*f);return np.convolve(a,np.ones(w)/w,'same')
bi=0.5*V.norm01(sm_(d,0.5))+0.5*V.norm01(sm_(dens,0.5)*0.5+sm_(lum,0.5)*0.5/ max(lum.max(),1))
tt=np.arange(0,T,0.1); idx=np.clip(np.round((tt+lag)*30).astype(int),0,len(bi)-1)
bi10=bi[idx]
ai10=ai[np.clip(np.round(tt*FPS).astype(int),0,n-1)]
ai10=V.norm01(ai10)
I=0.5*ai10+0.5*V.norm01(bi10); I=np.convolve(I,np.ones(5)/5,'same')
json.dump(dict(dt=0.1,bga_offset_s=lag,note='I(t)=0.5*audio+0.5*bga, 配乐时间轴, 0.5s 平滑',audio=np.round(ai10,3).tolist(),bga=np.round(V.norm01(bi10),3).tolist(),I=np.round(I,3).tolist()),open('work/intensity.json','w'))
print('corr audio vs bga intensity',np.corrcoef(ai10,bi10)[0,1])
# ---- 边界对照
sw=[float(s-lag) for s in P['sw_t']]   # BGA 切换点换算到配乐时间
bnd=[r['start'] for r in rows[1:]]
print('BGA switches (music time):',[round(s,2) for s in sw])
for bd in bnd:
    j=int(np.argmin([abs(s-bd) for s in sw]));print('audio boundary %.1f  nearest BGA switch %.2f  diff %d ms'%(bd,sw[j],round((sw[j]-bd)*1000)))
for s in sorted(sw):
    j=int(np.argmin([abs(s-bd) for bd in bnd]));print('BGA switch %.2f  nearest audio boundary %.1f  diff %d ms'%(s,bnd[j],round((s-bnd[j])*1000)))
pickle.dump(dict(rows=rows,lag=lag),open('work/an5.pkl','wb'))
