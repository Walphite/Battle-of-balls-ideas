import numpy as np,sys,os,pickle,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as V
FPS=V.FPS; A=pickle.load(open('work/audio_an.pkl','rb')); per=A['per']; t0=A['t0']; T=A['T']
x=V.load(V.MUS); mag=V.stft(x)
n=len(A['env']); t=np.arange(n)/FPS
fl={k:V.flux(mag,*r) for k,r in dict(low=(30,200),mid=(300,3500),high=(3500,10000),full=(30,10000)).items()}
e=np.maximum(V.z(fl['full']),0); w=int(4*FPS)
ph=np.exp(2j*np.pi*(t-t0)/(per/2)); Z=np.convolve(e*ph,np.ones(w),'same')/(np.convolve(e,np.ones(w),'same')+1e-9)
R8=np.abs(Z); R8s=V.smooth(R8,2.0)
ai=A['ai']; ai4=V.smooth(ai,4.0)
# --- 逐秒标签
N=int(T); lab=[]
for s in range(N):
    i=int((s+.5)*FPS); r=R8s[i]; a=ai4[i]
    if s<6 or a<0.12: L='intro'
    elif a>=0.72: L='peak'
    elif r>=0.45: L='regular'
    elif r<0.33: L='irregular'
    else: L='mixed'
    lab.append(L)
# 收束：最后 12 秒
for s in range(N-12,N): lab[s]='outro'
# 众数平滑（5 秒）+ 合并短段
from collections import Counter
sm=[Counter(lab[max(0,s-2):s+3]).most_common(1)[0][0] for s in range(N)]
for s in range(N-12,N): sm[s]='outro'
seg=[];st=0
for s in range(1,N+1):
    if s==N or sm[s]!=sm[st]: seg.append([st,s,sm[st]]); st=s
# <6s 的短段并入前一段
m=[]
for a,b,L in seg:
    if m and (b-a<6): m[-1][1]=b
    elif m and m[-1][2]==L: m[-1][1]=b
    else: m.append([a,b,L])
m[-1][1]=T
# 每段残差：起音相对 8 分拍自由相位网格
pk=A['pk']; q=per/2
def resid(a,b):
    p=pk[(pk>=a)&(pk<b)]
    if len(p)<6: return None,None
    ph_=np.exp(2j*np.pi*(p-t0)/q).mean(); mu=np.angle(ph_)/(2*np.pi)*q
    d=((p-t0-mu)/q); d=(d-np.round(d))*q*1000
    return float(np.median(np.abs(d))),float(mu*1000)
segs=[]
for a,b,L in m:
    r,mu=resid(a,b); segs.append(dict(start=round(a,1),end=round(b,1),type=L,resid_ms=None if r is None else round(r,1),drift_ms=None if mu is None else round(mu),R8=round(float(R8s[int(a*FPS):int(b*FPS)].mean()),2),ai=round(float(ai[int(a*FPS):int(b*FPS)].mean()),2)))
# 规则区若残差>30ms 改为 mixed
for s in segs:
    if s['type']=='regular' and (s['resid_ms'] is None or s['resid_ms']>30): s['type']='regular?(resid>30)'
# --- 人声/采样起音
sc=V.z(fl['mid'])-0.8*np.maximum(V.z(fl['low']),V.z(fl['high']))
pks=V.peaks(sc,1.5,int(0.25*FPS)); order=pks[np.argsort(-sc[pks])][:30]
voc=[(round(i/FPS,2),round(float(sc[i]),1)) for i in sorted(order)]
# --- BGA
d,lum,dens,samp=V.bga()
cols=V.kmeans(samp)
lag,pk1,pk2,_=V.offset()
fps_b=30
dz=V.z(d); sw=V.peaks(dz,4.0,int(1.0*fps_b)); sw_t=sw/fps_b
sw_s=sorted([(round(i/fps_b,2),round(float(dz[i]),1)) for i in sw],key=lambda p:-p[1])
pickle.dump(dict(R8s=R8s,ai=ai,fl=fl,segs=segs,voc=voc,d=d,lum=lum,dens=dens,cols=cols,lag=lag,sw_t=sw_t),open('work/an4.pkl','wb'))
print(json.dumps(segs,ensure_ascii=False,indent=0))
print('voc',voc)
print('BGA frames',len(d),'colors',cols,'lag(music-bga) s',lag,'xcorr peak',pk1,'3rd',pk2)
print('switches',len(sw_t),sw_s[:25])
print('bga diff mean',d.mean(),'lum',lum.mean())
