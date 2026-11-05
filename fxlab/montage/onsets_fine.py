"""精细起音检测（hop 64 @22.05k = 2.9ms，NFFT 512），并用合成噪声爆破做时间校准。输出 work/onsets_fine.npz"""
import numpy as np,sys,os,pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import beat as B
SR=22050; HOP=64; N=512
def stft_mag(x):
    win=np.hanning(N).astype(np.float32); n=1+(len(x)-N)//HOP; out=np.empty((n,N//2+1),np.float32)
    for s in range(0,n,4000):
        e=min(n,s+4000); idx=np.arange(N)[None]+HOP*np.arange(s,e)[:,None]; out[s:e]=np.abs(np.fft.rfft(x[idx]*win,axis=1))
    return out
def flux(x):
    m=stft_mag(x); f=np.fft.rfftfreq(N,1/SR); band=(f>=40)&(f<9000); lm=np.log1p(30*m[:,band])
    d=np.maximum(np.diff(lm,axis=0),0).sum(1); d=np.r_[0,d]
    # 帧 i 的时间 = 窗口中心
    return d,(np.arange(len(d))*HOP+N/2)/SR
def pick(d,tt,thr_z=1.2,mind=0.045):
    z=(d-d.mean())/(d.std()+1e-9); idx=[]
    for i in range(2,len(z)-2):
        if z[i]>thr_z and z[i]>=z[i-1] and z[i]>z[i+1] and z[i]>=z[i-2] and z[i]>z[i+2]:
            if idx and tt[i]-tt[idx[-1]]<mind:
                if z[i]>z[idx[-1]]: idx[-1]=i
            else: idx.append(i)
    return np.array(idx,int),z
def calibrate():
    """合成：已知时刻的短噪声爆破（0.5ms 起、20ms 衰减）+ 低噪声底；返回 检测时间−真实时间 的中位数(s)"""
    rng=np.random.default_rng(5); T=40; x=rng.normal(0,0.003,SR*T).astype(np.float32); true=np.arange(1,T-1,0.37)+rng.uniform(0,0.01,len(np.arange(1,T-1,0.37)))
    for t in true:
        n=int(0.12*SR); env=np.exp(-np.arange(n)/(0.02*SR))*np.minimum(1,np.arange(n)/(0.0005*SR)); i=int(t*SR); x[i:i+n]+=rng.normal(0,0.5,n)*env
    d,tt=flux(x); idx,z=pick(d,tt); det=tt[idx]; err=[]
    for t in true:
        j=np.argmin(np.abs(det-t)); 
        if abs(det[j]-t)<0.05: err.append(det[j]-t)
    return float(np.median(err)),float(np.std(err)),len(err),len(true)
if __name__=='__main__':
    bias,sd,n,m=calibrate(); print('校准：检测−真实 中位 %.1f ms，σ %.1f ms (%d/%d)'%(bias*1000,sd*1000,n,m))
    x=B.load_mono('input2/music/CreditsEX.by.Frums.mp3',SR); d,tt=flux(x); idx,z=pick(d,tt)
    t=tt[idx]-bias; s=z[idx]; print('精细起音数',len(t))
    np.savez('work/onsets_fine.npz',t=t,s=s,bias=bias,fl=d,tt=tt-bias)
    A=pickle.load(open('work/audio_an.pkl','rb')); pk=A['pk']; BT=np.array(__import__('json').load(open('work/beats_v4.json'))['beats'])
    # 与旧（粗）起音时间比较：每个粗起音找最近精细起音
    off=[]; 
    for p in pk:
        j=np.argmin(np.abs(t-p)); 
        if abs(t[j]-p)<0.12: off.append(t[j]-p)
    off=np.array(off); print('精细−粗 起音时间差 (ms) 分位 10/25/50/75/90:',np.round(np.percentile(off,[10,25,50,75,90])*1000),'n',len(off))
    # 强起音相对旧拍网格（BT，八分拍）的相位偏移
    q=np.diff(BT).mean()/2; hi=s>=np.percentile(s,70); tp=t[hi]
    # 八分拍点：拍点 + 中点
    E8=np.sort(np.r_[BT,(BT[:-1]+BT[1:])/2]); dev=[]
    for a in tp:
        j=np.argmin(np.abs(E8-a)); dev.append(a-E8[j])
    dev=np.array(dev); print('强起音(前30%) 相对旧八分拍网格 偏差 (ms) 分位 10/25/50/75/90:',np.round(np.percentile(dev,[10,25,50,75,90])*1000),' ≤30ms 占比',round(float((np.abs(dev)<=0.03).mean()),2))
    for a,b in ((16,79),(79,100),(100,188),(188,230),(230,266)):
        m=(tp>=a)&(tp<b); dd=dev[m]; print(' ',a,b,'中位 %+.0f ms'%(np.median(dd)*1000),'n',m.sum(),'≤30ms %.2f'%(np.abs(dd)<=0.03).mean())
