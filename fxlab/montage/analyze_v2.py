"""第 1 步 b/c/d：配乐结构、BGA 分析、联合强度。纯 ffmpeg + numpy。复用 beat.py 的 beat_track / fit_grid。"""
import subprocess, json, sys, numpy as np
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import beat as B
SR=22050; HOP=256; NFFT=2048; FPS=SR/HOP
MUS='input2/music/CreditsEX.by.Frums.mp3'; BGA='input2/bga/Credits.bga.mp4'

def load(path,sr=SR):
    return B.load_mono(path,sr)
def stft(x):
    win=np.hanning(NFFT); n=1+(len(x)-NFFT)//HOP
    out=np.empty((n,NFFT//2+1),np.float32)
    for s in range(0,n,2000):
        e=min(n,s+2000); idx=np.arange(NFFT)[None]+HOP*np.arange(s,e)[:,None]
        out[s:e]=np.abs(np.fft.rfft(x[idx]*win,axis=1))
    return out
def flux(mag,lo,hi):
    f=np.fft.rfftfreq(NFFT,1/SR); m=(f>=lo)&(f<hi)
    lm=np.log1p(10*mag[:,m]); d=np.maximum(np.diff(lm,axis=0),0).mean(1); return np.r_[0,d]
def z(a): return (a-a.mean())/(a.std()+1e-9)
def smooth(a,sec,fps=FPS):
    w=max(1,int(sec*fps)); return np.convolve(a,np.ones(w)/w,'same')
def norm01(a): a=np.asarray(a,float); lo,hi=np.percentile(a,2),np.percentile(a,98); return np.clip((a-lo)/(hi-lo+1e-9),0,1)
def peaks(env,thr,mind):
    idx=[]; 
    for i in range(1,len(env)-1):
        if env[i]>thr and env[i]>=env[i-1] and env[i]>env[i+1]:
            if idx and i-idx[-1]<mind:
                if env[i]>env[idx[-1]]: idx[-1]=i
            else: idx.append(i)
    return np.array(idx,int)

def audio():
    x=load(MUS); T=len(x)/SR; mag=stft(x)
    env_all=flux(mag,30,8000); env=z(env_all)
    rms=np.sqrt(np.convolve(x**2,np.ones(HOP)/HOP,'same')[::HOP][:len(env)])
    # 拍点：全曲 179 先验 + 多候选取残差最小
    res={}
    prior=B.beat_track(env,FPS,179,tight=400); t0,p=B.fit_grid(prior,179)
    r179=float(np.abs(prior-(t0+p*np.round((prior-t0)/p))).mean()*1000)
    best=B.best_grid(env,FPS,lo=80,hi=200)
    res['bpm_prior179']=dict(bpm=60/p,t0=t0,resid_ms=r179)
    res['bpm_best']=dict(bpm=best['bpm'],t0=best['t0'],resid_ms=best['resid_ms'],tried=best['tried'])
    beats=prior; per=p
    # 逐拍：相对「局部线性拟合」(±8 拍) 的残差 与 相对全局网格残差
    loc=np.empty(len(beats)); glob=beats-(t0+per*np.round((beats-t0)/per))
    for i in range(len(beats)):
        a,b=max(0,i-8),min(len(beats),i+9); k=np.arange(a,b); c=np.polyfit(k,beats[a:b],1)
        loc[i]=beats[i]-np.polyval(c,i)
    # 置信度：8 秒滑窗内「拍点处起音强度 / 拍间中点起音强度」对比 + 网格残差
    n=len(env); conf=np.zeros(n); half=int(4*FPS)
    on=np.interp(np.arange(n)/FPS,beats,np.ones(len(beats)))  # dummy
    def at(ts): 
        i=np.clip(np.round(np.array(ts)*FPS).astype(int),1,n-2); return np.maximum(env_all[i-1],np.maximum(env_all[i],env_all[i+1]))
    gb=t0+per*np.arange(-int(t0/per)-1,int((T-t0)/per)+2); gb=gb[(gb>0)&(gb<T)]
    sb=at(gb); sm_=at(gb+per/2)
    # 每个网格拍的对比度，再 8s 平滑
    contrast=(sb-sm_)/(sb+sm_+1e-9)
    cb=np.interp(np.arange(n)/FPS,gb,contrast); conf=smooth(cb,8)
    # 起音点
    pk=peaks(env,1.0,int(0.04*FPS)); pt=pk/FPS
    dens=smooth(np.bincount(pk,minlength=n).astype(float)*FPS,1.0)   # 每秒起音数
    # 起音间隔规则度：8s 内起音间隔与拍长的 1/4 倍数偏差
    ioi_irreg=np.zeros(n)
    gap=np.diff(pt); 
    q=per/4; dev=np.abs(gap/q-np.round(gap/q))*q*1000
    for i in range(n):
        t=i/FPS; m=(pt[1:]>t-4)&(pt[1:]<t+4)
        ioi_irreg[i]=np.median(dev[m]) if m.sum()>3 else 99
    # 起音相对网格 1/4 拍的偏差中位数（ms，8s 窗）
    off=(pt-t0)/(per/4); offms=np.abs(off-np.round(off))*(per/4)*1000
    resid_t=np.zeros(n)
    for i in range(n):
        t=i/FPS; m=(pt>t-4)&(pt<t+4); resid_t[i]=np.median(offms[m]) if m.sum()>3 else 99
    # 局部节拍残差曲线（8s 窗内 loc 的中位绝对值）
    lres=np.zeros(n); bt=beats
    for i in range(n):
        t=i/FPS; m=(bt>t-4)&(bt<t+4); lres[i]=np.median(np.abs(loc[m]))*1000 if m.sum()>2 else 99
    e_rms=smooth(rms,0.5); e_flux=smooth(np.maximum(env,0),0.5)
    ai=0.5*norm01(smooth(rms,0.5))+0.3*norm01(smooth(np.maximum(env,0),0.5))+0.2*norm01(smooth(dens,0.5))
    return dict(T=T,x=x,mag=mag,env=env,env_all=env_all,rms=rms,beats=beats,per=per,t0=t0,loc=loc,glob=glob,conf=conf,
                dens=dens,pk=pt,resid_t=resid_t,lres=lres,ai=ai,res=res,offms=offms,ioi=ioi_irreg,contrast=contrast,gb=gb)

def segments(A):
    n=len(A['env']); t=np.arange(n)/FPS
    # 以 1 秒为单位判定
    N=int(A['T']); lab=[]
    conf=A['conf']; resid=A['resid_t']; ai=A['ai']; lres=A['lres']
    cthr=np.median(conf)
    for s in range(N):
        i=slice(int(s*FPS),int((s+1)*FPS)+1)
        c=conf[i].mean(); r=np.median(resid[i]); l=np.median(lres[i]); a=ai[i].mean()
        reg=(c>=cthr*0.6) and (l<=30)
        lab.append((c,r,l,a,reg))
    return lab

def bga():
    W,H=160,90
    p=subprocess.Popen(['ffmpeg','-v','error','-i',BGA,'-an','-vf',f'fps=30,scale={W}:{H}','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    fs=W*H*3; diffs=[];lum=[];dens=[];samp=[];prev=None;i=0
    rng=np.random.default_rng(1)
    while True:
        b=p.stdout.read(fs)
        if len(b)<fs: break
        f=np.frombuffer(b,np.uint8).reshape(H,W,3).astype(np.float32); g=f.mean(2)
        diffs.append(0 if prev is None else np.abs(g-prev).mean()); prev=g
        lum.append(g.mean()); dens.append((g>60).mean())
        if i%30==0:
            px=f.reshape(-1,3); px=px[px.max(1)>60]
            if len(px): samp.append(px[rng.choice(len(px),min(200,len(px)),replace=False)])
        i+=1
    return np.array(diffs),np.array(lum),np.array(dens),np.concatenate(samp)
def kmeans(X,k=5,it=40,seed=0):
    rng=np.random.default_rng(seed); C=X[rng.choice(len(X),k,replace=False)].copy()
    for _ in range(it):
        d=((X[:,None]-C[None])**2).sum(2); l=d.argmin(1)
        for j in range(k):
            if (l==j).any(): C[j]=X[l==j].mean(0)
    cnt=np.bincount(l,minlength=k); o=np.argsort(-cnt)
    return [('#%02x%02x%02x'%tuple(int(v) for v in C[j]),round(cnt[j]/len(X),3)) for j in o]

def offset():
    a=B.load_mono(MUS,SR); b=B.load_mono(BGA,SR)
    def e(x):
        m=stft(x); return z(flux(m,30,8000))
    ea,eb=e(a),e(b); n=1<<int(np.ceil(np.log2(len(ea)+len(eb))))
    c=np.fft.irfft(np.fft.rfft(ea,n)*np.conj(np.fft.rfft(eb,n)),n)
    mx=int(8*FPS); lags=np.r_[np.arange(0,mx),np.arange(-mx,0)]; cc=np.r_[c[:mx],c[n-mx:]]
    k=int(np.argmax(cc)); lag=lags[k]/FPS
    # 细化到采样级：时域包络互相关 ±50ms 附近
    corr=cc[k]/(np.linalg.norm(ea)*np.linalg.norm(eb)+1e-9)
    second=np.sort(cc)[-5]; 
    return lag,float(cc[k]/len(ea)),float(np.sort(cc)[-3]/len(ea)), float(np.corrcoef(a[:SR*60][:len(b[:SR*60])],b[:SR*60][:len(a[:SR*60])])[0,1])
if __name__=='__main__':
    import pickle
    A=audio(); lab=segments(A)
    print('T',A['T']); print(json.dumps(A['res'],default=float)[:600])
    pickle.dump({k:v for k,v in A.items() if k not in('x','mag')},open('work/audio_an.pkl','wb'))
    for s,(c,r,l,a,reg) in enumerate(lab):
        if s%4==0: print(s,round(c,2),round(r),round(l),round(a,2),int(reg),round(A['dens'][int(s*FPS)],1))
