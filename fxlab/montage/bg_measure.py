"""第 1 步：BGA 背景测量。非字符像素 = 亮度低于该帧(或区域)第 60 分位的像素；取其中位 RGB。"""
import subprocess,json,sys,os,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
lum=lambda f:f@np.array([0.299,0.587,0.114],np.float32)
def med_rgb(f,mask=None):
    L=lum(f); 
    if mask is None: mask=np.ones(L.shape,bool)
    thr=np.percentile(L[mask],60); m=mask&(L<=thr); px=f[m]; return np.median(px,0),px.shape[0]
def bga_frames(n=20):
    out=[]
    for i in range(n):
        t=(i+0.5)*277.8/n
        p=subprocess.run(['ffmpeg','-v','error','-ss',f'{t:.2f}','-i','input2/bga/Credits.bga.mp4','-frames:v','1','-vf','scale=640:360','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
        out.append((t,np.frombuffer(p,np.uint8).reshape(360,640,3).astype(np.float32)))
    return out
if __name__=='__main__':
    res=[]; 
    for t,f in bga_frames():
        # 内容区（BGA 自身左右各 ~8% 为纯黑边，另报全幅）
        full,_=med_rgb(f); c=f[:,int(640*0.08):int(640*0.92)]; core,_=med_rgb(c); res.append((t,full,core))
    full=np.median([r[1] for r in res],0); core=np.median([r[2] for r in res],0)
    print('BGA 原始 非字符像素中位 RGB：全幅',full.round(1),'内容区(8–92%列)',core.round(1))
    print('  逐帧内容区中位（前 5）',[tuple(r[2].round().astype(int)) for r in res[:5]])
    print('  内容区通道差(max-min)',round(float(core.max()-core.min()),1),'亮度',round(float(core@np.array([0.299,0.587,0.114])/255*100),2),'%')
    json.dump(dict(bga_full=full.tolist(),bga_core=core.tolist()),open('work/bg_measure_old.json','w'))
    # 当前成片（sample_C1）窗外 BGA 背景区
    meta=json.load(open('work/preview/sample_C1_540.mp4.meta.json'))['frames']; meds=[]
    for ts in (0.5,2.5,4.5,6.5,8.5,10.5,12.0):
        p=subprocess.run(['ffmpeg','-v','error','-ss',str(ts),'-i','work/preview/sample_C1_540.mp4','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
        f=np.frombuffer(p,np.uint8).reshape(540,960,3).astype(np.float32); m=min(meta,key=lambda e:abs(e['t']-(139.039+ts)))
        mask=np.ones((540,960),bool); mg=int(m['fl']+12); mask[max(0,m['y']-mg):m['y']+m['wh']+mg,max(0,m['x']-mg):m['x']+m['ww']+mg]=False
        if mask.sum()>400: meds.append(med_rgb(f,mask)[0])
    cur=np.median(meds,0); print('当前成片 sample_C1 窗外背景中位 RGB',cur.round(1),'亮度',round(float(cur@np.array([0.299,0.587,0.114])/255*100),2),'%','通道差',round(float(cur.max()-cur.min()),1),'(n frames=%d)'%len(meds))
    json.dump(dict(current=cur.tolist()),open('work/bg_measure_cur.json','w'))
