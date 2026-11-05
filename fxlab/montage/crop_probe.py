"""盾点星标条行范围：对每个片段取 3 帧（640 宽），统计每行（原画面 0–14%）左 45% 宽度内「高饱和+高亮度」像素占比。"""
import csv,subprocess,numpy as np,json
from concurrent.futures import ThreadPoolExecutor
R=list(csv.DictReader(open('work/clips.csv',encoding='utf-8')))
W=640
def prof(r):
    h=int(round(W*int(r['h'])/int(r['w'])/2))*2; out=[]
    for ss in ('0.4','1.2','2.0'):
        p=subprocess.run(['ffmpeg','-v','error','-ss',ss,'-i','input2/clips/'+r['file'],'-frames:v','1','-vf',f'scale={W}:{h}','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
        if len(p)<W*h*3: continue
        f=np.frombuffer(p,np.uint8).reshape(h,W,3).astype(np.float32)/255; mx=f.max(2); mn=f.min(2); sat=(mx-mn)/(mx+1e-3)
        m=((sat>0.45)&(mx>0.6))[:int(h*0.14),:int(W*0.45)]; out.append(m.mean(1))
    n=int(h*0.14); a=np.mean(out,0); return np.interp(np.linspace(0,n-1,140),np.arange(n),a)    # 重采样为 0–14% 共 140 点（0.1%/点）
with ThreadPoolExecutor(4) as ex: P=np.array(list(ex.map(prof,R)))
m=P.mean(0); print('每 0.5% 行的平均占比：',[(round(i*0.1,1),round(float(m[i:i+5].mean())*100,2)) for i in range(0,140,5)])
np.save('work/crop_profile.npy',P)
# 每个片段：占比超过自身基线 3× 的最后一行
last=[]
for p in P:
    base=np.median(p[110:140])+1e-4; idx=np.where(p>max(0.02,3*base))[0]; idx=idx[idx<100]; last.append(idx.max()*0.1 if len(idx) else 0)
print('星标条下沿（%原画面高度）分位 10/50/90/最大：',np.percentile(last,[10,50,90,100]).round(1))
