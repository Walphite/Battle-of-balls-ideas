"""吞噬时刻检测（每片段一次）：
 信号 a：中心球体面积突增（以屏幕中心色为种子的颜色连通域，120×83 网格，容差 45）——合球吞噬时主球变大；
 信号 b：中心区域帧差峰值（吞噬时球体/碎屑剧烈变化）；
 综合分 = 0.6·z(a) + 0.4·z(b)（平滑 3 帧）；t_hit = 综合分最大处；conf = 0.5·clip((z峰−3)/4,0,1) + 0.5·clip((z峰−z第二峰)/z峰·2.5,0,1)；t_hit2 = 与 t_hit 相距 ≥0.4s 的第二峰。
 输出 work/hits.json：{clip_id:{t_hit,conf,t_hit2,dur}}（t_hit 为片段内秒，60fps 分辨率）"""
import csv,subprocess,json,numpy as np,os,sys
from concurrent.futures import ThreadPoolExecutor
os.chdir(os.path.dirname(os.path.abspath(__file__)))
R=list(csv.DictReader(open('work/clips.csv',encoding='utf-8')))
GW,GH=120,83
def z(a): return (a-np.median(a))/(a.std()+1e-6)
def area_series(F):
    n,h,w,_=F.shape; cy,cx=int(h*0.47),int(w*0.5); out=np.zeros(n)
    yy,xx=np.mgrid[0:h,0:w]
    for i,f in enumerate(F):
        c0=np.median(f[cy-1:cy+2,cx-1:cx+2].reshape(-1,3),0); cand=(np.abs(f-c0).max(2)<45)
        m=np.zeros((h,w),bool); m[cy,cx]=True
        for _ in range(45):
            d=m.copy(); d[1:]|=m[:-1]; d[:-1]|=m[1:]; d[:,1:]|=m[:,:-1]; d[:,:-1]|=m[:,1:]; d&=cand
            if (d==m).all(): break
            m=d
        out[i]=m.sum()
    return out
def detect(r):
    cid=int(r['clip_id']); fps=60; dur=float(r['dur'])
    p=subprocess.run(['ffmpeg','-v','error','-i','input2/clips/'+r['file'],'-an','-vf',f'fps={fps},scale={GW}:{GH}:flags=area','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    F=np.frombuffer(p,np.uint8).reshape(-1,GH,GW,3).astype(np.float32); n=len(F)
    A=area_series(F); As=np.convolve(A,np.ones(3)/3,'same'); k=6                      # 0.1 s
    g=np.zeros(n); g[k:]=(As[k:]-As[:-k])/np.maximum(As[:-k],15); g=np.maximum(g,0)
    c=F[:,int(GH*0.2):int(GH*0.85),int(GW*0.2):int(GW*0.85)]; D=np.r_[0,np.abs(np.diff(c,axis=0)).mean((1,2,3))]
    sc=0.6*z(g)+0.4*z(D); sc=np.convolve(sc,np.ones(3)/3,'same'); lo,hi=int(0.15*fps),n-int(0.15*fps); s2=sc.copy(); s2[:lo]=-9; s2[hi:]=-9
    i1=int(np.argmax(s2)); s3=s2.copy(); s3[max(0,i1-int(0.4*fps)):i1+int(0.4*fps)]=-9; i2=int(np.argmax(s3))
    zp=(s2[i1]-np.median(sc))/(sc.std()+1e-6); zp2=(s3[i2]-np.median(sc))/(sc.std()+1e-6)
    conf=0.5*np.clip((zp-3)/4,0,1)+0.5*np.clip((zp-zp2)/max(zp,1e-6)*2.5,0,1)          # 峰值显著度 + 相对第二峰的突出度
    return str(cid),dict(t_hit=round(i1/fps,3),conf=round(float(conf),2),t_hit2=round(i2/fps,3),dur=dur,zpeak=round(float(zp),2),area_growth=round(float(g[i1]),2))
with ThreadPoolExecutor(4) as ex: res=dict(ex.map(detect,R))
json.dump(res,open('work/hits.json','w'))
c=np.array([v['conf'] for v in res.values()]); print('置信度分布：<0.3 %d，0.3–0.5 %d，0.5–0.7 %d，≥0.7 %d；中位 %.2f'%((c<0.3).sum(),((c>=0.3)&(c<0.5)).sum(),((c>=0.5)&(c<0.7)).sum(),(c>=0.7).sum(),np.median(c)))
