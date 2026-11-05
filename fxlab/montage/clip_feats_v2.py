"""每片段特征（320 宽、15fps 代理）：运动量、色彩丰富度、帧差波动、亮度、主色；按 0.5s 窗也给出局部运动曲线。"""
import csv,subprocess,json,numpy as np
from concurrent.futures import ThreadPoolExecutor
R=list(csv.DictReader(open('work/clips.csv',encoding='utf-8')))
W,H=320,221
def feat(r):
    p=subprocess.run(['ffmpeg','-v','error','-i','input2/clips/'+r['file'],'-an','-vf',f'fps=15,scale={W}:{H}','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    F=np.frombuffer(p,np.uint8).reshape(-1,H,W,3).astype(np.float32)
    F=F[:,int(H*0.06):,:]            # 去掉顶部 HUD 条
    g=F.mean(3); d=np.abs(np.diff(g,axis=0)).mean((1,2))
    mx=F.max(3);mn=F.min(3);sat=((mx-mn)/(mx+1e-3)).mean((1,2))
    lum=g.mean((1,2))/255
    px=F[::3].reshape(-1,3); px=px[px.max(1)>40]
    hue=np.degrees(np.arctan2(np.sqrt(3)*(px[:,1]-px[:,2]),2*px[:,0]-px[:,1]-px[:,2]))%360
    hist=np.histogram(hue,12,(0,360))[0]/len(hue)
    n=len(d); k=max(1,int(7.5)); loc=[float(d[i:i+k].mean()) for i in range(0,n,k)]
    return dict(clip_id=int(r['clip_id']),dur=float(r['dur']),motion=float(d.mean()),motion_std=float(d.std()),sat=float(sat.mean()),lum=float(lum.mean()),lum_min=float(lum.min()),lum_max=float(lum.max()),hue_hist=np.round(hist,3).tolist(),motion_local=np.round(loc,2).tolist())
with ThreadPoolExecutor(4) as ex: out=list(ex.map(feat,R))
json.dump(out,open('work/clip_feats.json','w'))
m=np.array([o['motion'] for o in out]);s=np.array([o['sat'] for o in out]);l=np.array([o['lum'] for o in out])
print('motion pct',np.percentile(m,[10,50,90]).round(2),'sat',np.percentile(s,[10,50,90]).round(2),'lum',np.percentile(l,[10,50,90]).round(2))
print('brightest',sorted(out,key=lambda o:-o['lum'])[:6] and [(o['clip_id'],round(o['lum'],2)) for o in sorted(out,key=lambda o:-o['lum'])[:8]])
