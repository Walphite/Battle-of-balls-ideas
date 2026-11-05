"""逐片段决定顶部裁切方式：盾点星标条实测占原画面高度 7–9.1%（crop_probe）。
 - hard9：整段各帧在 6–9.1% 带内都没有球体/玩家名文字 → 硬裁 9%
 - grad6：任一帧该带内出现球体或白色文字 → 只裁 6%，改用窗口顶部 0–9% 高度的黑色渐变（透明度 100%→0）遮盖，不硬裁
判据（320 宽采样，每 0.25s 一帧，6–9.15% 带内）：x≥33% 区域 亮度>0.6 的像素占比>3% 或近白(饱和<0.25 且 V>0.8)占比>1.5%；x<33%（星标区，星标自带约 3% 近白轮廓）近白占比>6%。"""
import csv,subprocess,numpy as np,json
from concurrent.futures import ThreadPoolExecutor
R=list(csv.DictReader(open('work/clips.csv',encoding='utf-8'))); W=320
def judge(r):
    h=int(round(W*int(r['h'])/int(r['w'])/2))*2; d=float(r['dur']); hits=[]
    p=subprocess.run(['ffmpeg','-v','error','-i','input2/clips/'+r['file'],'-an','-vf',f'fps=4,scale={W}:{h}','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    F=np.frombuffer(p,np.uint8).reshape(-1,h,W,3).astype(np.float32)/255
    y0,y1=int(h*0.06),int(h*0.0915)
    for i,f in enumerate(F):
        band=f[y0:y1]; mx=band.max(2); mn=band.min(2); sat=(mx-mn)/(mx+1e-3)
        L=band@np.array([0.299,0.587,0.114],np.float32); white=(sat<0.25)&(mx>0.8); c=int(W*0.33)
        if (L[:,c:]>0.6).mean()>0.03 or white[:,c:].mean()>0.015 or white[:,:c].mean()>0.06: hits.append(round(i*0.25,2))
    return dict(clip_id=int(r['clip_id']),mode='grad6' if hits else 'hard9',hit_times=hits)
with ThreadPoolExecutor(4) as ex: res=list(ex.map(judge,R))
json.dump({str(r['clip_id']):r for r in res},open('work/crop_modes.json','w'))
n=sum(1 for r in res if r['mode']=='grad6'); print('hard9: %d 段；grad6（6%% + 黑色渐变）: %d 段'%(len(res)-n,n)); print('grad6 片段示例',[r['clip_id'] for r in res if r['mode']=='grad6'][:20])
