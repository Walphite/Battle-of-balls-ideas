"""顶部裁切 7% 风险检查：每片段取 6 帧（1280 宽），在 6–7% 条带里找「白色小连通块」（玩家名文字特征）与球体边缘。"""
import csv,subprocess,numpy as np,json
from concurrent.futures import ThreadPoolExecutor
R=list(csv.DictReader(open('work/clips.csv',encoding='utf-8')))
W=1280;H=883
def label_small_white(band):
    # band: (h,w,3) uint8 -> 小白块数量（简易 4 连通标记）
    g=band.astype(int); white=(g.min(2)>200)
    h,w=white.shape; lab=np.zeros((h,w),int); n=0; sizes=[]
    for y in range(h):
        for x in range(w):
            if white[y,x] and lab[y,x]==0:
                n+=1; st=[(y,x)]; lab[y,x]=n; s=0
                while st:
                    cy,cx=st.pop(); s+=1
                    for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=cy+dy,cx+dx
                        if 0<=ny<h and 0<=nx<w and white[ny,nx] and lab[ny,nx]==0: lab[ny,nx]=n; st.append((ny,nx))
                sizes.append(s)
    sizes=np.array(sizes); return int(((sizes>=3)&(sizes<=40)).sum()), int((sizes>40).sum())
def run(r):
    p=subprocess.run(['ffmpeg','-v','error','-i','input2/clips/'+r['file'],'-an','-vf',f'fps=2,scale={W}:{H}','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    F=np.frombuffer(p,np.uint8).reshape(-1,H,W,3)
    y0,y1=int(H*0.06),int(H*0.07)+1
    out=[]
    for f in F[::1]:
        band=f[y0:y1]
        small,big=label_small_white(band)
        out.append((small,big))
    return dict(clip_id=int(r['clip_id']),max_small=max(o[0] for o in out),max_big=max(o[1] for o in out),frames=len(out))
with ThreadPoolExecutor(4) as ex: res=list(ex.map(run,R))
json.dump(res,open('work/crop_check.json','w'))
sus=[(r['clip_id'],r['max_small'],r['max_big']) for r in res if r['max_small']>=6]
print('clips with >=6 small white blobs in 6-7% band:',sus)
print('clips with big white regions (white ball edge) in band:',[(r['clip_id'],r['max_big']) for r in res if r['max_big']>0][:40])
