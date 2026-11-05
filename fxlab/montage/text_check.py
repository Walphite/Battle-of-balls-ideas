"""BGA 文字行颜色均匀性：原始 BGA 同一行文字像素从左到右分 4 段（亮度与饱和度的中位）；成片同位置同样统计。"""
import os,sys,json,subprocess,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
W,H=960,540
def bga_frame(t):
    p=subprocess.run(['ffmpeg','-v','error','-ss',f'{t+0.16:.3f}','-i','input2/bga/Credits.bga.mp4','-frames:v','1','-vf','scale=960:540','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    return np.frombuffer(p,np.uint8).reshape(H,W,3).astype(np.float32)
def prev_frame(fn,t):
    p=subprocess.run(['ffmpeg','-v','error','-ss',f'{t:.4f}','-i',fn,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    return np.frombuffer(p,np.uint8).reshape(H,W,3).astype(np.float32)
def find_line(b,kind,top=1.0):
    R,G,B=b[...,0],b[...,1],b[...,2]
    m=((R>140)&(G<110)&(B<110)) if kind=='red' else ((R>170)&(G>160)&(B<R-35)&(B>60))
    m=m.copy(); m[int(top*H):]=False
    rows=np.where(m.sum(1)>=6)[0]
    if not len(rows): return None
    # 最长的连续行带
    bands=[[rows[0],rows[0]]]
    for r in rows[1:]:
        if r-bands[-1][1]<=3: bands[-1][1]=r
        else: bands.append([r,r])
    r0,r1=max(bands,key=lambda x:m[x[0]:x[1]+1].sum()); cols=np.where(m[r0:r1+1].any(0))[0]; return r0,r1,cols.min(),cols.max(),m
def stats(img,r0,r1,c0,c1,mask):
    out=[]; edges=np.linspace(c0,c1+1,5).astype(int)
    for i in range(4):
        mm=mask[r0:r1+1,edges[i]:edges[i+1]]; px=img[r0:r1+1,edges[i]:edges[i+1]][mm]
        if len(px)<4: out.append((None,None,0)); continue
        mx=px.max(1); mn=px.min(1); L=px@np.array([0.299,0.587,0.114],np.float32); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
        out.append((float(np.median(L)),float(np.median(sat)),len(px)))
    return out
def spread(v): v=[x for x in v if x is not None]; return (max(v)-min(v))/max(max(v),1e-6)
if __name__=='__main__':
    fn=sys.argv[1] if len(sys.argv)>1 else 'work/preview/v7/line.mp4'; res=[]
    plan=json.loads(sys.argv[2]) if len(sys.argv)>2 else []
    for kind,t,t_in_video in plan:
        b=bga_frame(t); ln=find_line(b,kind)
        if ln is None: print(kind,t,'未找到'); continue
        r0,r1,c0,c1,m=ln; o=stats(b,r0,r1,c0,c1,m); f=prev_frame(fn,t_in_video); a=stats(f,r0,r1,c0,c1,m)
        res.append(dict(kind=kind,t=t,line=[int(r0),int(r1),int(c0),int(c1)],orig=o,out=a,orig_dL=spread([x[0] for x in o]),orig_dS=spread([x[1] for x in o]),out_dL=spread([x[0] for x in a]),out_dS=spread([x[1] for x in a])))
        print(kind,t,'行框',(r0,r1,c0,c1),'\n  原始 4 段(亮度,饱和度)',[(round(x[0]),round(x[1],2)) for x in o],'亮度差 %.0f%% 饱和度差 %.0f%%'%(res[-1]['orig_dL']*100,res[-1]['orig_dS']*100),'\n  成片 4 段',[(round(x[0]),round(x[1],2)) for x in a],'亮度差 %.0f%% 饱和度差 %.0f%%'%(res[-1]['out_dL']*100,res[-1]['out_dS']*100))
    json.dump(res,open(sys.argv[3] if len(sys.argv)>3 else 'work/text_check.json','w'))
