"""版本 C（融合版·新合成层）：边缘羽化 + 光溢 + BGA 字符穿透 + 整帧统一后期 + 字符溶解预设族。
不改 EDL。用法：
  python3 -I render_c.py --name C1 --t0 140 --t1 152 --snap --schedule none --out work/preview/sample_C1_540.mp4
  python3 -I render_c.py --name C2 ... --edge                       # C1 + 变体 e（边缘带）
  python3 -I render_c.py --name D --schedule work/dissolve_demo.json --demo --out ...
  python3 -I render_c.py --name full --schedule work/dissolve_schedule.json ...   # 全片（后续步骤）"""
import os,sys,json,math,argparse,subprocess,numpy as np
from PIL import Image,ImageFilter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import render_v4 as R4
from effects.glyph_dissolve import GlyphDissolve,GlyphParams
FPS=R4.FPS; FR=R4.FR; T=R4.T; ASPECT=R4.ASPECT; BGA_MAIN=R4.BGA_MAIN
def smooth01(x): x=np.clip(x,0,1); return x*x*(3-2*x)
def blur_img(a,radius,scale=4):
    """大半径模糊：缩小 scale 倍后高斯再放大（a: float32 HxWx3 或 HxW）"""
    h,w=a.shape[:2]; sm=Image.fromarray(np.clip(a,0,255).astype(np.uint8)).resize((max(2,w//scale),max(2,h//scale)),Image.BILINEAR)
    sm=sm.filter(ImageFilter.GaussianBlur(radius/scale)); return np.asarray(sm.resize((w,h),Image.BILINEAR),np.float32)
class Composer:
    def __init__(self,W,H,schedule,edge=False,cap=1.0):
        self.W,self.H=W,H; self.k=W/960.0; self.fx=R4.FX(W,H); self.gd=GlyphDissolve(W,H); self.sched=schedule; self.edge=edge; self.cap=cap
        self.acache={}; self.gcache={}; self.meta=[]
    # ---- 几何与羽化
    def geometry(self,win,s):
        H,W=self.H,self.W; wh=int(round(H*min(win,self.cap)))//2*2; ww=int(round(wh*ASPECT))//2*2; ww=min(ww,W)
        x=(W-ww)//2; y=(H-wh)//2
        sw=min(1,max(0,(win-0.75)/0.25)); f=(24+16*sw)*self.k           # 羽化 24–40px@960，随窗口放大
        fl=fr=f; ft=max(f,16*self.k); fb=f if y+wh<H else 0           # 顶边始终羽化以吃掉剩余盾点条；贴底则不羽化
        if x==0: fl=fr=0
        return wh,ww,x,y,fl,fr,ft,fb
    def alpha(self,wh,ww,x,y,fl,fr,ft,fb):
        key=(wh,ww,x,y,round(fl),round(ft),round(fb))
        if key in self.acache: return self.acache[key]
        jx=np.arange(ww,dtype=np.float32); jy=np.arange(wh,dtype=np.float32)
        ax=np.ones(ww,np.float32)
        if fl>0: ax=np.minimum(ax,(jx+0.5)/fl)
        if fr>0: ax=np.minimum(ax,(ww-jx-0.5)/fr)
        ay=np.ones(wh,np.float32)
        if ft>0: ay=np.minimum(ay,(jy+0.5)/ft)
        if fb>0: ay=np.minimum(ay,(wh-jy-0.5)/fb)
        A=np.zeros((self.H,self.W),np.float32); A[y:y+wh,x:x+ww]=np.outer(smooth01(ay),smooth01(ax))
        # 窗内 BGA 墨水增益图：窗外 0.5；窗内 0.25；窗中心 ~60% 区域 ≤0.08
        cx=smooth01((0.5-np.abs((jx+0.5)/ww-0.5))/0.2-0.5+0.0); cy=smooth01((0.5-np.abs((jy+0.5)/wh-0.5))/0.2-0.5+0.0)
        cx=np.clip((0.5-np.abs((jx+0.5)/ww-0.5)-0.2)/0.1,0,1); cy=np.clip((0.5-np.abs((jy+0.5)/wh-0.5)-0.2)/0.1,0,1)   # 中心 60% 内为 1，边缘 10% 过渡
        C=np.zeros((self.H,self.W),np.float32); C[y:y+wh,x:x+ww]=np.outer(smooth01(cy),smooth01(cx))
        G=(1-A)*0.50+A*(0.25-0.17*C)
        if len(self.acache)>40: self.acache.clear()
        self.acache[key]=(A,G); return A,G
    # ---- 单帧
    def frame(self,k,t,shots,s,plan,cache,bga,style_ctx):
        W,H=self.W,self.H; fx=self.fx; tau=t-s['t']; base=s['base']; ev=s['events']; acc=R4.accent_at(t)
        win=R4.interp_win(shots,s,t); wh,ww,x,y,fl,fr,ft,fb=self.geometry(win,s)
        key=(s['n'],wh)
        if key not in cache: cache.clear(); cache[key]=R4.decode_frames(s,wh,plan[s['n']])
        frames,ww2=cache[key]; ww=ww2; x=(W-ww)//2
        fi=R4.src_index(s,tau); img=frames[fi].astype(np.float32)
        # ---- 窗内：调色（饱和度 −20%，阴影染向 BGA 蓝灰）与窗内形变类特效
        if 'contrast' in base: piv=img.mean(); img=piv+(img-piv)*base['contrast']
        if 'dim' in base: img=img*base['dim']
        img=fx.desat(img,0.20+(base.get('desat',0.0))); img=fx.tone(img,0.30 if base.get('tone') else 0.22)
        z=1.0
        if 'beatpulse' in base:
            amp,every=base['beatpulse']; bb=R4.BEATS[s['b0']:s['b0']+s['beats']+1:every]; prev=bb[bb<=t+0.012]
            if len(prev): z=max(z,1+amp*math.exp(-max(t-prev[-1],0)/0.12))
        img=fx.zoom(img,z)
        rgb_now=False; accent_now=False
        for off,typ,nf in ev:
            if not(off<=tau<off+nf*FR): continue
            if typ=='rgb': rgb_now=True
            elif typ=='block': img=fx.block(img)
            elif typ=='tear': img=fx.tear(img)
            elif typ=='bga_accent': accent_now=True
        if 'shake' in base: img=fx.shake(img,base['shake'])
        # ---- 字符溶解（窗口图像上；格子/字符集/颜色统一）
        dis_active=[]
        for ei,e in enumerate(self.sched):
            if e['kind']=='ring' or not(e['t']<=t<e['t']+e['dur']): continue
            et=t-e['t']; kk=int(round(e['t']*1000))
            img=self.gd.apply(img,e['kind'],et,acc,key=kk,scale=e.get('scale',1.0),tear_fn=fx.tear,rects=e.get('rects'),direction=e.get('direction','h'))
            dis_active.append(e['kind'])
        img=np.clip(img,0,255); wl=float(img[int(wh*0.2):int(wh*0.8),int(ww*0.2):int(ww*0.8)].mean()/255)
        # ---- 背景（BGA）：蓝灰基底 + 亮度键墨水
        b=bga.read().astype(np.float32); g=255*0.9/max(30,np.percentile(b[::4,::4],99.5)); b=np.clip(b*g,0,255)
        j=np.searchsorted(R4.PK,t); last=R4.PK[j-1] if j>0 else -9; pulse=math.exp(-(t-last)/0.12) if t-last<0.4 else 0
        if accent_now: pulse=max(pulse,1.0)+0.5
        b=b*(1+0.4*pulse)
        lum=b.mean(2)/255; kk_=smooth01((lum-0.12)/0.28); ink=b*kk_[...,None]
        A,Gm=self.alpha(wh,ww,x,y,fl,fr,ft,fb)
        # 窗内外亮度比 ≤1.6：窗口过亮时压窗口
        tgt=float(np.clip(wl/1.3,0.345,0.42))      # 后期（扫描线/暗角）整体约 ×0.73：窗外成片亮度 ≈25–30%，窗内/窗外 ≤1.5
        if wl>1.38*tgt: img=img*(1.38*tgt/wl); wl=1.38*tgt
        out_w=(1-A); ink_out=float((ink.mean(2)*0.5*out_w).sum()/(out_w.sum()+1e-6))
        kb=float(np.clip((tgt*255-ink_out)/BGA_MAIN.mean(),0.5,2.0))
        bg=BGA_MAIN*kb+b*0.25
        # ---- 合成（羽化 alpha）
        full=np.zeros((H,W,3),np.float32); full[y:y+wh,x:x+ww]=img
        canvas=bg*(1-A[...,None])+full*A[...,None]
        # ---- 光溢：窗口边缘 ~40px 内的平均色，大半径模糊后 screen 到窗外围
        band=np.zeros((H,W),np.float32); bw=int(40*self.k); inner=np.zeros((H,W),np.float32); inner[y+bw:y+wh-bw,x+bw:x+ww-bw]=1
        win_m=np.zeros((H,W),np.float32); win_m[y:y+wh,x:x+ww]=1; band=win_m-inner
        num=blur_img(full*band[...,None],80*self.k); den=blur_img(band*255,80*self.k)/255.0
        col=num/(den[...,None]*255*0+np.maximum(den[...,None],0.02)); col=col  # 局部平均色（0–255）
        strength=np.clip(den/0.35,0,1)[...,None]*(1-A[...,None])
        glow=np.clip(col,0,255)*strength*0.35
        canvas=255-(255-canvas)*(255-glow)/255
        # ---- 变体 e：边缘带字符
        if self.edge:
            rect=(x,y,ww,wh); canvas=self.gd.ring(canvas,rect,band=25*self.k,color=acc,key=1,tau=t)
            dis_active.append('ring')
        # ---- BGA 字符穿透（整图最上层 screen；窗外 ~50%，窗内 ~25%，窗中心 ≤8%）
        gm=Gm.copy()
        if dis_active and self.gd.p.bga_reveal>0: gm=gm*(1+self.gd.p.bga_reveal*A)
        inkl=ink*gm[...,None]; canvas=255-(255-canvas)*(255-inkl)/255
        # ---- 整帧统一后期：RGB 色差 → 扫描线 → 噪点 → 暗角 → 泛光 → 闪白
        if rgb_now: canvas=fx.rgb_split(canvas,max(1,int(W*0.006)))
        canvas=fx.scanlines(canvas,base.get('scan',0.1))
        if 'noise' in base: canvas=fx.add_noise(canvas,base['noise'])
        canvas=fx.vignette(canvas,base.get('vignette',0.25)*0.6)
        hi=np.clip((canvas.mean(2,keepdims=True)-0.68*255)/(0.32*255),0,1)*canvas
        canvas=255-(255-canvas)*(255-blur_img(hi,14*self.k)*0.25)/255
        if s['flash'] and tau<3*FR and style_ctx.get('flash',True):
            a=0.6*math.exp(-tau/0.05); canvas=canvas*(1-a)+255*a
        if t<1.5: canvas*=t/1.5
        if t>T-2.68: canvas*=max(0,(T-t)/2.68)
        self.meta.append(dict(k=k,t=round(t,4),x=x,y=y,ww=ww,wh=wh,fl=round(fl,1),ft=round(ft,1),fb=round(fb,1),dis=dis_active,wl=round(wl,3)))
        return np.clip(canvas,0,255)
def render_chunk(comp,c0,c1,fn,crf,maxrate,shots,style_ctx):
    W,H=comp.W,comp.H; n0=int(round(c0*FPS)); n1=int(round(c1*FPS))
    enc=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-pix_fmt','yuv420p','-g',str(FPS*2)]
    enc+=(['-b:v',maxrate,'-maxrate',maxrate,'-bufsize',maxrate] if maxrate else ['-crf',str(crf)])+['-f','mp4',fn]
    ep=subprocess.Popen(enc,stdin=subprocess.PIPE); bga=R4.BgaReader(c0,W,H,content_fit=True)
    plan={}
    for s in shots:
        if s['te']<=c0 or s['t']>=c1: continue
        tau,src=R4.src_map(s); s['_map']=(tau,src)
        plan[s['n']]=sorted({R4.src_index(s,(kk+0.5)*FR-s['t']) for kk in range(n0,n1) if s['t']<=(kk+0.5)*FR<s['te']})
    cache={}
    for kf in range(n0,n1):
        t=(kf+0.5)*FR; s=next(x for x in shots if x['t']<=t<x['te'])
        ep.stdin.write(comp.frame(kf,t,shots,s,plan,cache,bga,style_ctx).astype(np.uint8).tobytes())
    ep.stdin.close(); ep.wait(); bga.close()
def render(name,t0,t1,H,out,schedule,edge,crf=20,maxrate=None,chunk=10.0,snap=False,flash=True,cap=1.0):
    W=H*16//9; shots=R4.load_edl(); comp=Composer(W,H,schedule,edge,cap)
    if snap: t0,t1=R4.snap_bounds(shots,t0,t1); print('snapped',t0,t1)
    d=out+'.chunks'; os.makedirs(d,exist_ok=True); chunks=[]; c0=t0
    while c0<t1-1e-6:
        c1=min(t1,c0+chunk); fn=os.path.join(d,f'c_{int(round(c0*1000)):07d}.mp4'); chunks.append(fn)
        if not os.path.exists(fn): render_chunk(comp,c0,c1,fn+'.part.mp4',crf,maxrate,shots,dict(flash=flash)); os.replace(fn+'.part.mp4',fn)
        c0=c1
    lst=os.path.join(d,'list.txt'); open(lst,'w').write(''.join(f"file '{os.path.abspath(f)}'\n" for f in chunks))
    af=f'atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS,'
    if t0<0.01: af+='afade=t=in:st=0:d=1.5,'
    if t1>T-0.1: af+=f'afade=t=out:st={T-2.68:.3f}:d=2.68,'
    af+='loudnorm=I=-10.5:TP=-1.0:LRA=11'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',lst,'-i',R4.MUS,'-af',af,'-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out],check=True)
    json.dump(dict(t0=t0,t1=t1,W=W,H=H,frames=comp.meta),open(out+'.meta.json','w'))
    return out,t0,t1
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--name',default='C'); ap.add_argument('--t0',type=float,default=0); ap.add_argument('--t1',type=float,default=T)
    ap.add_argument('--height',type=int,default=540); ap.add_argument('--out',required=True); ap.add_argument('--schedule',default='none'); ap.add_argument('--edge',action='store_true')
    ap.add_argument('--snap',action='store_true'); ap.add_argument('--demo',action='store_true'); ap.add_argument('--crf',type=int,default=20); ap.add_argument('--maxrate',default=None); ap.add_argument('--noflash',action='store_true')
    a=ap.parse_args(); sched=[]
    if a.schedule!='none':
        j=json.load(open(a.schedule)); sched=j['events'] if isinstance(j,dict) else j
        if a.demo: a.t0,a.t1=j['t0'],j['t1']
    print(render(a.name,a.t0,a.t1,a.height,a.out,sched,a.edge,a.crf,a.maxrate,10.0,a.snap and not a.demo,not a.noflash))
