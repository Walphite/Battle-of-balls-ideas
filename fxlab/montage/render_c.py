"""版本 C（V5 合成层）：纯黑背景 + 向黑渐变 + 彩色外溢光晕 + BGA 字符穿透 + 整帧统一后期 + 锚点驱动的字符溶解。
不改 EDL。参数：fade_width / spill_width / spill_strength（960 宽度下的像素，按分辨率等比缩放）。
  python3 -I render_c.py --name E --schedule work/glyph_schedule_E.json --demo --out work/preview/sample_E_540.mp4"""
import os,sys,json,math,argparse,subprocess,numpy as np
from PIL import Image,ImageFilter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import render_v4 as R4
from effects.glyph_dissolve import GlyphDissolve,GlyphParams
FPS=R4.FPS; FR=R4.FR; T=R4.T; ASPECT=R4.ASPECT
FINE=np.load('work/onsets_fine.npz'); FT=FINE['t']           # 真实起音（校准后）
PARAMS=dict(fade_width=80.0,spill_width=120.0,spill_strength=0.38)
def smooth01(x): x=np.clip(x,0,1); return x*x*(3-2*x)
def blur_img(a,radius,scale=4):
    h,w=a.shape[:2]; sm=Image.fromarray(np.clip(a,0,255).astype(np.uint8)).resize((max(2,w//scale),max(2,h//scale)),Image.BILINEAR)
    sm=sm.filter(ImageFilter.GaussianBlur(max(0.5,radius/scale))); return np.asarray(sm.resize((w,h),Image.BILINEAR),np.float32)
def clean_bga(b):
    """BGA 背景：原始背景实测为纯黑 (0,0,0)。安全阀：若背景亮度>5%(13/255) 或通道差>8，则压黑去色。字符亮度键内的像素过暗时提到 ≥50% 亮度（色相不变）。"""
    low=np.percentile(b[::6,::6].reshape(-1,3),40,axis=0)
    if low.max()>13 or (low.max()-low.min())>8: b=np.maximum(b-low,0)
    L=b.mean(2)/255.0
    key=smooth01((L-0.10)/0.15)
    boost=np.minimum(np.maximum(1.0,0.5/np.maximum(L,1e-3)),255.0/np.maximum(b.max(2),1.0)/1.0)
    scale=1+key*(boost-1); out=b*scale[...,None]
    return np.minimum(out,255.0),key
class Composer:
    def __init__(self,W,H,schedule,params=None,ink=1.0):
        self.W,self.H=W,H; self.k=W/960.0; self.fx=R4.FX(W,H); self.P=dict(PARAMS,**(params or {})); self.ink=ink
        self.gd=GlyphDissolve(W,H,GlyphParams.spec_look()); self.sched=schedule; self.meta=[]; self.cc={}
        for e in self.sched: self._prep(e)
    def _prep(self,e):
        k=e['kind']; e['id']=e.get('id',f"{k}@{e.get('A0',e.get('t',0)):.3f}")
        if k=='out':
            e['S']=int(min(2*e['n_eighths'],math.floor((e['A1']-e['A0'])/0.0834))); e['t_s']=e['A0']-0.5; e['t_e']=e['A2']
        elif k=='in':
            e['S']=int(min(4,math.floor((e['A1']-e['A0'])/0.0834))); e['t_s']=e['B0']; e['t_e']=e['A1']+1e-6
        elif k=='scan': e['t_s']=e['A0']; e['t_e']=e['A2']
        elif k=='local': e['t_s']=e['t']; e['t_e']=e['t']+e['dur']
        elif k=='final':
            e['q']=e.get('q',0.16764); e['A1f']=e['A0']+6*e['q']; e['S']=int(min(12,math.floor((e['A1f']-e['A0'])/0.0834))); e['t_s']=e['A0']; e['t_e']=e['t_end']
    # ---- 事件状态（t = 帧中心时刻）
    def state(self,e,t):
        if not(e['t_s']<=t<e['t_e']): return None
        k=e['kind']
        if k=='out':
            if t<e['A0']-0.35: return dict(p=0.0,tear=True,refresh=-2)
            if t<e['A0']: return dict(p=0.0,tear=False,refresh=-1)
            S=e['S']; dt=(e['A1']-e['A0'])/S
            if t>=e['A1']: return dict(p=1.0,refresh=S)
            j=min(S,int((t-e['A0'])/dt+1e-9)); return dict(p=(j+1)/(S+1),refresh=j)
        if k=='in':
            if t<e['A0']: return dict(p=1.0,refresh=-1)
            S=e['S']; dt=(e['A1']-e['A0'])/S; j=min(S,int((t-e['A0'])/dt+1e-9)); return dict(p=1-(j+1)/(S+1),refresh=j)
        if k=='scan': return dict(front=(t-e['A0'])/(e['A2']-e['A0']),direction=e.get('direction','h'),refresh=int((t-e['A0'])/0.1))
        if k=='local': return dict(rects=e['rects'],refresh=0)
        if k=='final':
            if t<e['A1f']:
                S=e['S']; dt=(e['A1f']-e['A0'])/S; j=min(S,int((t-e['A0'])/dt+1e-9)); return dict(p=(j+1)/(S+1),refresh=j)
            prog=min(1.0,(t-e['A1f'])/2.1); return dict(p=1.0,alive=prog,refresh=100+int((t-e['A1f'])/0.1))
    # ---- 几何、渐变、光晕（按分辨率与余量缩放）
    def geometry(self,win):
        H,W,k=self.H,self.W,self.k; wh=int(round(H*min(win,1.0)))//2*2; ww=min(int(round(wh*ASPECT))//2*2,W); return wh,ww,(W-ww)//2,(H-wh)//2
    def fields(self,wh,ww,x,y):
        key=(wh,ww,x,y,round(self.P['fade_width']),round(self.P['spill_width']))
        if key in self.cc: return self.cc[key]
        H,W,k=self.H,self.W,self.k; fw=self.P['fade_width']*k; sw=self.P['spill_width']*k
        marg=dict(l=x,r=W-x-ww,t=y,b=H-y-wh); mref=0.125*H                          # 参照余量 = 75% 窗口时的单侧余量（约 67px@540）
        sc={e:float(np.clip(m/mref,0,1)) for e,m in marg.items()}                    # 渐变带/光晕随余量等比缩小，满屏趋于 0
        fe={e:fw*sc[e] for e in sc}; se={e:sw*sc[e] for e in sc}
        fe['l']=min(fe['l'],48*k)                                                    # 左侧按钮：缩短到下限 48px
        fe['t']=max(fe['t'],12*k)                                                    # 顶边：吃掉裁切后残留的盾点条（≤12px）
        jx=np.arange(ww,dtype=np.float32)+0.5; jy=np.arange(wh,dtype=np.float32)+0.5
        def ramp(d,f): return np.ones_like(d) if f<=0.5 else d/f
        mx=np.minimum(ramp(jx,fe['l']),ramp(ww-jx,fe['r']))                           # (ww,)
        # 右上小地图：右边缘在窗高 0–48% 区域缩短
        mr=np.where(jy<0.48*wh,np.minimum(ramp(ww-jx[None,:],min(fe['r'],16*k)),1)[0:1] if False else 1,1)
        Mx=np.minimum(ramp(jx,fe['l'])[None,:],np.where(jy[:,None]<0.48*wh,ramp(ww-jx,min(fe['r'],48*k))[None,:],ramp(ww-jx,fe['r'])[None,:]))
        My=np.minimum(ramp(jy,fe['t']),ramp(wh-jy,fe['b']))[:,None]
        lin=np.clip(np.minimum(Mx,My),0,1); M=np.zeros((H,W),np.float32); M[y:y+wh,x:x+ww]=0.5*lin+0.5*smooth01(lin)      # 线性与平滑曲线各半：斜率更均匀
        # 中心 60% 区域（墨水增益图用）
        cx=np.clip((0.5-np.abs(jx/ww-0.5)-0.2)/0.1,0,1); cy=np.clip((0.5-np.abs(jy/wh-0.5)-0.2)/0.1,0,1)
        Cn=np.zeros((H,W),np.float32); Cn[y:y+wh,x:x+ww]=np.outer(smooth01(cy),smooth01(cx))
        # 外溢光晕衰减：窗外按各边宽度线性衰减
        xs=np.arange(W,dtype=np.float32)[None,:]+0.5; ys=np.arange(H,dtype=np.float32)[:,None]+0.5
        dx=np.where(xs<x,x-xs,np.where(xs>x+ww,xs-(x+ww),0)); dxs=np.where(xs<x,max(se['l'],1e-3),max(se['r'],1e-3))
        dy=np.where(ys<y,y-ys,np.where(ys>y+wh,ys-(y+wh),0)); dys=np.where(ys<y,max(se['t'],1e-3),max(se['b'],1e-3))
        d=np.sqrt((dx/dxs)**2+(dy/dys)**2); inside=(xs>=x)&(xs<=x+ww)&(ys>=y)&(ys<=y+wh)
        fall=np.clip(1-d,0,1).astype(np.float32); fall[inside]=0
        if len(self.cc)>60: self.cc.clear()
        self.cc[key]=(M,Cn,fall); return self.cc[key]
    # ---- 单帧
    def frame(self,kf,t,shots,s,plan,cache,bga):
        W,H,k=self.W,self.H,self.k; fx=self.fx; tau=t-s['t']; base=s['base']; ev=s['events']
        win=R4.interp_win(shots,s,t); wh,ww,x,y=self.geometry(win)
        key=(s['n'],wh)
        if key not in cache: cache.clear(); cache[key]=R4.decode_frames(s,wh,plan[s['n']])
        frames,ww=cache[key]; x=(W-ww)//2
        img=frames[R4.src_index(s,tau)].astype(np.float32)
        # ---- 窗内：自然色彩（不调色、不降饱和）；形变类特效
        if 'contrast' in base: piv=img.mean(); img=piv+(img-piv)*base['contrast']
        if 'dim' in base: img=img*base['dim']
        z=1.0
        if 'beatpulse' in base:
            amp,every=base['beatpulse']; bb=R4.BEATS[s['b0']:s['b0']+s['beats']+1:every]; prev=bb[bb<=t+0.012]
            if len(prev): z=max(z,1+amp*math.exp(-max(t-prev[-1],0)/0.12))
        img=fx.zoom(img,z); rgb_now=False; accent_now=False
        for off,typ,nf in ev:
            if not(off<=tau<off+nf*FR): continue
            if typ=='rgb': rgb_now=True
            elif typ=='block': img=fx.block(img)
            elif typ=='tear': img=fx.tear(img)
            elif typ=='bga_accent': accent_now=True
        if 'shake' in base: img=fx.shake(img,base['shake'])
        # ---- 字符溶解（锚点驱动；字符色 = 米黄）
        covpx=None; act=[]; col=np.asarray(self.gd.p.glyph_color,np.float32)
        for e in self.sched:
            st=self.state(e,t)
            if st is None: continue
            img,m=self.gd.render_state(img,e['id'],st['refresh'],col,p=st.get('p'),front=st.get('front'),direction=st.get('direction','h'),rects=st.get('rects'),alive=st.get('alive'),tear_fn=fx.tear,tear=st.get('tear',False))
            cp=np.repeat(np.repeat(m,self.gd.ch,0),self.gd.cw,1); full=np.zeros((wh,ww),np.float32); full[:cp.shape[0],:cp.shape[1]]=cp[:wh,:ww]
            covpx=full if covpx is None else np.maximum(covpx,full)
            act.append(dict(id=e['id'],kind=e['kind'],p=round(float(st.get('p',st.get('front',1.0) if 'front' in st else 1.0)),3),refresh=st['refresh']))
        img=np.clip(img,0,255)
        # ---- BGA：原始背景（纯黑）+ 字符墨水
        b=bga.read().astype(np.float32); b,key_ink=clean_bga(b)
        j=np.searchsorted(FT,t); last=FT[j-1] if j>0 else -9; pulse=math.exp(-(t-last)/0.12) if t-last<0.4 else 0
        if accent_now: pulse=max(pulse,1.0)+0.5
        ink=np.minimum(b*(1+0.3*pulse),255.0)*self.ink
        M,Cn,fall=self.fields(wh,ww,x,y)
        # ---- 合成：窗口内容向黑渐变；外溢光晕（取自画面边缘 ~40px 平均色）
        full=np.zeros((H,W,3),np.float32); full[y:y+wh,x:x+ww]=img
        canvas=full*M[...,None]
        bw=int(40*k); inner=np.zeros((H,W),np.float32); inner[y+bw:y+wh-bw,x+bw:x+ww-bw]=1; win_m=np.zeros((H,W),np.float32); win_m[y:y+wh,x:x+ww]=1; band=win_m-inner
        num=blur_img(full*band[...,None],60*k); den=blur_img(band*255.0,60*k)/255.0
        colf=np.clip(num/np.maximum(den[...,None],0.03),0,255)
        s0=self.P['spill_strength']; inside_term=(1-M)*win_m                 # 窗内渐变带：向光晕色（而非纯黑）过渡，保证由内向外单调
        spill=colf*(s0*fall+s0*inside_term)[...,None]
        canvas=255-(255-canvas)*(255-np.minimum(spill,255))/255
        # ---- BGA 字符穿透：窗外 ~50%，窗内 ~25%，窗中心 ≤8%；溶解时覆盖格（多孔）露出字符
        Gm=(1-M)*0.50+M*(0.25-0.17*Cn)
        if covpx is not None:
            cp_full=np.zeros((H,W),np.float32); cp_full[y:y+wh,x:x+ww]=covpx; Gm=np.maximum(Gm,cp_full*M*0.75)
        canvas=255-(255-canvas)*(255-ink*Gm[...,None])/255
        # ---- 整帧统一后期
        if rgb_now: canvas=fx.rgb_split(canvas,max(1,int(W*0.006)))
        canvas=fx.scanlines(canvas,base.get('scan',0.1))
        if 'noise' in base: canvas=fx.add_noise(canvas,base['noise'])
        canvas=fx.vignette(canvas,base.get('vignette',0.25)*0.6)
        hi=np.clip((canvas.mean(2,keepdims=True)-0.68*255)/(0.32*255),0,1)*canvas
        canvas=255-(255-canvas)*(255-blur_img(hi,14*k)*0.25)/255
        if s['flash'] and tau<3*FR:
            a=0.6*math.exp(-tau/0.05); canvas=canvas*(1-a)+255*a
        if t<1.5: canvas*=t/1.5
        if t>T-2.68: canvas*=max(0,(T-t)/2.68)
        self.meta.append(dict(k=kf,t=round(t,4),shot=s['n'],x=x,y=y,ww=ww,wh=wh,dis=act))
        return np.clip(canvas,0,255)
def render_chunk(comp,c0,c1,fn,crf,maxrate,shots):
    W,H=comp.W,comp.H; n0=int(round(c0*FPS)); n1=int(round(c1*FPS))
    enc=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-pix_fmt','yuv420p','-g',str(FPS*2)]
    enc+=(['-b:v',maxrate,'-maxrate',maxrate,'-bufsize',maxrate] if maxrate else ['-crf',str(crf)])+['-f','mp4',fn]
    ep=subprocess.Popen(enc,stdin=subprocess.PIPE); bga=R4.BgaReader(c0,W,H,content_fit=True); plan={}
    for s in shots:
        if s['te']<=c0 or s['t']>=c1: continue
        tau,src=R4.src_map(s); s['_map']=(tau,src)
        plan[s['n']]=sorted({R4.src_index(s,(kk+0.5)*FR-s['t']) for kk in range(n0,n1) if s['t']<=(kk+0.5)*FR<s['te']})
    cache={}
    for kf in range(n0,n1):
        t=(kf+0.5)*FR; s=next(x for x in shots if x['t']<=t<x['te'])
        ep.stdin.write(comp.frame(kf,t,shots,s,plan,cache,bga).astype(np.uint8).tobytes())
    ep.stdin.close(); ep.wait(); bga.close()
def render(t0,t1,H,out,schedule,params=None,ink=1.0,crf=20,maxrate=None,chunk=10.0,snap=False):
    W=H*16//9; shots=R4.load_edl(); comp=Composer(W,H,schedule,params,ink)
    if snap: t0,t1=R4.snap_bounds(shots,t0,t1); print('snapped',t0,t1)
    d=out+'.chunks'; os.makedirs(d,exist_ok=True); chunks=[]; c0=t0
    while c0<t1-1e-6:
        c1=min(t1,c0+chunk); fn=os.path.join(d,f'c_{int(round(c0*1000)):07d}.mp4'); chunks.append(fn)
        if not os.path.exists(fn): render_chunk(comp,c0,c1,fn+'.part.mp4',crf,maxrate,shots); os.replace(fn+'.part.mp4',fn)
        c0=c1
    lst=os.path.join(d,'list.txt'); open(lst,'w').write(''.join(f"file '{os.path.abspath(f)}'\n" for f in chunks))
    af=f'atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS,'
    if t0<0.01: af+='afade=t=in:st=0:d=1.5,'
    if t1>T-0.1: af+=f'afade=t=out:st={T-2.68:.3f}:d=2.68,'
    af+='loudnorm=I=-10.5:TP=-1.0:LRA=11'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',lst,'-i',R4.MUS,'-af',af,'-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out],check=True)
    json.dump(dict(t0=t0,t1=t1,W=W,H=H,sched=comp.sched,frames=comp.meta),open(out+'.meta.json','w'),default=float)
    return out,t0,t1
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--t0',type=float,default=0); ap.add_argument('--t1',type=float,default=T); ap.add_argument('--height',type=int,default=540); ap.add_argument('--out',required=True)
    ap.add_argument('--schedule',default='none'); ap.add_argument('--demo',action='store_true'); ap.add_argument('--snap',action='store_true'); ap.add_argument('--crf',type=int,default=20); ap.add_argument('--maxrate',default=None)
    ap.add_argument('--ink',type=float,default=1.0); ap.add_argument('--fade_width',type=float,default=PARAMS['fade_width']); ap.add_argument('--spill_width',type=float,default=PARAMS['spill_width']); ap.add_argument('--spill_strength',type=float,default=PARAMS['spill_strength'])
    a=ap.parse_args(); sched=[]
    if a.schedule!='none':
        j=json.load(open(a.schedule)); sched=j['events'] if isinstance(j,dict) else j
        if a.demo: a.t0,a.t1=j['t0'],j['t1']
    print(render(a.t0,a.t1,a.height,a.out,sched,dict(fade_width=a.fade_width,spill_width=a.spill_width,spill_strength=a.spill_strength),a.ink,a.crf,a.maxrate,10.0,a.snap and not a.demo))
