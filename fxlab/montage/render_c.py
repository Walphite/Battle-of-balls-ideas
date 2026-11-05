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
PROBE=os.environ.get('PROBE')=='1'                                    # 1 = 只计算逐帧统计，不编码
LEGACY=os.environ.get('LEGACY')=='1'                                   # 1 = 左上角清理与整帧变暗修复全部关闭（用于「修改前」对照渲染）
CROP_MODES=json.load(open('work/crop_modes.json')) if os.path.exists('work/crop_modes.json') else {}
BLUEGRAY=np.array([110.0,125.0,150.0],np.float32)                       # BGA 字符蓝灰（图标终端化的目标色相）
def eff_tau(s,tau):
    """内容时间 τ'：定格重复（stutter）= 暂停，而不是回放后跳帧——每个已结束的 stutter 让内容时间落后其时长；stutter 进行中内容时间停在其起点。
    （旧实现在释放时内容向前跳 N 帧，素材自身快速变暗/变亮时就变成整帧亮度突变。）"""
    if LEGACY: 
        for off,typ,nf in s['events']:
            if typ=='stutter' and off<=tau<off+nf*FR: return off
        return tau
    lag=0.0
    for off,typ,nf in sorted(e for e in s['events'] if e[1]=='stutter'):
        d=nf*FR
        if tau>=off+d: lag+=d
        elif tau>=off: return off-lag
    return tau-lag
def src_index_c(s,tau):
    tg,src=s['_map']; st=float(np.interp(eff_tau(s,tau),tg,src)); return max(0,int(st*float(R4.CLIPS[s['clip']]['fps'])+1e-6))
def crop_for(s):
    """该镜头的顶部裁切比例与是否用黑色渐变：hard9（整段没有球体/文字进入 6–9% 带）→ 硬裁 9%；否则 6% + 窗口顶部 0–9% 黑色渐变"""
    if LEGACY: return 0.06,False
    m=CROP_MODES.get(str(s['clip']),{}).get('mode','grad6'); return (0.09,False) if m=='hard9' else (0.06,True)
def decode_frames2(s,wh,need_idx,crop):
    r=R4.CLIPS[s['clip']]; ct=int(R4.SRC_H*crop)//2*2; ch=R4.SRC_H-ct; ww=int(round(wh*R4.SRC_W/ch))//2*2
    vf=f'crop=iw:{ch}:0:{ct},scale={ww}:{wh}:flags=bicubic'; need=s['src_out']-s['src_in']+0.1
    p=subprocess.Popen(['ffmpeg','-v','error','-ss',f"{s['src_in']:.3f}",'-t',f'{need:.3f}','-i','input2/clips/'+r['file'],'-an','-vf',vf,'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    fs=ww*wh*3; out={}; i=0; mx=max(need_idx) if need_idx else 0; want=set(need_idx); last=None
    while True:
        b=p.stdout.read(fs)
        if len(b)<fs: break
        if i in want: out[i]=np.frombuffer(b,np.uint8).reshape(wh,ww,3)
        last=b; i+=1
        if i>mx: break
    try: p.stdout.close(); p.kill()
    except Exception: pass
    if last is not None:
        lf=np.frombuffer(last,np.uint8).reshape(wh,ww,3)
        for k in want:
            if k not in out: out[k]=lf
    return out,ww
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
        gp=GlyphParams.spec_look()
        if os.environ.get('GD_BGDIM') is not None: gp.bg_dim=float(os.environ['GD_BGDIM'])          # 诊断开关
        self.gd=GlyphDissolve(W,H,gp); self.meta=[]; self.cc={}
        off=set(filter(None,os.environ.get('GD_OFF','').split(',')))                                   # 诊断开关：关闭某类事件
        self.sched=[e for e in schedule if e['kind'] not in off]
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
    def geometry(self,win,crop=0.06):
        H,W,k=self.H,self.W,self.k; wh=int(round(H*min(win,1.0)))//2*2; asp=R4.SRC_W/(R4.SRC_H-int(R4.SRC_H*crop)//2*2); ww=min(int(round(wh*asp))//2*2,W); return wh,ww,(W-ww)//2,(H-wh)//2
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
    # ---- 左上角清理
    def terminalize_icons(self,img,crop,wh,ww):
        """左侧按钮柱里的喇叭/麦克风/白色圆钮/下拉箭头：去饱和 70%、亮度 ×0.6、向 BGA 蓝灰靠拢；闪电键、摇杆、小地图不动"""
        key=('ico',wh,ww,crop)
        if key not in self.cc:
            v=(np.arange(wh,dtype=np.float32)+0.5)/wh; u=(np.arange(ww,dtype=np.float32)+0.5)/ww; m=np.zeros((wh,ww),np.float32)
            for yc in (0.020,0.098,0.205,0.300):
                yy=yc if crop>=0.09 else (0.03+0.91*yc)/0.94                           # 6% 裁切时图标位置下移
                rx=0.034; ry=rx*ww/wh; d=np.sqrt(((u[None,:]-0.075)/rx)**2+((v[:,None]-yy)/ry)**2); m=np.maximum(m,1-smooth01((d-0.85)/0.40))
            self.cc[key]=m
        m=self.cc[key][...,None]
        L=img.mean(2,keepdims=True); c=(L+0.30*(img-L))*0.6; c=c*0.65+0.35*(L*0.6)*(BLUEGRAY/BLUEGRAY.mean())
        return img*(1-m)+c*m
    def finish_game(self,img,crop,grad):
        """游戏层：左上角黑色渐变（角落 60%，到窗宽 35%、窗高 32% 为 0）+ 顶部黑色渐变（grad6 镜头：窗高 0–9% 内 100%→0；hard9 只在顶部 1.5% 内收口）"""
        wh,ww=img.shape[:2]; key=('cor',wh,ww,grad)
        if key not in self.cc:
            u=(np.arange(ww,dtype=np.float32)+0.5)/ww; v=(np.arange(wh,dtype=np.float32)+0.5)/wh
            r=np.sqrt((u[None,:]/0.35)**2+(v[:,None]/0.32)**2); a=0.6*(1-smooth01(r))
            top=1-np.clip(v/(0.09 if grad else 0.015),0,1); a=1-(1-a)*(1-top[:,None]); self.cc[key]=a.astype(np.float32)
        return img*(1-self.cc[key])[...,None]
    def title_region(self):
        key=('title',self.W,self.H)
        if key not in self.cc:
            xx=np.arange(self.W,dtype=np.float32)[None,:]/(0.40*self.W); yy=np.arange(self.H,dtype=np.float32)[:,None]/(0.30*self.H)
            self.cc[key]=((1-smooth01((xx-0.85)/0.15))*(1-smooth01((yy-0.85)/0.15))).astype(np.float32)
        return self.cc[key]
    def text_substrate(self,key_ink):
        """左上区域内：由 BGA 亮度键提取每行文字的外接框，外扩 12px，高斯软边 ~8px → 衬底遮罩 (H,W)∈[0,1]"""
        H,W,k=self.H,self.W,self.k; rh,rw=int(0.30*H),int(0.40*W); kr=key_ink[:rh,:rw]>0.5
        hit=np.where(kr.sum(1)>=max(2,int(2*k)))[0]; sm=np.zeros((H//4,W//4),np.float32)
        if len(hit):
            groups=[[hit[0],hit[0]]]
            for r in hit[1:]:
                if r-groups[-1][1]<=int(4*k)+1: groups[-1][1]=r
                else: groups.append([r,r])
            pad=int(round(12*k))
            for r0,r1 in groups:
                cols=np.where(kr[r0:r1+1].any(0))[0]; c0,c1=cols.min(),cols.max()
                sm[max(0,(r0-pad))//4:(r1+pad)//4+1,max(0,(c0-pad))//4:(c1+pad)//4+1]=1
        if sm.max()<=0: return np.zeros((H,W),np.float32)
        im=Image.fromarray((sm*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(0.5,8*k/4))).resize((W,H),Image.BILINEAR)
        return np.asarray(im,np.float32)/255.0
    def text_layer(self,key_ink):
        """BGA 文字层（V7）：按「文字行段」设置强度，整段同一强度、同一行内不随 x 变化。
        行段 = 行带（行间隙≤4px）内按列间隙>40px 切开的连续文字；起点 x<40% 宽且中心 y<30% 高的行段（左上标题区）强度 0.95，其余 0.55。
        返回 (alpha (H,W)，暗晕 (H,W)∈[0,1])：暗晕 = 字形掩膜膨胀 ~2.5px、高斯 ~5px，只在字形周围。"""
        H,W,k=self.H,self.W,self.k; K=key_ink>0.5; rows=np.where(K.sum(1)>=2)[0]; A=np.zeros((H,W),np.float32); segs=[]
        if len(rows):
            bands=[[rows[0],rows[0]]]
            for r in rows[1:]:
                if r-bands[-1][1]<=int(4*k)+1: bands[-1][1]=r
                else: bands.append([r,r])
            for r0,r1 in bands:
                cols=np.where(K[r0:r1+1].any(0))[0]
                if not len(cols): continue
                runs=[[cols[0],cols[0]]]
                for c in cols[1:]:
                    if c-runs[-1][1]<=int(40*k): runs[-1][1]=c
                    else: runs.append([c,c])
                for c0,c1 in runs:
                    title=(c0<0.40*W) and ((r0+r1)/2<0.30*H); st=0.95 if title else 0.85
                    A[r0:r1+1,c0:c1+1]=np.maximum(A[r0:r1+1,c0:c1+1],key_ink[r0:r1+1,c0:c1+1]*st); segs.append((r0,r1,c0,c1,st))
        if not segs: return A,np.zeros((H,W),np.float32),np.zeros((H,W),np.float32)
        im=Image.fromarray((K*255).astype(np.uint8)); m=im.filter(ImageFilter.MaxFilter(2*max(1,int(round(2.5*k)))+1)).filter(ImageFilter.GaussianBlur(5*k))
        ring=np.asarray(im.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.6*k)),np.float32)/255.0      # ~1.5px 深色描边
        return A,np.asarray(m,np.float32)/255.0,np.clip(ring-K.astype(np.float32),0,1)
    def sat(self,t):
        """崩坏滤镜：游戏层全局饱和度 S(t)（控制点表 work/sat_curve.csv，线性插值后 ~0.25s 汉宁平滑）"""
        if LEGACY: return 1.0
        if not hasattr(self,'_sat'):
            pts=[tuple(map(float,l.split(','))) for l in open('work/sat_curve.csv').read().split('\n')[1:] if l.strip()]
            tt=np.arange(0,T+1,1/60); v=np.interp(tt,[p[0] for p in pts],[p[1]/100 for p in pts]); k=np.hanning(15); k/=k.sum(); self._sat=(tt,np.convolve(np.pad(v,7,mode='edge'),k,'valid'))
        return float(np.interp(t,*self._sat))
    # ---- 单帧
    def frame(self,kf,t,shots,s,plan,cache,bga):
        W,H,k=self.W,self.H,self.k; fx=self.fx; tau=t-s['t']; base=s['base']; ev=s['events']
        crop,grad=crop_for(s)
        win=R4.interp_win(shots,s,t); wh,ww,x,y=self.geometry(win,crop)
        key=(s['n'],wh)
        if key not in cache: cache.clear(); cache[key]=decode_frames2(s,wh,plan[s['n']],crop)
        frames,ww=cache[key]; x=(W-ww)//2
        img=frames[src_index_c(s,tau)].astype(np.float32)
        # ---- 窗内：自然色彩（不调色、不降饱和）；形变类特效
        if 'contrast' in base: piv=img.mean(); img=piv+(img-piv)*base['contrast']
        if 'dim' in base: img=img*base['dim']
        z=1.0
        for off,typ,nf in ev:                                                      # 吞噬重音：缩放脉冲 ≤4%（3–4 帧衰减）
            if typ=='hit' and off<=tau<off+nf*FR: z=max(z,1+0.04*(1-(tau-off)/(nf*FR)))
        if 'beatpulse' in base:
            amp,every=base['beatpulse']; bb=R4.BEATS[s['b0']:s['b0']+s['beats']+1:every]; prev=bb[bb<=t+0.012]
            if len(prev): z=max(z,1+amp*math.exp(-max(t-prev[-1],0)/0.12))
        img=fx.zoom(img,z); rgb_now=False; accent_now=False
        S=self.sat(t)
        if S<0.999: L_=img@np.array([0.299,0.587,0.114],np.float32); img=L_[...,None]+(img-L_[...,None])*S          # 只对游戏层去饱和，BGA 文字层不受影响
        for off,typ,nf in ev:
            if not(off<=tau<off+nf*FR): continue
            if typ=='rgb': rgb_now=True
            elif typ=='block': img=fx.block(img)
            elif typ=='tear': img=fx.tear(img)
            elif typ=='bga_accent': accent_now=True
            elif typ=='hit' and tau<off+2*FR: rgb_now=True                       # 吞噬重音：前 2 帧 RGB 色差
        if 'shake' in base: img=fx.shake(img,base['shake'])
        if not LEGACY: img=self.terminalize_icons(img,crop,wh,ww)
        img_pre=img.copy()                                                       # 溶解前的游戏层：光晕颜色取自它（背景变暗不得作用于 BGA 层）
        # ---- 字符溶解（锚点驱动；字符色 = 米黄）
        covpx=None; act=[]; col=np.asarray(self.gd.p.glyph_color,np.float32)
        for e in self.sched:
            st=self.state(e,t)
            if st is None: continue
            if e['kind']=='out' and st.get('p',0)>0 and not LEGACY: L_=img@np.array([0.299,0.587,0.114],np.float32); img=L_[...,None]+(img-L_[...,None])*max(0.0,1-0.85*st['p'])   # 溶出覆盖期间饱和度随覆盖率下降（100% 时 ≤15%）
            img,m=self.gd.render_state(img,e['id'],st['refresh'],col,p=st.get('p'),front=st.get('front'),direction=st.get('direction','h'),rects=st.get('rects'),alive=st.get('alive'),tear_fn=fx.tear,tear=st.get('tear',False),no_dim=(not LEGACY and st.get('rects') is not None))
            cp=np.repeat(np.repeat(m,self.gd.ch,0),self.gd.cw,1); full=np.zeros((wh,ww),np.float32); full[:cp.shape[0],:cp.shape[1]]=cp[:wh,:ww]
            covpx=full if covpx is None else np.maximum(covpx,full)
            act.append(dict(id=e['id'],kind=e['kind'],p=round(float(st.get('p',st.get('front',1.0) if 'front' in st else 1.0)),3),refresh=st['refresh']))
        img=np.clip(img,0,255)
        if not LEGACY: img=self.finish_game(img,crop,grad); img_pre=self.finish_game(img_pre,crop,grad)
        else: img_pre=img
        # ---- BGA：原始背景（纯黑）+ 字符墨水
        b=bga.read().astype(np.float32); b,key_ink=clean_bga(b)
        j=np.searchsorted(FT,t); last=FT[j-1] if j>0 else -9; pulse=math.exp(-(t-last)/0.12) if t-last<0.4 else 0
        if accent_now: pulse=max(pulse,1.0)+0.5
        ink=np.minimum(b*(1+0.3*pulse),255.0)*self.ink
        M,Cn,fall=self.fields(wh,ww,x,y)
        # ---- 合成：窗口内容向黑渐变；外溢光晕（取自画面边缘 ~40px 平均色）
        full=np.zeros((H,W,3),np.float32); full[y:y+wh,x:x+ww]=img
        canvas=full*M[...,None]
        if not LEGACY:
            TA,TH,TO=self.text_layer(key_ink)
            if self.ink<=0: TA=TA*0; TH=TH*0; TO=TO*0                                  # 关闭文字层的诊断渲染
            canvas=canvas*(1-0.50*TH)[...,None]                                          # 字形暗晕（压暗 50%）
            canvas=canvas*(1-0.75*TO)[...,None]                                          # 1.5px 深色描边（不透明度 75%）               # 跟字形走的暗晕：游戏层压暗 50%
        full=np.zeros((H,W,3),np.float32); full[y:y+wh,x:x+ww]=img_pre                  # 以下光晕取色用溶解前的画面
        bw=int(40*k); inner=np.zeros((H,W),np.float32); inner[y+bw:y+wh-bw,x+bw:x+ww-bw]=1; win_m=np.zeros((H,W),np.float32); win_m[y:y+wh,x:x+ww]=1; band=win_m-inner
        num=blur_img(full*band[...,None],60*k); den=blur_img(band*255.0,60*k)/255.0
        colf=np.clip(num/np.maximum(den[...,None],0.03),0,255)
        s0=self.P['spill_strength']; inside_term=(1-M)*win_m                 # 窗内渐变带：向光晕色（而非纯黑）过渡，保证由内向外单调
        spill=colf*(s0*fall+s0*inside_term)[...,None]
        canvas=255-(255-canvas)*(255-np.minimum(spill,255))/255
        if LEGACY:
            Gm=(1-M)*0.50+M*(0.25-0.17*Cn)
            if covpx is not None:
                cp_full=np.zeros((H,W),np.float32); cp_full[y:y+wh,x:x+ww]=covpx; Gm=np.maximum(Gm,cp_full*M*0.75)
            canvas=255-(255-canvas)*(255-ink*Gm[...,None])/255
        else:
            canvas=fx.vignette(canvas,base.get('vignette',0.25)*0.6)                         # 暗角只作用于游戏层+光晕：先于文字层，避免同一文字行随位置变暗
            hi=np.clip((canvas.mean(2,keepdims=True)-0.68*255)/(0.32*255),0,1)*canvas; canvas=255-(255-canvas)*(255-blur_img(hi,14*k)*0.25)/255   # 泛光同样先于文字层：亮球的泛光不得染亮文字行
            canvas=canvas*(1-0.9*(key_ink*(TA>0))[...,None])                                 # 字形内游戏层压暗 90%（跟字形走，被字形自身遮住），使行内字色不受亮球影响
            canvas=canvas*(1-TA[...,None])+ink*TA[...,None]                                   # 文字行：正常 alpha 叠加（不用 screen），字色保持 BGA 原色，位于游戏层之上
        # ---- 整帧统一后期
        if rgb_now: canvas=fx.rgb_split(canvas,max(1,int(W*0.006)))
        canvas=fx.scanlines(canvas,base.get('scan',0.1))
        if 'noise' in base: canvas=fx.add_noise(canvas,base['noise'])
        if LEGACY: canvas=fx.vignette(canvas,base.get('vignette',0.25)*0.6)
        if LEGACY:
            hi=np.clip((canvas.mean(2,keepdims=True)-0.68*255)/(0.32*255),0,1)*canvas; canvas=255-(255-canvas)*(255-blur_img(hi,14*k)*0.25)/255
        if s['flash'] and tau<3*FR:
            a=0.6*math.exp(-tau/0.05); canvas=canvas*(1-a)+255*a
        if t<1.5: canvas*=t/1.5
        if t>T-2.68: canvas*=max(0,(T-t)/2.68)
        canvas=np.clip(canvas,0,255); self.meta.append(dict(k=kf,t=round(t,4),shot=s['n'],x=x,y=y,ww=ww,wh=wh,dis=act,lum=round(float(canvas.mean()/255),4)))
        return canvas
def render_chunk(comp,c0,c1,fn,crf,maxrate,shots):
    W,H=comp.W,comp.H; n0=int(round(c0*FPS)); n1=int(round(c1*FPS))
    enc=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset',os.environ.get('PRESET','medium'),'-pix_fmt','yuv420p','-g',str(FPS*2)]
    enc+=(['-crf',str(crf),'-maxrate',maxrate,'-bufsize',str(2*int(maxrate[:-1]))+'k'] if maxrate else ['-crf',str(crf)])+['-f','mp4',fn]
    ep=subprocess.Popen(enc,stdin=subprocess.PIPE) if not PROBE else None; bga=R4.BgaReader(c0,W,H,content_fit=True); plan={}
    for s in shots:
        if s['te']<=c0 or s['t']>=c1: continue
        tau,src=R4.src_map(s); s['_map']=(tau,src)
        plan[s['n']]=sorted({src_index_c(s,(kk+0.5)*FR-s['t']) for kk in range(n0,n1) if s['t']<=(kk+0.5)*FR<s['te']})
    cache={}
    for kf in range(n0,n1):
        t=(kf+0.5)*FR; s=next((x for x in shots if x['t']<=t<x['te']),None)
        if s is None:                       # 第一个镜头之前 / 最后一个镜头之后：纯黑
            bga.read(); fr=np.zeros((H,W,3),np.float32); comp.meta.append(dict(k=kf,t=round(t,4),shot=0,x=0,y=0,ww=0,wh=0,dis=[],lum=0.0))
        else: fr=comp.frame(kf,t,shots,s,plan,cache,bga)
        if ep: ep.stdin.write(fr.astype(np.uint8).tobytes())
    if ep: ep.stdin.close(); ep.wait()
    bga.close()
def render(t0,t1,H,out,schedule,params=None,ink=1.0,crf=20,maxrate=None,chunk=10.0,snap=False):
    W=H*16//9; shots=R4.load_edl(); comp=Composer(W,H,schedule,params,ink)
    if snap: t0,t1=R4.snap_bounds(shots,t0,t1); print('snapped',t0,t1)
    d=out+'.chunks'; os.makedirs(d,exist_ok=True); chunks=[]; c0=t0
    while c0<t1-1e-6:
        c1=min(t1,c0+chunk); fn=os.path.join(d,f'c_{int(round(c0*1000)):07d}.mp4'); chunks.append(fn)
        if PROBE: render_chunk(comp,c0,c1,fn,crf,maxrate,shots)
        elif not os.path.exists(fn): render_chunk(comp,c0,c1,fn+'.part.mp4',crf,maxrate,shots); os.replace(fn+'.part.mp4',fn)
        c0=c1
    if PROBE:
        json.dump(dict(t0=t0,t1=t1,W=W,H=H,sched=comp.sched,frames=comp.meta),open(out+'.meta.json','w'),default=float); return out,t0,t1
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
