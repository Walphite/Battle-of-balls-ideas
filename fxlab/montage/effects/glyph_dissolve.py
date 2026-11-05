"""字符溶解（glyph dissolve）预设族。
默认参数 = 原样抽取自 render_v4.FX.ascii（sample_C 里 144.3 s 的 ASCII 转场）：
  - 字符格 W/120 × H/45（960×540 下 8×12 px），字符集 " .:-=+*#%@"，格子 = 纯黑底(floor 8) + 字符色 ×0.95
  - 覆盖掩码 = 逐格随机顺序阈值 (order < p)（legacy）；字符色默认取 BGA 字符前景色（调用方传入）
新增（默认关闭，不改变 legacy 视觉）：noise_scale>0 成团覆盖、focus_protect 主体最晚覆盖、bg_dim、bga_reveal。
变体：out 溶出 / in 凝出 / scan 扫描式 / local 局部溶解 / ring 边缘带 / final 收束（字符熄灭至黑屏）。
字符格内容与覆盖比例按 hold_hz（≤12）量化更新，避免高频闪烁。"""
import numpy as np
from dataclasses import dataclass
from PIL import Image,ImageDraw,ImageFont,ImageFilter

@dataclass
class GlyphParams:
    cell_w: int = None            # None → round(W/120)  (legacy)
    cell_h: int = None            # None → round(H/45)   (legacy)
    charset: str = " .:-=+*#%@"
    glyph_color: tuple = None     # None → 调用方传入的 BGA 字符前景色
    glyph_gain: float = 0.95
    cell_floor: float = 8.0
    noise_scale: float = 0.0      # 0 = 逐格随机(legacy)；>0 = 低频噪声成团，单位：格
    focus_protect: float = 0.0    # 0 = 关；1 = 最亮/最大的主体最晚被覆盖
    bg_dim: float = 0.0           # 未覆盖区域暗化比例
    bga_reveal: float = 0.0       # 字符态时窗内 BGA 墨水增益的额外倍数（合成层读取）
    hold_hz: float = 10.0
    seed: int = 3
    font_path: str = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    # 时序（秒）
    tear: float = 0.15; still: float = 0.35; cover: float = 0.35; hold: float = 0.40
    restore: float = 0.30; sweep: float = 0.50; sweep_hold: float = 0.10
    final_cover: float = 1.2; final_fade: float = 2.1; final_black: float = 0.5
    @classmethod
    def spec_look(cls): return cls(noise_scale=4.0,focus_protect=0.8,bg_dim=0.35,bga_reveal=1.0)

class GlyphDissolve:
    def __init__(self,W,H,params=None,order=None):
        self.W,self.H=W,H; self.p=params or GlyphParams()
        self.cw=self.p.cell_w or max(6,round(W/120)); self.ch=self.p.cell_h or max(10,round(H/45))
        self.rng=np.random.default_rng(self.p.seed)
        font=ImageFont.truetype(self.p.font_path,max(10,round(H/45)))
        gl=[]
        for c in self.p.charset:
            im=Image.new('L',(self.cw,self.ch),0); ImageDraw.Draw(im).text((0,-1),c,fill=255,font=font); gl.append(np.asarray(im,np.float32)/255)
        self.G=np.stack(gl); self.n=len(self.p.charset)
        self.order=order if order is not None else self.rng.random((H//self.ch+1,W//self.cw+1)).astype(np.float32)
        self.order2=self.rng.random((H//self.ch+1,W//self.cw+1)).astype(np.float32)   # 熄灭顺序
        self._cl={}; self._cache={}
    # ---------- 基础
    def tiles(self,img):
        h,w=img.shape[:2]; R,C=h//self.ch,w//self.cw
        L=img[:R*self.ch,:C*self.cw].mean(2).reshape(R,self.ch,C,self.cw).mean((1,3))/255
        return np.clip((L*(self.n-1)+0.5).astype(int),0,self.n-1),L
    def cells(self,idx,color,gain=1.0):
        R,C=idx.shape; g=self.G[idx].transpose(0,2,1,3).reshape(R*self.ch,C*self.cw)[...,None]
        return (g*np.asarray(color,np.float32)[None,None,:]*self.p.glyph_gain*gain+self.p.cell_floor)
    def order_field(self,R,C,L=None):
        """覆盖顺序场 ∈[0,1)：legacy 逐格随机；noise_scale>0 成团；focus_protect 让亮主体更晚"""
        if self.p.noise_scale<=0: o=self.order[:R,:C].copy()
        else:
            key=(R,C)
            if key not in self._cl:
                f=self.rng.random((R,C)).astype(np.float32); im=Image.fromarray((f*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(self.p.noise_scale))
                a=np.asarray(im,np.float32); rank=np.argsort(np.argsort(a.ravel())).reshape(R,C)/(R*C); self._cl[key]=rank.astype(np.float32)
            o=self._cl[key]
        if self.p.focus_protect>0 and L is not None:
            Ls=np.asarray(Image.fromarray((np.clip(L,0,1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),np.float32)/255
            rk=np.argsort(np.argsort(Ls.ravel())).reshape(R,C)/(R*C); o=o*(1-self.p.focus_protect)+rk*self.p.focus_protect
        return o
    def paint(self,img,idx,mask,color,gain=1.0,dim_rest=None):
        """mask(R,C) 0–1：1 = 字符格（纯黑底+字符），0 = 保留原画面"""
        R,C=idx.shape; out=img.copy(); asc=self.cells(idx,color,gain)
        m=np.repeat(np.repeat(mask.astype(np.float32),self.ch,0),self.cw,1)[...,None]
        reg=out[:R*self.ch,:C*self.cw]
        dim=self.p.bg_dim if dim_rest is None else dim_rest
        out[:R*self.ch,:C*self.cw]=reg*(1-dim*(1-m))*(1-m)+asc*m      # dim=0 时与 legacy 完全一致
        return out
    # ---------- 时序
    def total(self,kind,scale=1.0,dur=None):
        p=self.p
        if kind=='out': return (p.tear+p.still+p.cover+p.hold)*scale
        if kind=='in': return (p.hold+p.restore)*scale
        if kind=='scan': return p.sweep+p.sweep_hold
        if kind=='final': return p.final_cover+p.final_fade+p.final_black
        return dur or 0.2
    def coverage(self,kind,tau,scale=1.0):
        p=self.p
        if kind=='out':
            t2=(p.tear+p.still)*scale; return float(np.clip((tau-t2)/(p.cover*scale),0,1))
        if kind=='in':
            h=p.hold*scale; return float(np.clip(1-(tau-h)/(p.restore*scale),0,1))
        if kind=='final': return float(np.clip(tau/p.final_cover,0,1))
        return 1.0
    def apply(self,img,kind,tau,color,key=0,scale=1.0,tear_fn=None,rects=None,direction='h',quant=True):
        """对窗口图像 img 施加字符溶解；tau = 距事件起点的秒数。返回 img。"""
        h,w=img.shape[:2]; R,C=h//self.ch,w//self.cw; p=self.p
        q=int(tau*p.hold_hz+1e-9) if quant else tau; tq=q/p.hold_hz if quant else tau
        ck=(key,q); 
        if self._cache.get('k')!=key: self._cache={'k':key}
        if ck not in self._cache:
            idx,L=self.tiles(img); self._cache={'k':key,ck:(idx,L)}      # 内容在量化步内保持
        idx,L=self._cache[ck]
        if idx.shape!=(R,C):       # 量化步内窗口尺寸在变：最近邻重采样缓存的字符内容，保持 ≤hold_hz 的内容更新
            ri=np.minimum((np.arange(R)*idx.shape[0]/R).astype(int),idx.shape[0]-1); ci=np.minimum((np.arange(C)*idx.shape[1]/C).astype(int),idx.shape[1]-1)
            idx=idx[np.ix_(ri,ci)]; L=L[np.ix_(ri,ci)]
        out=img
        if kind=='out' and tau<p.tear*scale and tear_fn is not None: out=tear_fn(out)
        if kind in('out','in','final'):
            cov=self.coverage(kind,tq,scale)
            if kind=='final' and tq>=p.final_cover:
                prog=np.clip((tq-p.final_cover)/p.final_fade,0,1); alive=(self.order2[:R,:C]>=prog)
                full=np.ones((R,C),np.float32); res=self.paint(out,idx,full,color)
                dead=np.repeat(np.repeat((~alive).astype(np.float32),self.ch,0),self.cw,1)[...,None]
                res[:R*self.ch,:C*self.cw]*=(1-dead); return res
            if cov<=0: return out
            mask=(self.order_field(R,C,L)<cov).astype(np.float32) if cov<1 else np.ones((R,C),np.float32)
            return self.paint(out,idx,mask,color)
        if kind=='scan':
            fr=np.clip(tq/p.sweep,0,1.02); ax=(np.arange(C)[None,:]/C if direction=='h' else np.arange(R)[:,None]/R)*np.ones((R,C))
            mask=(ax<fr).astype(np.float32); res=self.paint(out,idx,mask,color)
            if fr<1.0:      # 终端光标：前沿一格更亮（最重字符）
                cur=(np.abs(ax-fr)<(1.0/(C if direction=='h' else R))).astype(np.float32)
                res=self.paint(res,np.full_like(idx,self.n-1),cur,np.minimum(255,np.asarray(color)*1.2))
            return res
        if kind=='local':
            mask=np.zeros((R,C),np.float32)
            for (u0,v0,u1,v1) in rects or []:
                mask[int(v0*R):max(int(v0*R)+1,int(v1*R)),int(u0*C):max(int(u0*C)+1,int(u1*C))]=1
            return self.paint(out,idx,mask,color)
        return out
    def ring(self,canvas,rect,band,color,key=0,boost=1.8,alpha=0.5,quant=True,tau=0.0,ring_gain=0.42):
        """边缘带：窗口矩形 rect=(x,y,ww,wh) 外沿 band px 的字符格（内容取自当前合成图亮度）"""
        H,W=canvas.shape[:2]; R,C=H//self.ch,W//self.cw; p=self.p
        q=int(tau*p.hold_hz+1e-9) if quant else tau
        if self._cache.get('rk')!=(key,q):
            idx,L=self.tiles(np.clip(canvas*boost,0,255)); self._cache['rk']=(key,q); self._cache['r']=idx
        idx=self._cache['r']; x,y,ww,wh=rect
        cx=(np.arange(C)+0.5)*self.cw; cy=(np.arange(R)+0.5)*self.ch
        dx=np.maximum(np.maximum(x-cx,cx-(x+ww)),0)[None,:]; dy=np.maximum(np.maximum(y-cy,cy-(y+wh)),0)[:,None]
        d=np.sqrt(dx**2+dy**2); mask=((d>0)&(d<=band)).astype(np.float32)*alpha*np.clip(1-d/band*0.6,0,1)   # 由内向外渐隐，避免硬边
        return self.paint(canvas,idx,mask,color,gain=ring_gain)
