"""V3 渲染器：按 work/edl.csv 逐镜头合成（ffmpeg 解码/编码 + numpy/PIL 在工作分辨率做特效）。
用法：python3 -I render_v3.py --style B --t0 95 --t1 107 --height 540 --out work/preview/sampleB.mp4
       python3 -I render_v3.py --style B --height 540 --out work/preview/full540_B.mp4   # 整首，按 10s 分块可续跑
"""
import csv,json,os,sys,subprocess,argparse,pickle,math,numpy as np
from PIL import Image,ImageFilter,ImageFont,ImageDraw
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
FPS=30; FR=1/FPS; T=278.806
MUS='input2/music/CreditsEX.by.Frums.mp3'; BGA='input2/bga/Credits.bga.mp4'; BGA_OFF=0.16
WIN_CAP_B=0.88; BAR=0.022
CROP_TOP=0.06; SRC_W,SRC_H=3132,2160; CT=int(SRC_H*CROP_TOP)//2*2; CH=SRC_H-CT; ASPECT=SRC_W/CH
BGA_MAIN=np.array([41,59,79],np.float32)            # #293b4f 冷暗色
TONE_LO=np.array([41,59,79],np.float32); TONE_HI=np.array([225,222,205],np.float32)
A=pickle.load(open('work/audio_an.pkl','rb')); PK=A['pk']; AFPS=22050/256
ACC=json.load(open('work/bga_accent.json'))
CLIPS={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}
def hex2rgb(h): return np.array([int(h[i:i+2],16) for i in (1,3,5)],np.float32)
def accent_at(t):
    """强调色：BGA 字符前景色（10s 代表色）按同色相提亮到 L≈0.75"""
    c=hex2rgb(ACC['per10s'][min(len(ACC['per10s'])-1,int(t//10))][1])
    m=c.max()
    if m<8: c=hex2rgb(ACC['median']); m=c.max()
    c=c/m; lum=c.mean(); c=c+(0.75-lum)*0.9*(1-c)*2 if lum<0.75 else c
    return np.clip(c*255*0.85+30,0,255)
# ---------------- EDL
def load_edl():
    rows=[]
    for r in csv.DictReader(open('work/edl.csv',encoding='utf-8')):
        s=dict(n=int(r['shot']),seg=r['seg'],t=float(r['t_start']),te=float(r['t_end']),clip=int(r['clip_id']),src_in=float(r['src_in']),src_out=float(r['src_out']),
               speed=r['speed'],win=float(r['win_pct'])/100,variant=r['variant'],trig=r['trigger'],fx=r['fx'].split(),ascii=float(r['ascii_s']),flash=int(r['flash']))
        s['dur']=s['te']-s['t']; rows.append(s)
    return rows
def fxval(s,name):
    for f in s['fx']:
        if f.startswith(name+':'): return f.split(':',1)[1].split('@')[0]
    return None
def fx_offsets(s,name):
    out=[]
    for f in s['fx']:
        if f.startswith(name+':'): out.append(float(f.split('@')[1]) if '@' in f else 0.0)
    return out
def src_time(s,tau):
    """镜头内时间 tau → 源素材时间偏移（秒）"""
    sp=s['speed']
    if sp=='ramp':  # 0.5→1→0.5 线性三角
        d=s['dur']; x=min(tau,d)
        # v(x)=0.5+0.5*(1-|2x/d-1|)；积分
        h=d/2
        if x<=h: return 0.5*x+0.5*(x*x/(2*h))
        y=x-h; return (0.5*h+0.5*h/2)+0.5*y+0.5*(y-y*y/(2*h))
    v=float(sp)
    hold=fxval(s,'stutter')
    if hold:
        n=int(hold[:-1])*FR
        if tau<n: return 0.0
    return tau*v
# ---------------- 解码
def decode_window(s,wh):
    """解码镜头的源窗口，返回按输出帧需要的帧列表 (list of HxWx3 uint8)。wh=窗口高度(px)"""
    c=s['clip']; r=CLIPS[c]; sfps=float(r['fps']); ww=int(round(wh*ASPECT))//2*2
    vf=[f'crop=iw:{CH}:0:{CT}']
    v=s['variant']
    if v.startswith('fill:'):
        z=int(v[9:v.index('%')])/100; cx,cy=[float(x) for x in v[v.index('(')+1:v.index(')')].split(',')]
        cw,ch=SRC_W/z,CH/z; x=min(max(cx*SRC_W-cw/2,0),SRC_W-cw); y=min(max(cy*CH-ch/2,0),CH-ch)
        vf.append(f'crop={cw:.0f}:{ch:.0f}:{x:.0f}:{y:.0f}')
    elif v.startswith('zoom'):
        z=int(v[4:v.index('%')])/100; pos=v.split('@')[1]
        cw,ch=SRC_W/z,CH/z; x={'TL':0,'BL':0,'C':(SRC_W-cw)/2}.get(pos,SRC_W-cw); y={'TL':0,'TR':0,'C':(CH-ch)/2}.get(pos,CH-ch)
        vf.append(f'crop={cw:.0f}:{ch:.0f}:{x:.0f}:{y:.0f}')
    vf.append(f'scale={ww}:{wh}:flags=bicubic')
    need=s['src_out']-s['src_in']+0.1
    p=subprocess.Popen(['ffmpeg','-v','error','-ss',f"{s['src_in']:.3f}",'-t',f'{need:.3f}','-i','input2/clips/'+r['file'],'-an','-vf',','.join(vf),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    fs=ww*wh*3; frames=[]
    while True:
        b=p.stdout.read(fs)
        if len(b)<fs: break
        frames.append(np.frombuffer(b,np.uint8).reshape(wh,ww,3))
    p.wait()
    return frames,sfps
class BgaReader:
    def __init__(self,t_start,W,H,content_fit=False):
        self.W,self.H=W,H
        # 裁掉 BGA 自身左右黑边（取 8–92% 列），铺满画布宽，上下居中裁
        vf=f'fps={FPS},crop=iw*0.84:ih:iw*0.08:0,scale={W}:-2,crop={W}:{H}' if content_fit else f'fps={FPS},scale={W}:{H}'
        self.p=subprocess.Popen(['ffmpeg','-v','error','-ss',f'{t_start+BGA_OFF:.3f}','-i',BGA,'-an','-vf',vf,'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
        self.last=np.zeros((H,W,3),np.uint8)
    def read(self):
        b=self.p.stdout.read(self.W*self.H*3)
        if len(b)==self.W*self.H*3: self.last=np.frombuffer(b,np.uint8).reshape(self.H,self.W,3)
        return self.last   # 不足补末帧
    def close(self):
        try: self.p.stdout.close(); self.p.kill()
        except Exception: pass
# ---------------- 特效原语（float32 0–255）
class FX:
    def __init__(self,W,H):
        self.W,self.H=W,H
        self.vigc={}
        per=max(2,round(H/270)); row=np.ones(H,np.float32); row[::per]=0.0
        self.scan=row[:,None,None]
        rng=np.random.default_rng(3); self.noise=[rng.normal(0,1,(H,W,1)).astype(np.float32) for _ in range(12)]; self.rng=rng
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',max(10,round(H/45)))
        self.cell=(max(6,round(W/120)),max(10,round(H/45)))
        self.ramp=" .:-=+*#%@"; self.glyphs=[]
        cw,ch=self.cell
        for ch_ in self.ramp:
            im=Image.new('L',(cw,ch),0); ImageDraw.Draw(im).text((0,-1),ch_,fill=255,font=self.font); self.glyphs.append(np.asarray(im,np.float32)/255)
        self.ascii_order=rng.random((H//ch+1,W//cw+1)).astype(np.float32)
    def vig(self,h,w):
        if (h,w) not in self.vigc:
            yy,xx=np.mgrid[0:h,0:w]; r2=((xx-w/2)/(w/2))**2+((yy-h/2)/(h/2))**2
            self.vigc[(h,w)]=(1-np.clip(r2,0,1.4)*0.55)[...,None].astype(np.float32)
        return self.vigc[(h,w)]
    def vignette(self,img,a): return img*(1-a)+img*self.vig(*img.shape[:2])*a
    def scanlines(self,img,a): return img*(1-a*self.scan[:img.shape[0]])
    def add_noise(self,img,a):
        h,w=img.shape[:2]; n=self.noise[self.rng.integers(len(self.noise))]; return img+np.roll(n,self.rng.integers(h),0)[:h,:w]*a*45
    def rgb_split(self,img,d):
        out=img.copy(); out[:,:,0]=np.roll(img[:,:,0],-d,1); out[:,:,2]=np.roll(img[:,:,2],d,1); return out
    def block(self,img,strength=1.0):
        out=img.copy(); H=img.shape[0]
        for _ in range(self.rng.integers(4,9)):
            y=self.rng.integers(0,H-8); h=self.rng.integers(4,max(6,H//10)); dx=int(self.rng.uniform(-0.08,0.08)*img.shape[1]*strength)
            out[y:y+h]=np.roll(img[y:y+h],dx,1)
        return self.rgb_split(out,max(1,img.shape[1]//200))
    def tear(self,img):
        y=self.rng.integers(img.shape[0]//4,img.shape[0]*3//4); dx=int(self.rng.uniform(0.02,0.04)*img.shape[1]*self.rng.choice([-1,1]))
        out=img.copy(); out[:y]=np.roll(img[:y],dx,1); return out
    def zoom(self,img,z):
        if z<=1.002: return img
        h,w=img.shape[:2]; im=Image.fromarray(np.clip(img,0,255).astype(np.uint8)).resize((int(w*z)+1,int(h*z)+1),Image.BILINEAR)
        x=(im.size[0]-w)//2; y=(im.size[1]-h)//2; return np.asarray(im.crop((x,y,x+w,y+h)),np.float32)
    def shake(self,img,amp):
        dx=int(self.rng.uniform(-amp,amp)*img.shape[1]); dy=int(self.rng.uniform(-amp,amp)*img.shape[0]); out=np.zeros_like(img)
        h,w=img.shape[:2]; out[max(0,dy):h+min(0,dy),max(0,dx):w+min(0,dx)]=img[max(0,-dy):h-max(0,dy),max(0,-dx):w-max(0,dx)]; return out
    def tone(self,img,k):
        L=(img.mean(2,keepdims=True)/255); return img*(1-k)+k*(TONE_LO*(1-L)+TONE_HI*L)
    def desat(self,img,a): L=img.mean(2,keepdims=True); return img*(1-a)+L*a
    def ascii(self,img,p,color):
        """渐进 ASCII 化：p∈[0,1] 为已转换格子的比例"""
        cw,ch=self.cell; h,w=img.shape[:2]; R,C=h//ch,w//cw
        L=img[:R*ch,:C*cw].mean(2).reshape(R,ch,C,cw).mean((1,3))/255
        idx=np.clip((L*(len(self.ramp)-1)+0.5).astype(int),0,len(self.ramp)-1)
        G=np.stack(self.glyphs)  # (n,ch,cw)
        tiles=G[idx]             # (R,C,ch,cw)
        asc=tiles.transpose(0,2,1,3).reshape(R*ch,C*cw)[...,None]*color[None,None,:]
        mask=(self.ascii_order[:R,:C]<p).astype(np.float32)
        m=np.repeat(np.repeat(mask,ch,0),cw,1)[...,None]
        out=img.copy(); out[:R*ch,:C*cw]=img[:R*ch,:C*cw]*(1-m)+(asc*0.95+8)*m
        return out
# ---------------- 主渲染
def render(style,t0,t1,H,out,crf=20,maxrate=None,chunk=10.0):
    W=H*16//9; fx=FX(W,H); shots=load_edl()
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    tmpdir=out+'.chunks'; os.makedirs(tmpdir,exist_ok=True)
    chunks=[]; c0=t0
    while c0<t1-1e-6:
        c1=min(t1,c0+chunk); fn=os.path.join(tmpdir,f'c_{int(round(c0*1000)):07d}.mp4'); chunks.append((c0,c1,fn))
        if not os.path.exists(fn):
            render_chunk(style,c0,c1,W,H,fx,shots,fn+'.part.mp4',crf,maxrate); os.replace(fn+'.part.mp4',fn)
        c0=c1
    lst=os.path.join(tmpdir,'list.txt'); open(lst,'w').write(''.join(f"file '{os.path.abspath(f)}'\n" for _,_,f in chunks))
    dur=t1-t0
    af=f'atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS,'
    if t0<0.01: af+='afade=t=in:st=0:d=1.5,'
    if t1>T-0.1: af+=f'afade=t=out:st={T-2.68:.3f}:d=2.68,'
    af+='loudnorm=I=-10.5:TP=-1.0:LRA=11'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',lst,'-i',MUS,'-af',af,'-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out],check=True)
    return out
def render_chunk(style,c0,c1,W,H,fx,shots,fn,crf,maxrate):
    n0=int(round(c0*FPS)); n1=int(round(c1*FPS))
    enc=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-pix_fmt','yuv420p','-g',str(FPS*2)]
    enc+=(['-b:v',maxrate,'-maxrate',maxrate,'-bufsize',maxrate] if maxrate else ['-crf',str(crf)])+['-f','mp4',fn]
    ep=subprocess.Popen(enc,stdin=subprocess.PIPE)
    bga=BgaReader(c0,W,H,content_fit=(style=='B')) if style=='B' or c0<16 or c1>266 else None
    # 帧→镜头：帧中心时间所在镜头
    si=0
    cache={}  # shot n -> (frames,sfps,wh)
    pk=PK
    for k in range(n0,n1):
        t=(k+0.5)*FR
        while si+1<len(shots) and shots[si+1]['t']<=t: si+=1
        while si>0 and shots[si]['t']>t: si-=1
        s=shots[si]; tau=t-s['t']
        # 取景窗高度：相邻镜头 win 线性插值（按镜头中点）
        w0=s['win']; mid=s['t']+s['dur']/2
        if t<mid and si>0: wn=shots[si-1]; m2=wn['t']+wn['dur']/2; w=w0+(wn['win']-w0)*(mid-t)/(mid-m2)
        elif t>=mid and si+1<len(shots): wn=shots[si+1]; m2=wn['t']+wn['dur']/2; w=w0+(wn['win']-w0)*(t-mid)/(m2-mid)
        else: w=w0
        if style=='B': w=min(w,WIN_CAP_B)
        wh=int(round(H*w))//2*2; wh=min(wh,H)
        if s['n'] not in cache or cache[s['n']][2]!=wh:
            frames,sfps=decode_window(s,wh); cache={s['n']:(frames,sfps,wh)}
        frames,sfps,_=cache[s['n']]
        st=src_time(s,tau); fi=min(len(frames)-1,int(st*sfps)) if frames else 0
        img=frames[fi].astype(np.float32) if frames else np.zeros((wh,int(wh*ASPECT)//2*2,3),np.float32)
        e=0  # 特效
        tone=fxval(s,'tone'); img=fx.tone(img,0.28 if tone else 0.15)
        d=fxval(s,'desat');  img=fx.desat(img,float(d)) if d else img
        zp=fxval(s,'zoompulse')
        if zp:
            amp=float(zp[:-1])/100; z=1.0
            for off in fx_offsets(s,'zoompulse'):
                if tau>=off: z=max(z,1+amp*math.exp(-(tau-off)/0.12))
            img=fx.zoom(img,z)
        rg=fxval(s,'rgb')
        if rg:
            n=int(rg[:-1])
            for off in fx_offsets(s,'rgb'):
                if off<=tau<off+n*FR: img=fx.rgb_split(img,max(1,int(img.shape[1]*0.006)))
        bl=fxval(s,'block')
        if bl and tau<int(bl[:-1])*FR: img=fx.block(img)
        tr=fxval(s,'tear')
        if tr and tau<int(tr[:-1])*FR: img=fx.tear(img)
        sh=fxval(s,'shake')
        if sh: img=fx.shake(img,float(sh[:-1])/100)
        sc=fxval(s,'scan'); img=fx.scanlines(img,float(sc) if sc else 0.1)
        no=fxval(s,'noise')
        if no: img=fx.add_noise(img,float(no))
        vg=fxval(s,'vignette'); img=fx.vignette(img,float(vg) if vg else 0.25)
        acc=accent_at(t)
        if s['ascii'] and tau>=s['dur']-s['ascii']:
            p=(tau-(s['dur']-s['ascii']))/s['ascii']; img=fx.ascii(img,min(1,p*1.05),acc)
        if s['flash'] and tau<3*FR:
            a=0.6*math.exp(-tau/0.05); img=img*(1-a)+255*a
        img=np.clip(img,0,255)
        # ---- 背景
        wl=img.mean()/255
        if style=='A':
            im=Image.fromarray(img.astype(np.uint8)); z=max(W/im.size[0],H/im.size[1])*1.15
            im=im.resize((int(im.size[0]*z)+1,int(im.size[1]*z)+1),Image.BILINEAR).filter(ImageFilter.GaussianBlur(H/40))
            x=(im.size[0]-W)//2; y=(im.size[1]-H)//2; bg=np.asarray(im.crop((x,y,x+W,y+H)),np.float32)*0.42
            if bga is not None:
                b=bga.read().astype(np.float32); g=255*0.9/max(30,np.percentile(b,99.5)); b=np.clip(b*g,0,255)
                a=min(1,(16-t)/10) if t<16 else min(1,(t-266)/8) if t>266 else 0
                if a>0: bg=bg*(1-a*0.6)+b*a*0.8
        else:
            b=bga.read().astype(np.float32); g=255*0.9/max(30,np.percentile(b,99.5)); b=np.clip(b*g,0,255)
            # 起音脉冲
            j=np.searchsorted(pk,t); last=pk[j-1] if j>0 else -9; pulse=math.exp(-(t-last)/0.12) if t-last<0.4 else 0
            if 'bga_accent' in s['fx'] and tau<4*FR: pulse=max(pulse,1.0)+0.5
            b=b*(1+0.4*pulse)
            target=min(0.30,0.85*wl); base=BGA_MAIN*(target*255/BGA_MAIN.mean())*0.85
            bg=np.clip(base+b*(1-0.3),0,255)
            if bg.mean()/255>wl*0.95: bg=bg*(wl*0.95*255/bg.mean())
        # ---- 合成
        out=bg.copy(); wh_,ww_=img.shape[:2]; x=(W-ww_)//2; y=(H-wh_)//2
        out[y:y+wh_,x:x+ww_]=img
        bar=max(3,int(round(wh_*BAR)))
        out[y:y+bar,x:x+ww_]=BGA_MAIN*0.6
        if style=='B':
            lw=max(1,H//540); col=acc*(1.6 if ('bga_accent' in s['fx'] and tau<4*FR) else 1.0)
            out[y+bar-lw:y+bar,x:x+ww_]=col; out[y+wh_-lw:y+wh_,x:x+ww_]=col; out[y:y+wh_,x:x+lw]=col; out[y:y+wh_,x+ww_-lw:x+ww_]=col
        else:
            out[y+bar-1:y+bar,x:x+ww_]=acc*0.5
        if t<1.5: out*=t/1.5
        if t>T-2.68: out*=max(0,(T-t)/2.68)
        ep.stdin.write(np.clip(out,0,255).astype(np.uint8).tobytes())
    ep.stdin.close(); ep.wait()
    if bga: bga.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--style',default='B'); ap.add_argument('--t0',type=float,default=0); ap.add_argument('--t1',type=float,default=T)
    ap.add_argument('--height',type=int,default=540); ap.add_argument('--out',required=True); ap.add_argument('--crf',type=int,default=20); ap.add_argument('--maxrate',default=None); ap.add_argument('--chunk',type=float,default=10.0)
    a=ap.parse_args(); print(render(a.style,a.t0,a.t1,a.height,a.out,a.crf,a.maxrate,a.chunk))
