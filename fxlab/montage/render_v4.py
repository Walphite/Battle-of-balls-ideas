"""V4 渲染器：按 work/edl.csv（片段不可分割版）逐帧合成。ffmpeg 解码/编码，numpy/PIL 在工作分辨率做特效。
用法：python3 -I render_v4.py --style B --t0 95 --t1 107 --snap --height 540 --out work/preview/sample_B.mp4
     python3 -I render_v4.py --style B --height 540 --maxrate 2300k --out work/preview/full540_B.mp4   # 整首，10s 分块可续跑"""
import csv,json,os,sys,subprocess,argparse,pickle,math,numpy as np
from PIL import Image,ImageFilter,ImageFont,ImageDraw
os.chdir(os.path.dirname(os.path.abspath(__file__)))
FPS=30; FR=1/FPS; T=278.806
MUS='input2/music/CreditsEX.by.Frums.mp3'; BGA='input2/bga/Credits.bga.mp4'; BGA_OFF=0.16
CROP_TOP=0.06; SRC_W,SRC_H=3132,2160; CT=int(SRC_H*CROP_TOP)//2*2; CH=SRC_H-CT; ASPECT=SRC_W/CH
WIN_CAP_B=0.88; BAR=0.022
BGA_MAIN=np.array([41,59,79],np.float32); TONE_LO=np.array([41,59,79],np.float32); TONE_HI=np.array([225,222,205],np.float32)
A=pickle.load(open('work/audio_an.pkl','rb')); PK=A['pk']
ACC=json.load(open('work/bga_accent.json')); BEATS=np.array(json.load(open('work/beats_v4.json'))['beats'])
CLIPS={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}
EVENT_TYPES=('rgb','block','tear','stutter','bga_accent','hit')
def hex2rgb(h): return np.array([int(h[i:i+2],16) for i in (1,3,5)],np.float32)
def accent_at(t):
    c=hex2rgb(ACC['per10s'][min(len(ACC['per10s'])-1,int(t//10))][1]); m=c.max()
    if m<8: c=hex2rgb(ACC['median']); m=c.max()
    c=c/m; lum=c.mean(); c=c+(0.75-lum)*0.9*(1-c)*2 if lum<0.75 else c
    return np.clip(c*255*0.85+30,0,255)
# ---------------- EDL
def parse_fx(tokens):
    base={}; ev=[]
    for tok in tokens:
        off=0.0
        if '@' in tok: tok,o=tok.split('@'); off=float(o)
        name,_,val=tok.partition(':')
        if name in EVENT_TYPES:
            ev.append((off,name,int(val[:-1]) if val else 4))
        elif name=='beatpulse':
            amp,every=val.split(':'); base['beatpulse']=(float(amp[:-1])/100,int(every))
        elif name in('shake',): base[name]=float(val[:-1])/100
        elif name=='tone': base[name]=val
        else: base[name]=float(val) if val else 1.0
    return base,sorted(ev)
def load_edl():
    out=[]
    for r in csv.DictReader(open(os.environ.get('EDL','work/edl.csv'),encoding='utf-8')):
        sp=r['speed']; s=dict(n=int(r['shot']),seg=r['seg'],t=float(r['t_start']),te=float(r['t_end']),beats=int(r['beats']),b0=int(r['beat_start']),clip=int(r['clip_id']),
            src_in=float(r['src_in']),src_out=float(r['src_out']),win=float(r['win_pct'])/100,ascii=float(r['ascii_s']),flash=int(r['flash']))
        s['dur']=s['te']-s['t']
        if sp.startswith('hit2'):
            p=sp.split(':'); s['mode']='hit2'; s['va']=float(p[1]); s['vb']=float(p[2]); s['tauh']=float(p[3]); s['w']=float(p[4]); s['dip']=float(p[5]); s['peak']=float(p[6]); s['v']=(s['va']+s['vb'])/2
        elif sp.startswith('hit'):
            p=sp.split(':'); s['mode']='hit'; s['va']=float(p[1]); s['vb']=float(p[2]); s['tauh']=float(p[3]); s['w']=float(p[4]); s['v']=(s['va']+s['vb'])/2
        elif sp.startswith('ramp'):
            p=sp.split(':'); s['mode']='ramp'; s['v']=float(p[1]); kv={k:float(v) for k,v in (x.split('=') for x in p[2:])}; s['hit']=kv['hit']; s['dip']=kv.get('dip',0.5); s['w']=kv.get('w',0.5)
        else: s['mode']='uniform'; s['v']=float(sp)
        s['base'],s['events']=parse_fx(r['fx'].split()); out.append(s)
    return out
def src_map(s):
    """返回 τ 网格与源时间（相对 src_in）数组。ramp：击中点附近梯形 dip（0.1 坡+0.3 平台+0.1 坡）降到 dip×。"""
    L=s['dur']; dt=1/600; n=int(L/dt)+2; tau=np.arange(n)*dt
    if s['mode']=='hit2':     # 两段基线速度 + 局部包络 m(τ)（吞噬前 0.4s 慢、吞噬后 0.12s 内加速再回落），与 plan_v9.m_profile 一致
        Oh=s['tauh']; sm_=lambda x:(lambda y:y*y*(3-2*y))(np.clip(x,0,1)); x=np.clip((tau-(Oh-s['w']))/(2*s['w']),0,1); sm=x*x*(3-2*x)
        pre=1-(1-s['dip'])*sm_((tau-(Oh-0.4))/0.3); up=s['dip']+(s['peak']-s['dip'])*sm_((tau-Oh)/0.12); down=s['peak']+(1-s['peak'])*sm_((tau-(Oh+0.12))/0.48)
        m=np.where(tau<=Oh,pre,np.where(tau<Oh+0.12,up,down)); v=m*(s['va']*(1-sm)+s['vb']*sm); src=np.cumsum(v)*dt; src-=src[0]; return tau,src
    if s['mode']=='hit':      # 吞噬前后两段速度，吞噬点两侧 ±w 内平滑缓动（smoothstep），不拆分素材
        x=np.clip((tau-(s['tauh']-s['w']))/(2*s['w']),0,1); sm=x*x*(3-2*x); v=s['va']+(s['vb']-s['va'])*sm; src=np.cumsum(v)*dt; src-=src[0]; return tau,src
    if s['mode']!='ramp': return tau,tau*s['v']
    v0=s['v']; w=s['w']; plateau=w-0.2; th=min(max(s['hit']/v0+0.1,0.3),L-0.3)
    a=th-w/2; m=np.ones(n)
    x=tau-a
    ramp_up=np.clip(x/0.1,0,1); ramp_dn=np.clip((w-x)/0.1,0,1); win=np.minimum(ramp_up,ramp_dn)*((x>=0)&(x<=w))
    m=1-(1-s['dip'])*win
    src=np.cumsum(m)*dt*v0; src-=src[0]
    return tau,src
# ---------------- 解码
def decode_frames(s,wh,need_idx):
    """解码源窗口，只保留 need_idx 里的帧（源帧序号，相对 src_in）。返回 dict idx->HxWx3 uint8, 窗口宽。"""
    r=CLIPS[s['clip']]; ww=int(round(wh*ASPECT))//2*2
    vf=f'crop=iw:{CH}:0:{CT},scale={ww}:{wh}:flags=bicubic'
    need=s['src_out']-s['src_in']+0.1
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
class BgaReader:
    def __init__(self,t_start,W,H,content_fit):
        self.W,self.H=W,H
        vf=(f'fps={FPS},crop=iw*0.84:ih:iw*0.08:0,scale={W}:-2,crop={W}:{H}' if (content_fit and os.environ.get('BGA_LEGACY')=='1') else f'fps={FPS},scale={W}:{H}')   # V10：BGA 整帧 100% 缩放（1280×720→画布，无裁切无平移）；旧映射会把最底文字行推出画面
        self.p=subprocess.Popen(['ffmpeg','-v','error','-ss',f'{t_start+BGA_OFF:.3f}','-i',BGA,'-an','-vf',vf,'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
        self.last=np.zeros((H,W,3),np.uint8)
    def read(self):
        b=self.p.stdout.read(self.W*self.H*3)
        if len(b)==self.W*self.H*3: self.last=np.frombuffer(b,np.uint8).reshape(self.H,self.W,3)
        return self.last
    def close(self):
        try: self.p.stdout.close(); self.p.kill()
        except Exception: pass
# ---------------- 特效
class FX:
    def __init__(self,W,H):
        self.W,self.H=W,H; self.vigc={}
        per=max(2,round(H/270)); row=np.ones(H,np.float32); row[::per]=0.0; self.scan=row[:,None,None]
        rng=np.random.default_rng(3); self.noise=[rng.normal(0,1,(H,W,1)).astype(np.float32) for _ in range(12)]; self.rng=rng
        self.cell=(max(6,round(W/120)),max(10,round(H/45))); cw,ch=self.cell
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',max(10,round(H/45)))
        self.ramp=" .:-=+*#%@"; self.glyphs=[]
        for ch_ in self.ramp:
            im=Image.new('L',(cw,ch),0); ImageDraw.Draw(im).text((0,-1),ch_,fill=255,font=font); self.glyphs.append(np.asarray(im,np.float32)/255)
        self.G=np.stack(self.glyphs); self.ascii_order=rng.random((H//ch+1,W//cw+1)).astype(np.float32)
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
    def block(self,img):
        out=img.copy(); H=img.shape[0]
        for _ in range(self.rng.integers(4,9)):
            y=self.rng.integers(0,H-8); h=self.rng.integers(4,max(6,H//10)); dx=int(self.rng.uniform(-0.08,0.08)*img.shape[1])
            out[y:y+h]=np.roll(img[y:y+h],dx,1)
        return self.rgb_split(out,max(1,img.shape[1]//200))
    def tear(self,img):
        y=self.rng.integers(img.shape[0]//4,img.shape[0]*3//4); dx=int(self.rng.uniform(0.02,0.04)*img.shape[1]*self.rng.choice([-1,1]))
        out=img.copy(); out[:y]=np.roll(img[:y],dx,1)
        out[y:y+2]=np.minimum(255,out[y:y+2]*1.6+40); return out
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
        cw,ch=self.cell; h,w=img.shape[:2]; R,C=h//ch,w//cw
        L=img[:R*ch,:C*cw].mean(2).reshape(R,ch,C,cw).mean((1,3))/255
        idx=np.clip((L*(len(self.ramp)-1)+0.5).astype(int),0,len(self.ramp)-1)
        asc=self.G[idx].transpose(0,2,1,3).reshape(R*ch,C*cw)[...,None]*color[None,None,:]
        mask=(self.ascii_order[:R,:C]<p).astype(np.float32); m=np.repeat(np.repeat(mask,ch,0),cw,1)[...,None]
        out=img.copy(); out[:R*ch,:C*cw]=img[:R*ch,:C*cw]*(1-m)+(asc*0.95+8)*m; return out
# ---------------- 主渲染
def snap_bounds(shots,t0,t1):
    bs=sorted({s['t'] for s in shots}|{s['te'] for s in shots})
    return min(bs,key=lambda x:abs(x-t0)),min(bs,key=lambda x:abs(x-t1))
def render(style,t0,t1,H,out,crf=20,maxrate=None,chunk=10.0,snap=False):
    W=H*16//9; fx=FX(W,H); shots=load_edl()
    if snap: t0,t1=snap_bounds(shots,t0,t1); print('snapped',t0,t1)
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True); tmpdir=out+'.chunks'; os.makedirs(tmpdir,exist_ok=True)
    chunks=[]; c0=t0
    while c0<t1-1e-6:
        c1=min(t1,c0+chunk); fn=os.path.join(tmpdir,f'c_{int(round(c0*1000)):07d}.mp4'); chunks.append(fn)
        if not os.path.exists(fn): render_chunk(style,c0,c1,W,H,fx,shots,fn+'.part.mp4',crf,maxrate); os.replace(fn+'.part.mp4',fn)
        c0=c1
    lst=os.path.join(tmpdir,'list.txt'); open(lst,'w').write(''.join(f"file '{os.path.abspath(f)}'\n" for f in chunks))
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
    useB=(style=='B'); bga=BgaReader(c0,W,H,content_fit=useB) if useB or c0<16 or c1>266 else None
    # 本块内各镜头需要的源帧序号
    plan={}
    for s in shots:
        if s['te']<=c0 or s['t']>=c1: continue
        tau,src=src_map(s); s['_map']=(tau,src)
        fr=[k for k in range(n0,n1) if s['t']<=(k+0.5)*FR<s['te']]
        idx=set()
        for k in fr:
            idx.add(src_index(s,(k+0.5)*FR-s['t']))
        plan[s['n']]=sorted(idx)
    cache={}
    for k in range(n0,n1):
        t=(k+0.5)*FR; s=next((x for x in shots if x['t']<=t<x['te']),None)
        acc=accent_at(t)
        if s is None:
            img=None; wl=0.25
        else:
            tau=t-s['t']; base=s['base']; ev=s['events']
            # 取景窗高度：相邻镜头 win 按镜头中点线性插值
            w=interp_win(shots,s,t)
            if useB: w=min(w,WIN_CAP_B)
            wh=int(round(H*w))//2*2
            key=(s['n'],wh)
            if key not in cache: cache={key:decode_frames(s,wh,plan[s['n']])}
            frames,ww=cache[key]
            fi=src_index(s,tau); img=frames[fi].astype(np.float32)
            # ---- 画面内特效
            if 'contrast' in base: piv=img.mean(); img=piv+(img-piv)*base['contrast']
            if 'dim' in base: img=img*base['dim']
            img=fx.tone(img,0.28 if base.get('tone') else 0.15)
            if 'desat' in base: img=fx.desat(img,base['desat'])
            z=1.0
            if 'beatpulse' in base:
                amp,every=base['beatpulse']; amp*= (0.5 if style=='A' else 1.0)
                bb=BEATS[s['b0']:s['b0']+s['beats']+1:every]; prev=bb[bb<=t+0.012]
                if len(prev): dtb=t-prev[-1]; z=max(z,1+amp*math.exp(-max(dtb,0)/0.12))
            img=fx.zoom(img,z)
            if style!='A':
                for off,typ,nf in ev:
                    if not(off<=tau<off+nf*FR): continue
                    if typ=='rgb': img=fx.rgb_split(img,max(1,int(img.shape[1]*0.006)))
                    elif typ=='block': img=fx.block(img)
                    elif typ=='tear': img=fx.tear(img)
                if 'shake' in base: img=fx.shake(img,base['shake'])
            img=fx.scanlines(img,base.get('scan',0.1))
            if 'noise' in base: img=fx.add_noise(img,base['noise'])
            img=fx.vignette(img,base.get('vignette',0.25))
            if style!='A' and s['ascii'] and tau>=s['dur']-s['ascii']:
                p=(tau-(s['dur']-s['ascii']))/s['ascii']; img=fx.ascii(img,min(1,p*1.05),acc)
            if style!='A' and s['flash'] and tau<3*FR:
                a=0.6*math.exp(-tau/0.05); img=img*(1-a)+255*a
            img=np.clip(img,0,255); wl=img.mean()/255
        accent_now=s is not None and any(typ=='bga_accent' and off<=tau<off+nf*FR for off,typ,nf in s['events'])
        # ---- 背景
        if style=='A':
            if img is not None:
                im=Image.fromarray(img.astype(np.uint8)); zz=max(W/im.size[0],H/im.size[1])*1.15
                im=im.resize((int(im.size[0]*zz)+1,int(im.size[1]*zz)+1),Image.BILINEAR).filter(ImageFilter.GaussianBlur(H/40))
                x=(im.size[0]-W)//2; y=(im.size[1]-H)//2; bg=np.asarray(im.crop((x,y,x+W,y+H)),np.float32)*0.42
            else: bg=np.zeros((H,W,3),np.float32)+BGA_MAIN*0.3
            if bga is not None:
                b=bga.read().astype(np.float32); g=255*0.9/max(30,np.percentile(b,99.5)); b=np.clip(b*g,0,255)
                a=min(1,(16-t)/10) if t<16 else min(1,(t-266)/8) if t>266 else 0
                if a>0: bg=bg*(1-a*0.6)+b*a*0.8
        else:
            b=bga.read().astype(np.float32); g=255*0.9/max(30,np.percentile(b,99.5)); b=np.clip(b*g,0,255)
            j=np.searchsorted(PK,t); last=PK[j-1] if j>0 else -9; pulse=math.exp(-(t-last)/0.12) if t-last<0.4 else 0
            if accent_now: pulse=max(pulse,1.0)+0.5
            b=b*(1+0.4*pulse); target=min(0.30,0.85*wl); base_t=BGA_MAIN*(target*255/BGA_MAIN.mean())*0.85
            bg=np.clip(base_t+b*0.7,0,255)
            if bg.mean()/255>wl*0.95: bg=bg*(wl*0.95*255/bg.mean())
        out=bg.copy()
        if img is not None:
            wh_,ww_=img.shape[:2]; x=(W-ww_)//2; y=(H-wh_)//2; out[y:y+wh_,x:x+ww_]=img
            bar=max(3,int(round(wh_*BAR))); out[y:y+bar,x:x+ww_]=BGA_MAIN*0.6
            if style=='B':
                lw=max(1,H//540); col=acc*(1.6 if accent_now else 1.0)
                out[y+bar-lw:y+bar,x:x+ww_]=col; out[y+wh_-lw:y+wh_,x:x+ww_]=col; out[y:y+wh_,x:x+lw]=col; out[y:y+wh_,x+ww_-lw:x+ww_]=col
            else: out[y+bar-1:y+bar,x:x+ww_]=acc*0.5
        if t<1.5: out*=t/1.5
        if t>T-2.68: out*=max(0,(T-t)/2.68)
        ep.stdin.write(np.clip(out,0,255).astype(np.uint8).tobytes())
    ep.stdin.close(); ep.wait()
    if bga: bga.close()
def src_index(s,tau):
    tau_eff=tau
    for off,typ,nf in s['events']:
        if typ=='stutter' and off<=tau<off+nf*FR: tau_eff=off; break
    tg,src=s['_map']; st=float(np.interp(tau_eff,tg,src)); sfps=float(CLIPS[s['clip']]['fps'])
    return max(0,int(st*sfps+1e-6))
def interp_win(shots,s,t):
    i=shots.index(s); w0=s['win']; mid=s['t']+s['dur']/2
    if t<mid and i>0 and shots[i-1]['te']>=s['t']-0.01: p=shots[i-1]; m2=p['t']+p['dur']/2; return w0+(p['win']-w0)*(mid-t)/(mid-m2)
    if t>=mid and i+1<len(shots) and shots[i+1]['t']<=s['te']+0.01: p=shots[i+1]; m2=p['t']+p['dur']/2; return w0+(p['win']-w0)*(t-mid)/(m2-mid)
    return w0
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--style',default='B'); ap.add_argument('--t0',type=float,default=0); ap.add_argument('--t1',type=float,default=T)
    ap.add_argument('--height',type=int,default=540); ap.add_argument('--out',required=True); ap.add_argument('--crf',type=int,default=20); ap.add_argument('--maxrate',default=None)
    ap.add_argument('--chunk',type=float,default=10.0); ap.add_argument('--snap',action='store_true')
    a=ap.parse_args(); print(render(a.style,a.t0,a.t1,a.height,a.out,a.crf,a.maxrate,a.chunk,a.snap))
