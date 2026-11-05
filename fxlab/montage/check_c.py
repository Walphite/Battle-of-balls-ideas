"""版本 C 合成层自检：边缘跳变/亮线、窗内外亮度比、BGA 区亮度、溶解 1 s 内亮度跳变、触发点时间差。"""
import os,sys,json,subprocess,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import render_v4 as R4
W,H=960,540
def read(fn,ts=None,scale=None,fps=None):
    vf=[]; 
    if fps: vf.append(f'fps={fps}')
    if scale: vf.append(f'scale={scale[0]}:{scale[1]}')
    if ts is not None:
        out=[]
        for t in ts:
            p=subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',fn,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
            out.append(np.frombuffer(p,np.uint8).reshape(H,W,3).astype(np.float32))
        return out
    p=subprocess.run(['ffmpeg','-v','error','-i',fn,'-vf',','.join(vf),'-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    h=scale[1]; return np.frombuffer(p,np.uint8).reshape(-1,h,scale[0],3).astype(np.float32)
lum=lambda f:f@np.array([0.299,0.587,0.114],np.float32)
def old_geo(t,shots):
    s=next(x for x in shots if x['t']<=t<x['te']); w=min(R4.interp_win(shots,s,t),0.88); wh=int(round(H*w))//2*2; ww=int(round(wh*R4.ASPECT))//2*2
    return dict(x=(W-ww)//2,y=(H-wh)//2,ww=ww,wh=wh,fl=12)
def edge_metrics(f,g):
    L=lum(f); x,y,ww,wh=g['x'],g['y'],g['ww'],g['wh']; r0,r1=int(y+0.2*wh),int(y+0.8*wh); prof=L[r0:r1].mean(0); res=[]
    for ex in (x,x+ww):
        if ex<8 or ex>W-8: continue
        a,b=max(4,ex-60),min(W-4,ex+60); p=prof[a:b]
        jump=float(np.abs(np.diff(p)).max()); spike=float(max(p[i]-max(p[max(0,i-3)],p[min(len(p)-1,i+3)]) for i in range(len(p))))
        res.append((jump,spike))
    return res
def region_means(f,g,margin):
    L=lum(f)/255; x,y,ww,wh=g['x'],g['y'],g['ww'],g['wh']
    win=L[int(y+0.2*wh):int(y+0.8*wh),int(x+0.2*ww):int(x+0.8*ww)].mean()
    m=np.ones((H,W),bool); m[max(0,int(y-margin)):int(y+wh+margin),max(0,int(x-margin)):int(x+ww+margin)]=False
    return float(win),float(L[m].mean()) if m.sum()>200 else float('nan'),int(m.sum())
def analyze(name,fn,geo_fn,margin_fn,t0):
    ts=[0.5,2.5,4.5,6.5,8.5,10.5]; F=read(fn,ts); out=[]
    for t,f in zip(ts,F):
        g=geo_fn(t0+t,t); em=edge_metrics(f,g); win,bg,n=region_means(f,g,margin_fn(g)); out.append(dict(t=t,jump=max(e[0] for e in em) if em else None,spike=max(e[1] for e in em) if em else None,win=win,bga=bg,ratio=win/bg if bg==bg and bg>0 else None))
    return out
if __name__=='__main__':
    shots=R4.load_edl(); res={}
    t0=139.039
    old=analyze('old','work/preview/sample_C_540.mp4',lambda T,t:old_geo(T,shots),lambda g:20,t0)
    for nm in ('C1','C2'):
        meta=json.load(open(f'work/preview/sample_{nm}_540.mp4.meta.json'))['frames']
        def geo(T,t,meta=meta): m=min(meta,key=lambda e:abs(e['t']-T)); return dict(x=m['x'],y=m['y'],ww=m['ww'],wh=m['wh'],fl=m['fl'])
        res[nm]=analyze(nm,f'work/preview/sample_{nm}_540.mp4',geo,lambda g:g['fl']+12,t0)
    res['old']=old
    # 全帧均值：BGA 区亮度、窗内外比（每 3 帧一帧，480×270）
    for nm,fn in (('old','work/preview/sample_C_540.mp4'),('C1','work/preview/sample_C1_540.mp4'),('C2','work/preview/sample_C2_540.mp4')):
        V=read(fn,scale=(480,270),fps=10); wins=[];bgs=[]
        meta=None if nm=='old' else json.load(open(f'work/preview/sample_{nm}_540.mp4.meta.json'))['frames']
        for i,f in enumerate(V):
            T=t0+i/10.0+0.05; g=old_geo(T,shots) if nm=='old' else (lambda m:dict(x=m['x'],y=m['y'],ww=m['ww'],wh=m['wh'],fl=m['fl']))(min(meta,key=lambda e:abs(e['t']-T)))
            g2=dict(x=g['x']/2,y=g['y']/2,ww=g['ww']/2,wh=g['wh']/2); Lf=lum(f)/255
            win=Lf[int(g2['y']+0.2*g2['wh']):int(g2['y']+0.8*g2['wh']),int(g2['x']+0.2*g2['ww']):int(g2['x']+0.8*g2['ww'])].mean()
            mk=np.ones((270,480),bool); mg=(g['fl']+12)/2 if nm!='old' else 10; mk[max(0,int(g2['y']-mg)):int(g2['y']+g2['wh']+mg),max(0,int(g2['x']-mg)):int(g2['x']+g2['ww']+mg)]=False
            if mk.sum()>150: wins.append(win); bgs.append(Lf[mk].mean())
        res[nm+'_all']=dict(win=float(np.mean(wins)),bga=float(np.mean(bgs)),ratio=float(np.mean(np.array(wins)/np.array(bgs))),ratio_max=float(np.max(np.array(wins)/np.array(bgs))),n=len(wins))
    for k in ('old','C1','C2'):
        r=res[k]; j=[e['jump'] for e in r if e['jump'] is not None]; sp=[e['spike'] for e in r if e['spike'] is not None]
        print(k,'edge max jump per-frame',[round(x) for x in j],'max',round(max(j)),'| max spike (1px bright line)',round(max(sp)),'| bga mean',round(res[k+'_all']['bga'],3),'win mean',round(res[k+'_all']['win'],3),'ratio mean/max',round(res[k+'_all']['ratio'],2),round(res[k+'_all']['ratio_max'],2))
    # 溶解 1 s 内亮度大幅跳变（帧均亮度 30fps 相邻差 ≥0.05）
    def jumps(fn):
        V=read(fn,scale=(160,90)); m=lum(V).mean((1,2))/255; d=np.abs(np.diff(m)); idx=np.where(d>=0.05)[0]
        return int(max([((idx>=i)&(idx<i+30)).sum() for i in idx],default=0)),int(len(idx))
    for nm in ('C','C1','C2','D'):
        fn=f'work/preview/sample_{nm}_540.mp4'
        if os.path.exists(fn): res['jumps_'+nm]=jumps(fn); print('1s 内亮度大幅跳变次数 最大/总数',nm,res['jumps_'+nm])
    json.dump(res,open('work/check_c.json','w'),indent=0)
