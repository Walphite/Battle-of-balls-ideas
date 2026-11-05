"""第 4 项验收（i）：8 个时刻×6 个单元（3 暗背景 + 3 亮背景/球体）。输出 JSON/表。"""
import os,sys,json,subprocess,colorsys
os.environ['EDL']='work/edl_v10.csv'; os.environ.setdefault('PRESET','veryfast')
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import numpy as np, render_c as RC, render_v4 as R4
from PIL import Image,ImageFilter
SC=0.75
def rl(c):
    c=np.asarray(c,np.float64)/255.0; c=np.where(c<=0.03928,c/12.92,((c+0.055)/1.055)**2.4); return 0.2126*c[...,0]+0.7152*c[...,1]+0.0722*c[...,2]
def hls(c): return colorsys.rgb_to_hls(*[v/255 for v in c])
def render(t,ink):
    shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v10.json'))['events']; comp=RC.Composer(960,540,sched,ink=ink); k=int(round(t*30-0.5)); c0=k/30
    fn=f'work/v10/c_{ink}_{k}.mp4'; RC.render_chunk(comp,c0,c0+1/30,fn,16,None,shots); png=fn[:-4]+'.png'
    subprocess.run(['ffmpeg','-v','error','-y','-i',fn,'-frames:v','1',png],check=True)
    bga=R4.BgaReader(c0,960,540,content_fit=True); b0,key=RC.clean_bga(bga.read().astype(np.float32)); bga.close()
    return np.asarray(Image.open(png).convert('RGB')).astype(np.float32),b0,key,comp.meta[0].get('pbs')
def one(t):
    out,b0,key,meta=render(t,1.0); bg,_,_,_=render(t,0.0)
    bgb=np.asarray(Image.fromarray(bg.astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))).astype(np.float32)
    cells=[]
    for r in range(4):
        for c in range(6):
            x0=int((128+133.5*c)*SC); x1=int((222+133.5*c)*SC); y0=int((74+118*r-30)*SC); y1=int((74+118*r+30)*SC); cells.append(('PBS r%dc%d'%(r+1,c+1),x0,x1,y0,y1))
    cells+= [('底部三行',int(128*SC),int(625*SC),int(505*SC),int(585*SC)),('Sending行',int(128*SC),int(330*SC),int(622*SC),int(642*SC))]
    rows=[]
    for n,x0,x1,y0,y1 in cells:
        g=key[y0:y1,x0:x1]>0.8
        if g.sum()<40: continue
        po=out[y0:y1,x0:x1][g]; pb=bgb[y0:y1,x0:x1][g]; po0=b0[y0:y1,x0:x1][g]
        Lb=float(np.median(rl(pb)));  bl=float(np.median(hls(np.median(pb,0))[1]))
        # 字像素：取最「像字」的一半（与背景亮度差最大），避开抗锯齿边缘
        d=np.abs(rl(po)-rl(pb)); sel=d>=np.percentile(d,50); c=(rl(po)[sel]+0.05)/(rl(pb)[sel]+0.05); c=np.maximum(c,1/c)
        cmed=float(np.median(c)); pc=np.median(po[sel],0); oc=np.median(po0,0); hp,lp,sp=hls(pc); ho,lo,so=hls(oc)
        dh=min(abs(hp-ho),1-abs(hp-ho))*360 if so>0.2 and sp>0.1 else None
        near=float((np.array([hls(p)[1] for p in po[::max(1,len(po)//200)]])<0.15).mean())
        rows.append(dict(cell=n,bg_L=round(bl,2),kind='球体/亮' if bl>0.35 else '暗',contrast=round(cmed,2),dhue=None if dh is None else round(dh,1),S_ratio=round(sp/max(so,1e-3),2) if so>0.2 else None,near_black=round(near,3),px=int(g.sum())))
    dark=sorted([r for r in rows if r['kind']=='暗'],key=lambda r:-r['px'])[:3]; br=sorted([r for r in rows if r['kind']!='暗'],key=lambda r:-r['bg_L'])[:3]
    return dict(t=t,cells=dark+br,meta=meta)
if __name__=='__main__':
    from multiprocessing import Pool
    ts=[192.5,196.5,200.5,208.5,212.5,216.5,220.5,224.5]
    with Pool(4) as p: R=p.map(one,ts)
    json.dump(R,open(sys.argv[1] if len(sys.argv)>1 else 'work/v10/pbs_accept.json','w'),ensure_ascii=False,default=float)
    ok=True
    for m in R:
        for r in m['cells']: print(m['t'],r)
        print(m['t'],'cov',m['meta'])
