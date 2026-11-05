"""项目一验证：每 5 s（80–125）底行文字可见比例；输出对照图 fxlab/docs/bottom_text_compare_v10.png"""
import os,sys,json,subprocess
os.environ['EDL']='work/edl_v10.csv'; os.environ.setdefault('PRESET','veryfast')
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import numpy as np, render_c as RC, render_v4 as R4
from PIL import Image
def one(t):
    shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v10.json'))['events']
    comp=RC.Composer(960,540,sched); k=int(round(t*30-0.5)); c0=k/30; fn=f'work/v10/b_{k}.mp4'
    RC.render_chunk(comp,c0,c0+1/30,fn,18,None,shots)
    png=f'work/v10/b_{k}.png'; subprocess.run(['ffmpeg','-v','error','-y','-i',fn,'-frames:v','1',png],check=True)
    out=np.asarray(Image.open(png).convert('RGB')).astype(np.float32)
    bga=R4.BgaReader(c0,960,540,content_fit=True); b0,ink=RC.clean_bga(bga.read().astype(np.float32)); bga.close()
    ink=np.asarray(ink); ink=ink if ink.ndim==2 else ink.max(2)
    rows=np.where((ink>0.3).sum(1)>2)[0]; y1=rows.max(); y0=y1
    while y0-1 in set(rows): y0-=1          # 最后一条文字带
    band=ink[y0:y1+1]>0.3; lum=out@np.array([0.299,0.587,0.114],np.float32)
    ob=lum[y0:y1+1]; ref=(b0.max(2) if b0.ndim==3 else b0)[y0:y1+1]
    vis=float(((ob>=0.5*np.minimum(ref,255)*0.85)&band).sum()/max(1,band.sum())) if band.sum()>0 else None
    return dict(t=t,band_y=[int(y0),int(y1)],margin_px=int(540-1-y1),visible=None if vis is None else round(vis,3),px=int(band.sum()))
if __name__=='__main__':
    from multiprocessing import Pool
    ts=[80+5*i+0.5 for i in range(10)]
    with Pool(4) as p: R=p.map(one,ts)
    for r in R: print(r)
    json.dump(R,open('work/v10/bottom_check.json','w'))
