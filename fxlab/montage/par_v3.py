import os,sys,json
os.environ['EDL']='work/edl_v10.csv'; os.environ['PRESET']='veryfast'
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import render_final3 as RF
from multiprocessing import Pool
def todo():
    f=json.load(open('work/final3/segments.json'))['frames']
    return [(p,c,min(f[p],c+60)) for p in (1,2,3) for c in range(f[p-1],f[p],60) if not os.path.exists(RF.cpath('1080',p,c))]
def go(i):
    import render_c as RC, render_v4 as R4
    shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v10.json'))['events']; comp=RC.Composer(1920,1080,sched)
    for p,c0,c1 in todo()[i::4]:
        fn=RF.cpath('1080',p,c0); comp.meta=[]
        RC.render_chunk(comp,c0/30,c1/30,fn+f'.part{os.getpid()}.mp4',14,None,shots)
        json.dump(comp.meta,open(fn+'.meta.json','w'),default=float); os.replace(fn+f'.part{os.getpid()}.mp4',fn)
if __name__=='__main__':
    with Pool(4) as pl: pl.map(go,range(4))
