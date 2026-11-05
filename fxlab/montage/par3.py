import os,sys,json
os.environ['EDL']='work/edl_v10.csv'; os.environ['PRESET']='veryfast'; os.environ['CH']='60'
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import render_final2 as RF
from multiprocessing import Pool
def go(i):
    a,b=RF.seg(3); todo=[c for c in range(a,b,60) if not os.path.exists(RF.cpath('1080',3,c))]
    mine=todo[i::4]
    import render_c as RC, render_v4 as R4
    shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v10.json'))['events']; comp=RC.Composer(1920,1080,sched)
    for c0 in mine:
        c1=min(b,c0+60); fn=RF.cpath('1080',3,c0); comp.meta=[]
        RC.render_chunk(comp,c0/30,c1/30,fn+f'.part{os.getpid()}.mp4',14,None,shots)
        json.dump(comp.meta,open(fn+'.meta.json','w'),default=float); os.replace(fn+f'.part{os.getpid()}.mp4',fn)
if __name__=='__main__':
    with Pool(4) as p: p.map(go,range(4))
