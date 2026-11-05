import os,sys,json,subprocess
os.environ['EDL']='work/edl_v10.csv'; os.environ.setdefault('PRESET','veryfast')
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import numpy as np, render_c as RC, render_v4 as R4
def one(a):
    t,on=a; RC.PBS_ON=bool(on)
    shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v10.json'))['events']
    comp=RC.Composer(960,540,sched); k=int(round(t*30-0.5)); c0=k/30; fn=f'work/v10/p{on}_{k}.mp4'
    RC.render_chunk(comp,c0,c0+1/30,fn,18,None,shots); subprocess.run(['ffmpeg','-v','error','-y','-i',fn,'-frames:v','1',f'work/v10/p{on}_{t}.png'],check=True)
    return dict(t=t,on=on,pbs=comp.meta[0].get('pbs'))
if __name__=='__main__':
    from multiprocessing import Pool
    ts=[192.5+4*i for i in range(9)]+[]
    with Pool(4) as p: R=p.map(one,[(t,1) for t in ts]+[(t,0) for t in (196.5,208.5,220.5)])
    json.dump(R,open('work/v10/pbs_check.json','w'))
    for r in R: print(r)
