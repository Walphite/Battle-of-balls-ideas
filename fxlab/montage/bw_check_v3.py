"""V3 黑白自检：每个脉冲谷底帧的 亮度（彩色 vs 黑白）与 饱和度；以及全片解析 S(t) 曲线指标。输出 work/v3/bw_check.json"""
import os,sys,json,subprocess
os.environ['EDL']='work/edl_v10.csv'; os.environ.setdefault('PRESET','veryfast')
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import numpy as np
from PIL import Image
def frame(a):
    t,nobw,ink=a; os.environ['NOBW']='1' if nobw else '0'
    import render_c as RC, render_v4 as R4
    shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v10.json'))['events']; comp=RC.Composer(960,540,sched,ink=ink)
    k=int(round(t*30-0.5)); fn=f'work/v3/f{nobw}{int(ink)}_{k}.mp4'; os.makedirs('work/v3',exist_ok=True)
    RC.render_chunk(comp,k/30,k/30+1/30,fn,16,None,shots); png=fn[:-4]+'.png'; subprocess.run(['ffmpeg','-v','error','-y','-i',fn,'-frames:v','1',png],check=True)
    a=np.asarray(Image.open(png).convert('RGB')).astype(np.float32)/255; c=a[135:405,240:720]; mx=c.max(2); mn=c.min(2); sat=float(((mx-mn)/np.maximum(mx,1e-3)).mean())
    return dict(t=t,nobw=nobw,lum=float(a.mean()),y709=float((a@np.array([0.2126,0.7152,0.0722])).mean()),csat=sat,meta=comp.meta[0]['lum'])
if __name__=='__main__':
    from multiprocessing import Pool
    P=json.load(open('work/bw_pulses.json'))['pulses']; jobs=[]
    for p in P:
        t=p['t_peak']+min(0.3,(p['hold_s'] or 0.3)/2); jobs+=[(t,0,1.0),(t,1,1.0),(t,0,0.0),(t,1,0.0)]
    with Pool(4) as pl: R=pl.map(frame,jobs)
    out=[]
    for i,p in enumerate(P):
        bw,col,gbw,gcol=R[4*i:4*i+4]; out.append(dict(pulse=p['id'],t=bw['t'],game_sat_bw=round(gbw['csat'],4),game_sat_color=round(gcol['csat'],4),mean_rgb_change=round((bw['lum']-col['lum'])/max(col['lum'],1e-3),4),Y709_change=round((bw['y709']-col['y709'])/max(col['y709'],1e-3),4),game_Y709_change=round((gbw['y709']-gcol['y709'])/max(gcol['y709'],1e-3),4)))
        print(out[-1])
    json.dump(out,open('work/v3/bw_check.json','w'))
