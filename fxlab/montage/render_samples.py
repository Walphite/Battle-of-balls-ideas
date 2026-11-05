"""V10 四个短样（540p，配乐+BGA，画面领先音频 25ms）。 python3 -I render_samples.py S1|S2|S3|S4"""
import os,sys,json,subprocess
os.environ['EDL']='work/edl_v10.csv'; os.environ.setdefault('PRESET','veryfast')
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import render_c as RC, render_v4 as R4
SM=dict(S1=(80,95),S2=(123,140),S3=(192,206),S4=(10.2,13.0),PBS=(192,206),BWa=(120,135),BWb=(135,150),BWc=(195,210),BWd=(210,225))
def run(name):
    t0,t1=SM[name]; sched=json.load(open('work/glyph_schedule_v10.json'))['events']
    if name=='S4':      # 演示：真实相邻镜头 4→5 的边界 M=11.7333，3+3 个八分音符
        M=11.7333; e=(11.7333-8.7)/18; sched=[dict(kind='f',variant='交叉溶解',A0=M-3*e,M=M,A1=M+3*e,n_before=3,n_after=3,shot=4,t=M-3*e,dur=6*e,pmax=1.0)]
    comp=RC.Composer(960,540,sched); shots=R4.load_edl(); os.makedirs('work/v10',exist_ok=True); fn=f'work/v10/{name}_v.mp4'
    RC.render_chunk(comp,t0,t1,fn,22,'2200k',shots)
    json.dump(comp.meta,open(f'work/v10/{name}.meta.json','w'),default=float)
    out=f'work/v10/{name}.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',fn,'-ss',str(t0),'-t',str(t1-t0),'-i','work/audio/mix_lead25.wav','-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','256k','-aac_pns','0','-aac_is','0','-shortest',out],check=True)
    print('done',name,flush=True)
if __name__=='__main__':
    from multiprocessing import Pool
    names=sys.argv[1:]
    with Pool(len(names)) as p: p.map(run,names)
