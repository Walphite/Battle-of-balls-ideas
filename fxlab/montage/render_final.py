"""终版渲染（分段、分块 10s、可断点续跑）。
 python3 -I render_final.py init                    计算段界（EDL 镜头边界，最近 93 s / 186 s）并生成音频
 python3 -I render_final.py chunks RES PART         渲染 RES(540/720/1080) 的 PART(1/2/3) 里还没完成的块
 python3 -I render_final.py status RES
EDL=work/edl_v9.csv；舞台：画面不提前，音频整体延迟 25ms 封装（= 画面相对音乐提前 25ms）。"""
import os,sys,json,csv,subprocess
os.environ.setdefault('EDL','work/edl_v9.csv')
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import render_c as RC, render_v4 as R4
FPS=30; NF=8364; CH=300; LEAD_MS=25
SEGF='work/final/segments.json'
def init():
    os.makedirs('work/final',exist_ok=True)
    R=list(csv.DictReader(open(os.environ['EDL'],encoding='utf-8'))); B=sorted({float(r['t_start']) for r in R}|{float(r['t_end']) for r in R})
    cut=[min(B,key=lambda x:abs(x-t)) for t in (93.0,186.0)]; f=[0]+[int(round(t*FPS)) for t in cut]+[NF]
    json.dump(dict(frames=f,times=[x/FPS for x in f]),open(SEGF,'w')); print('段界 帧',f,'秒',[round(x/FPS,3) for x in f])
    # 音频：mix.wav（静态增益+限幅）整体延迟 25ms；编码一次（AAC 256k，关 PNS/强度立体声）
    subprocess.run(['ffmpeg','-v','error','-y','-i','work/audio/mix.wav','-af',f'adelay={LEAD_MS}:all=1','-t','278.8','-ar','48000','-ac','2','-c:a','pcm_s16le','work/audio/mix_lead25.wav'],check=True)
    subprocess.run(['ffmpeg','-v','error','-y','-i','work/audio/mix_lead25.wav','-c:a','aac','-b:a','256k','-aac_pns','0','-aac_is','0','work/audio/final_audio.m4a'],check=True)
def seg(part): f=json.load(open(SEGF))['frames']; return f[part-1],f[part]
def cpath(res,part,c): return f'work/final/{res}/p{part}/c_{c:05d}.mp4'
def chunks(res,part):
    a,b=seg(part); H=int(res); shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v5.json'))['events']; comp=RC.Composer(H*16//9,H,sched)
    os.makedirs(f'work/final/{res}/p{part}',exist_ok=True)
    rng=list(range(a,b,CH)); rng=rng[::-1] if os.environ.get('REV')=='1' else rng
    for c0 in rng:
        c1=min(b,c0+CH); fn=cpath(res,part,c0)
        if os.path.exists(fn): continue
        comp.meta=[]
        if H<=540: RC.render_chunk(comp,c0/FPS,c1/FPS,fn+f'.part{os.getpid()}.mp4',22,'2200k',shots)
        else: RC.render_chunk(comp,c0/FPS,c1/FPS,fn+f'.part{os.getpid()}.mp4',14,None,shots)
        json.dump(comp.meta,open(fn+'.meta.json','w'),default=float); os.replace(fn+f'.part{os.getpid()}.mp4',fn); print('chunk',res,part,c0,c1,flush=True)
def status(res):
    for p in (1,2,3):
        a,b=seg(p); n=len(range(a,b,CH)); d=sum(1 for c in range(a,b,CH) if os.path.exists(cpath(res,p,c))); print(res,'part',p,d,'/',n)
if __name__=='__main__':
    m=sys.argv[1]
    if m=='init': init()
    elif m=='chunks': chunks(sys.argv[2],int(sys.argv[3]))
    elif m=='status': status(sys.argv[2])
