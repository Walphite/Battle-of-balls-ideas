"""第 4 步：全片 540p 预览，分段、分块（10 s）可断点续跑。
 python3 -I render_full.py A|B|C     渲染该段的剩余块；该段全部完成后自动封装 preview_X.mp4
 python3 -I render_full.py audio     生成全片音频母带（淡入/淡出 + 两遍 loudnorm）"""
import os,sys,json,subprocess,glob
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import render_c as RC, render_v4 as R4
FPS=30; NF=8364                         # 全片 278.8 s = 8364 帧（配乐 278.806 s）
SEG={'A':(0,2976),'B':(2976,6254),'C':(6254,NF)}      # 段界 = EDL 镜头边界：99.21 s、208.48 s
H=540; W=960; CH=300                    # 块 = 300 帧 = 10 s
DELIV='../deliver/preview'; os.makedirs(DELIV,exist_ok=True); os.makedirs('work/full',exist_ok=True)
def audio_master():
    out='work/full/audio_master.wav'
    if os.path.exists(out): return out
    af='afade=t=in:st=0:d=1.5,afade=t=out:st=276.126:d=2.68'
    m=subprocess.run(['ffmpeg','-hide_banner','-i',R4.MUS,'-af',af+',loudnorm=I=-10.5:TP=-3.0:LRA=11:print_format=json','-f','null','-'],capture_output=True,text=True).stderr
    j=json.loads(m[m.rindex('{'):]); 
    ln=f"loudnorm=I=-10.5:TP=-3.0:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true"
    subprocess.run(['ffmpeg','-v','error','-y','-i',R4.MUS,'-af',af+','+ln,'-ar','48000','-ac','2',out],check=True); return out
def seg_chunks(name):
    a,b=SEG[name]; return [(c,min(b,c+CH)) for c in range(a,b,CH)]
def chunk_path(name,c): return f'work/full/{name}/c_{c:05d}.mp4'
def render_seg(name):
    shots=R4.load_edl(); sched=json.load(open('work/glyph_schedule_v5.json'))['events']; comp=RC.Composer(W,H,sched); os.makedirs(f'work/full/{name}',exist_ok=True)
    for c0,c1 in seg_chunks(name):
        fn=chunk_path(name,c0)
        if os.path.exists(fn): continue
        comp.meta=[]; RC.render_chunk(comp,c0/FPS,c1/FPS,fn+'.part.mp4',22,'2300k',shots)
        json.dump(comp.meta,open(fn+'.meta.json','w'),default=float); os.replace(fn+'.part.mp4',fn); print('chunk',name,c0,c1,flush=True)
def done(name): return all(os.path.exists(chunk_path(name,c0)) for c0,c1 in seg_chunks(name))
def mux(name):
    a,b=SEG[name]; d=f'work/full/{name}'; lst=f'{d}/list.txt'; open(lst,'w').write(''.join(f"file '{os.path.abspath(chunk_path(name,c0))}'\n" for c0,c1 in seg_chunks(name)))
    v=f'{d}/video.mp4'; subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',lst,'-c','copy',v],check=True)
    m=audio_master(); out=f'{DELIV}/preview_{name}.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',v,'-ss',f'{a/FPS:.6f}','-t',f'{(b-a)/FPS:.6f}','-i',m,'-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','256k','-movflags','+faststart',out],check=True)
    print('muxed',out,os.path.getsize(out)/1e6,'MB'); return out
if __name__=='__main__':
    n=sys.argv[1]
    if n=='audio': print(audio_master())
    else:
        render_seg(n)
        if done(n) and not os.path.exists(f'{DELIV}/preview_{n}.mp4'): mux(n)
