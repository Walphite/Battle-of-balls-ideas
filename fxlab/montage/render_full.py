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
    """连续 mix.wav（0–278.8 s）：原曲 + 淡入淡出 + 静态增益 −3.4 dB + 限幅器（仅防真峰值，lookahead 7ms，release 150ms）。
    不再使用 loudnorm（原曲真峰值 +2.46 dBTP，目标 TP 下 linear 模式无法成立，回退为动态模式，77.5–101 s 被限幅器压到 −42 dB）。"""
    return 'work/audio/mix.wav'
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
    subprocess.run(['ffmpeg','-v','error','-y','-i',v,'-ss',f'{a/FPS:.6f}','-t',f'{(b-a)/FPS:.6f}','-i',m,'-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','256k','-aac_pns','0','-aac_is','0','-movflags','+faststart',out],check=True)
    print('muxed',out,os.path.getsize(out)/1e6,'MB'); return out
if __name__=='__main__':
    n=sys.argv[1]
    if n=='audio': print(audio_master())
    else:
        render_seg(n)
        if done(n) and not os.path.exists(f'{DELIV}/preview_{n}.mp4'): mux(n)
