import os,sys,json,subprocess,time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
p=int(sys.argv[1]); step=sys.argv[2]; f=json.load(open('work/final3/segments.json'))['frames']; a,b=f[p-1],f[p]
d='work/final3/1080'; lst=f'{d}/list_p{p}.txt'
open(lst,'w').write(''.join(f"file '{os.path.abspath(f'{d}/p{p}/c_{c:05d}.mp4')}'\n" for c in range(a,b,60)))
VB='7200k'; log=f'{d}/x264_p{p}'
base=['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',lst,'-c:v','libx264','-preset','fast','-b:v',VB,'-maxrate','10000k','-bufsize','14000k','-pix_fmt','yuv420p','-g','60','-passlogfile',log]
if step=='1': subprocess.run(base+['-pass','1','-an','-f','null','/dev/null'],check=True)
else:
    out=f'../deliver/final_v3/final_1080p_v3_part{p}.mp4'; tmp=f'{d}/v{p}.mp4'
    subprocess.run(base+['-pass','2','-an',tmp],check=True)
    t0,t1=a/30,b/30
    subprocess.run(['ffmpeg','-v','error','-y','-i',tmp,'-ss',f'{t0:.4f}','-t',f'{t1-t0:.4f}','-i','work/audio/final_audio.m4a','-map','0:v','-map','1:a','-c','copy','-movflags','+faststart',out],check=True)
    print(out,os.path.getsize(out)/1e6,'MB')
