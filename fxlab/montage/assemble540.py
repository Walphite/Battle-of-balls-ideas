import os,json,subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
f=json.load(open('work/final/segments.json'))['frames']; lst=[]
for p in (1,2,3):
    for c in range(f[p-1],f[p],300): lst.append(os.path.abspath(f'work/final/540/p{p}/c_{c:05d}.mp4'))
open('work/final/list540.txt','w').write(''.join(f"file '{x}'\n" for x in lst))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i','work/final/list540.txt','-c','copy','work/final/video540.mp4'],check=True)
subprocess.run(['ffmpeg','-v','error','-y','-i','work/final/video540.mp4','-i','work/audio/final_audio.m4a','-map','0:v','-map','1:a','-c','copy','-t','278.8','-movflags','+faststart','../deliver/final/final_540p.mp4'],check=True)
print(os.path.getsize('../deliver/final/final_540p.mp4')/1e6,'MB')
V='work/final/video540.mp4'; out=[]; os.makedirs('work/final/cmp',exist_ok=True)
for si,(a,b) in enumerate([(140,152),(209,221)]):
    for L in (0,25,45):
        fn=f'work/final/cmp/s{si}_L{L}.mp4'
        vf=f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{L}':x=24:y=18:fontsize=64:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=10:enable='between(t,0,1)'"
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(a),'-t',str(b-a),'-i',V,'-ss',str(a),'-t',str(b-a),'-i','work/audio/mix.wav','-vf',vf,'-af',f'adelay={L}:all=1','-map','0:v','-map','1:a','-c:v','libx264','-crf','22','-preset','medium','-maxrate','2500k','-bufsize','5000k','-pix_fmt','yuv420p','-c:a','aac','-b:a','256k','-aac_pns','0','-aac_is','0','-t',str(b-a),fn],check=True); out.append(fn)
open('work/final/cmp/list.txt','w').write(''.join(f"file '{os.path.abspath(x)}'\n" for x in out))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i','work/final/cmp/list.txt','-c','copy','-movflags','+faststart','../deliver/final/lead_compare_540p.mp4'],check=True); print(os.path.getsize('../deliver/final/lead_compare_540p.mp4')/1e6,'MB')
