import os,json,subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
f=json.load(open('work/final3/segments.json'))['frames']; lst=[]
for p in (1,2,3):
    for c in range(f[p-1],f[p],300): lst.append(os.path.abspath(f'work/final3/540/p{p}/c_{c:05d}.mp4'))
open('work/final3/list540.txt','w').write(''.join(f"file '{x}'\n" for x in lst))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i','work/final3/list540.txt','-c','copy','work/final3/video540.mp4'],check=True)
subprocess.run(['ffmpeg','-v','error','-y','-i','work/final3/video540.mp4','-i','work/audio/final_audio.m4a','-map','0:v','-map','1:a','-c','copy','-t','278.8','-movflags','+faststart','../deliver/final_v3/final_540p_v3.mp4'],check=True)
print(os.path.getsize('../deliver/final_v3/final_540p_v3.mp4')/1e6,'MB')
