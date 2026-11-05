import os,sys,subprocess,json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
res=sys.argv[1]; to=sys.argv[2] if len(sys.argv)>2 else '585'
env=dict(os.environ,CH='60' if res=='1080' else os.environ.get('CH','300'),PRESET='veryfast' if res=='1080' else 'medium')
f=json.load(open('work/final2/segments.json'))['frames']; CH=int(env['CH'])
rem={p:sum(1 for c in range(f[p-1],f[p],CH) if not os.path.exists(f'work/final2/{res}/p{p}/c_{c:05d}.mp4')) for p in (1,2,3)}
jobs=[(p,'0') for p in (1,2,3) if rem[p]>0]
if rem:
    big=max(rem,key=rem.get)
    if rem[big]>2: jobs.append((big,'1'))
    if len([j for j in jobs if j[0]==big])>0 and rem[big]>8 and len(jobs)<4: jobs.append((big,'1'))
ps=[subprocess.Popen(['timeout',to,sys.executable,'-I','render_final2.py','chunks',res,str(p)],env=dict(env,REV=r),stdout=open(f'work/final2/log{res}_{p}{"r" if r=="1" else ""}.txt','a'),stderr=subprocess.STDOUT) for p,r in jobs[:4]]
[p.wait() for p in ps]
print('剩余(启动前)',rem)
