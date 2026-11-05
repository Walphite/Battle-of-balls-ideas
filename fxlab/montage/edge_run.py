import json,csv,subprocess,sys
R=list(csv.DictReader(open('work/edl.csv',encoding='utf-8')))
sel=[('红色画面',20),('深蓝画面',23),('亮白画面',80),('暗色/紫蓝',22),('暗红(收束,窗77%)',108)]
res=[]
for name,n in sel:
    r=next(x for x in R if int(x['shot'])==n); t0=float(r['t_start'])+0.4; t1=t0+0.5; out=f'work/preview/edge/e{n}.mp4'
    subprocess.run(['rm','-rf',out,out+'.chunks']); subprocess.run([sys.executable,'-I','render_c.py','--t0',f'{t0:.3f}','--t1',f'{t1:.3f}','--ink','0','--out',out]+sys.argv[1:],check=True,capture_output=True); res.append((name,n,t0,out))
json.dump(res,open('work/preview/edge/list.json','w'))
