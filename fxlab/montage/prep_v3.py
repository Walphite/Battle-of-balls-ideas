"""把未受影响的 final_v2 分块直接复用（硬链接），只留下受影响的块待渲染。"""
import os,sys,json,csv,shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('work/final3',exist_ok=True); shutil.copy('work/final2/segments.json','work/final3/segments.json')
R=list(csv.DictReader(open('work/edl_v10.csv',encoding='utf-8'))); sh={int(r['shot']):(float(r['t_start']),float(r['t_end'])) for r in R}
iv=[]
old=json.load(open('work/bw_plan.json'))
for n in old['bw_shots']: iv.append(sh[int(n)])
for t0,du in old['fadeback']: iv.append((t0,t0+du))
for e in json.load(open('work/glyph_schedule_v10.json'))['events']:
    if e['kind'] in('out','in','scan','f'): iv.append((e.get('t',e.get('A0'))-0.6,e.get('A2',e.get('A1',e.get('A0')))+0.6))
iv.append((202,219.7))
for p in json.load(open('work/bw_pulses.json'))['pulses']: iv.append((p['t_fall0']-0.2,(p['t_rise1'] or 279)+0.2))
def hit(c0,c1): return any(a<c1/30 and b>c0/30 for a,b in iv)
f=json.load(open('work/final3/segments.json'))['frames']; tot=aff=0
for res,CH in(('540',300),('1080',60)):
    for p in (1,2,3):
        os.makedirs(f'work/final3/{res}/p{p}',exist_ok=True)
        for c in range(f[p-1],f[p],CH):
            tot+=1; src=f'work/final2/{res}/p{p}/c_{c:05d}.mp4'; dst=f'work/final3/{res}/p{p}/c_{c:05d}.mp4'
            if hit(c,min(f[p],c+CH)): aff+=1; continue
            if os.path.exists(src) and not os.path.exists(dst):
                os.link(src,dst)
                if os.path.exists(src+'.meta.json'): shutil.copy(src+'.meta.json',dst+'.meta.json')
    print(res,'待渲染块',sum(1 for p in (1,2,3) for c in range(f[p-1],f[p],CH) if not os.path.exists(f'work/final3/{res}/p{p}/c_{c:05d}.mp4')),'/',sum(len(range(f[p-1],f[p],CH)) for p in (1,2,3)))
