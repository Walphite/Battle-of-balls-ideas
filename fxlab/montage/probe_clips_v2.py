import subprocess,json,csv,sys,os,re,glob
root='input2/clips'
fs=[os.path.join(d,f) for d,_,ff in os.walk(root) for f in ff]
def nk(s): return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)',s)]
fs.sort(key=lambda p:nk(os.path.relpath(p,root)))
rows=[]
for i,p in enumerate(fs,1):
    j=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',p]))
    v=[s for s in j['streams'] if s['codec_type']=='video'][0]
    a=[s for s in j['streams'] if s['codec_type']=='audio']
    n,d=map(int,v['r_frame_rate'].split('/'))
    rot=0
    for sd in v.get('side_data_list',[]): rot=sd.get('rotation',rot)
    rot=int(rot or v.get('tags',{}).get('rotate',0))
    w,h=v['width'],v['height']
    if abs(rot)%180==90: w,h=h,w
    rows.append(dict(clip_id=i,file=os.path.relpath(p,root),w=w,h=h,rot=rot,fps=round(n/d,3),dur=round(float(j['format']['duration']),3),codec=v['codec_name'],pix=v.get('pix_fmt'),audio=int(bool(a)),size_mb=round(int(j['format']['size'])/1e6,1)))
os.makedirs('work',exist_ok=True)
w=csv.DictWriter(open('work/clips.csv','w',newline='',encoding='utf-8'),fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
import collections,statistics as st
D=[r['dur'] for r in rows]
print(len(rows),'total',round(sum(D),1),'min/med/max',min(D),st.median(D),max(D))
print('res',collections.Counter((r['w'],r['h']) for r in rows));print('fps',collections.Counter(r['fps'] for r in rows))
print('codec',collections.Counter((r['codec'],r['pix']) for r in rows),'audio',sum(r['audio'] for r in rows),'rot',collections.Counter(r['rot'] for r in rows))
print('<0.8',[r['clip_id'] for r in rows if r['dur']<0.8])
print('hist',collections.Counter(min(int(x//2)*2,20) for x in D))
