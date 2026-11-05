import csv,json,subprocess,io,os
from PIL import Image,ImageDraw
from concurrent.futures import ThreadPoolExecutor
os.chdir(os.path.dirname(os.path.abspath(__file__)))
R={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}; h=json.load(open('work/hits.json'))
FW,FH=192,132
def fr(cid,t):
    r=R[cid]; t=min(max(t,0),float(r['dur'])-0.02)
    p=subprocess.run(['ffmpeg','-v','error','-ss',f'{t:.3f}','-i','input2/clips/'+r['file'],'-frames:v','1','-vf',f'scale={FW}:{FH}','-f','image2pipe','-vcodec','png','-'],capture_output=True).stdout
    return Image.open(io.BytesIO(p)).convert('RGB')
def cell(cid):
    th=h[str(cid)]['t_hit']; c=Image.new('RGB',(3*FW+4,FH+16),'black'); d=ImageDraw.Draw(c)
    for i,dt in enumerate((-0.1,0,0.1)): c.paste(fr(cid,th+dt),(i*(FW+2),16))
    d.text((3,2),f"clip {cid}  t_hit={th:.2f}s  conf={h[str(cid)]['conf']:.2f}  (alt {h[str(cid)]['t_hit2']:.2f}s)",fill=(255,255,0)); return c
ids=sorted(R)
with ThreadPoolExecutor(4) as ex: cells=dict(zip(ids,ex.map(cell,ids)))
os.makedirs('../docs/hit_sheets',exist_ok=True)
for s in range(0,len(ids),28):
    chunk=ids[s:s+28]; cw,ch=cells[chunk[0]].size; S=Image.new('RGB',(2*cw+6,14*(ch+4)),'black')
    for k,cid in enumerate(chunk): S.paste(cells[cid],((k%2)*(cw+6),(k//2)*(ch+4)))
    S.save(f'../docs/hit_sheets/hit_sheet_{s//28+1}.jpg',quality=80)
print('ok')
