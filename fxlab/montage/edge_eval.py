import json,sys,numpy as np
sys.path.insert(0,'.')
import check_e as C
res=json.load(open('work/preview/edge/list.json')); rows=[]
for name,n,t0,out in res:
    meta=json.load(open(out+'.meta.json')); F=C.read_frames(out); i=len(F)//2; f=F[i]; m=meta['frames'][i]
    L=C.lum(f); x,y,ww,wh=m['x'],m['y'],m['ww'],m['wh']; wm=float(L[int(y+0.2*wh):int(y+0.8*wh),int(x+0.2*ww):int(x+0.8*ww)].mean())
    pm=C.profile_metrics(f,m,wm); rows.append((name,n,round(wm/255*100,1),[(a,round(b,1),round(c,1)) for a,b,c in pm]))
    print(name,'shot',n,'窗内均亮 %.1f%%'%(wm/255*100),'窗 %dx%d'%(ww,wh),'| (边,反弹%,10px变化%):',[(a,round(b,1),round(c,1)) for a,b,c in pm])
mr=max(b for r in rows for a,b,c in r[3]); mc=max(c for r in rows for a,b,c in r[3]); print('最大反弹 %.1f%%  每10px最大变化 %.1f%%'%(mr,mc)); json.dump(rows,open('work/edge_check.json','w'),ensure_ascii=False)
