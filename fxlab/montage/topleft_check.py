"""左上角清理自检：图标像素占比、BGA 标题对比度、4 倍放大前后对比图。"""
import os,sys,json,subprocess,numpy as np
from PIL import Image,ImageFilter
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
W,H=960,540
def fr(fn,t):
    p=subprocess.run(['ffmpeg','-v','error','-ss',f'{t:.3f}','-i',fn,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    return np.frombuffer(p,np.uint8).reshape(H,W,3).astype(np.float32)
def icon_frac(f):
    c=f[:int(0.30*H),:int(0.30*W)]/255; mx=c.max(2); mn=c.min(2); sat=(mx-mn)/(mx+1e-3); return float(((sat>0.5)&(mx>0.6)).mean())
def bga_text(t):
    p=subprocess.run(['ffmpeg','-v','error','-ss',f'{t+0.16:.3f}','-i','input2/bga/Credits.bga.mp4','-frames:v','1','-vf','crop=iw*0.84:ih:iw*0.08:0,scale=960:-2,crop=960:540','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    b=np.frombuffer(p,np.uint8).reshape(H,W,3).astype(np.float32); L=b.mean(2); m=np.zeros((H,W),bool); r=L[:int(0.30*H),:int(0.40*W)]>60; m[:r.shape[0],:r.shape[1]]=r; return m
def contrast(f,m):
    if m.sum()<30: return None
    L=f@np.array([0.299,0.587,0.114],np.float32); im=Image.fromarray((m*255).astype(np.uint8)); d8=np.asarray(im.filter(ImageFilter.MaxFilter(17)))>0; d1=np.asarray(im.filter(ImageFilter.MaxFilter(3)))>0
    ring=d8&~d1; 
    return float(L[m].mean()/max(L[ring].mean(),1.0))
out={}
# E 前后（样内每 1 s）
for tag,fn in (('before','work/preview/sample_E_prev.mp4'),('after','work/preview/sample_E_540.mp4')):
    fr_=[];cr=[]
    for i in range(12):
        t=i+0.5; f=fr(fn,t); fr_.append(icon_frac(f)); c=contrast(f,bga_text(61.997+t)); 
        if c: cr.append(c)
    out['E_'+tag]=dict(icon=float(np.mean(fr_)),contrast_mean=float(np.mean(cr)),contrast_min=float(np.min(cr)),n=len(cr))
for tag in ('old','new'):
    fn=f'work/preview/v6/s2_{tag}.mp4'; f=fr(fn,0.2); out['s2_'+tag]=dict(icon=icon_frac(f),contrast=contrast(f,bga_text(4.2)))
print(json.dumps(out,indent=1,ensure_ascii=False)); json.dump(out,open('work/topleft_check.json','w'))
# 4× 放大对比图：左上角 30%×30%
def zoom(f): return Image.fromarray(f[:int(0.30*H),:int(0.30*W)].astype(np.uint8)).resize((int(0.30*W)*4,int(0.30*H)*4),Image.LANCZOS)
pairs=[('75% 窗口（shot 2，歌曲 4.2 s）',fr('work/preview/v6/s2_old.mp4',0.2),fr('work/preview/v6/s2_new.mp4',0.2),'topleft_75pct_v6.jpg'),
       ('满屏窗口（样 E 8.5 s，窗高 98%）',fr('work/preview/sample_E_prev.mp4',8.5),fr('work/preview/sample_E_540.mp4',8.5),'topleft_full_v6.jpg'),
       ('字符溶解中（样 E 1.6 s，溶出覆盖中）',fr('work/preview/sample_E_prev.mp4',1.6),fr('work/preview/sample_E_540.mp4',1.6),'topleft_dissolve_v6.jpg')]
for name,a,b,fn in pairs:
    za,zb=zoom(a),zoom(b); S=Image.new('RGB',(za.size[0],za.size[1]*2+30)); S.paste(za,(0,0)); S.paste(zb,(0,za.size[1]+30))
    from PIL import ImageDraw; d=ImageDraw.Draw(S); d.text((8,4),'修改前 '+name,fill=(255,255,0)); d.text((8,za.size[1]+34),'修改后',fill=(255,255,0)); S.save('../docs/'+fn,quality=88)
print('saved')
