"""第 8 条：BGA 字符前景色（每帧亮度最高 3% 像素均值）→ 时间曲线 + 全片中位色。"""
import numpy as np,subprocess,json
W,H=320,180
p=subprocess.Popen(['ffmpeg','-v','error','-i','input2/bga/Credits.bga.mp4','-an','-vf',f'fps=10,scale={W}:{H}','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
fs=W*H*3; cols=[]
while True:
    b=p.stdout.read(fs)
    if len(b)<fs:break
    f=np.frombuffer(b,np.uint8).reshape(-1,3).astype(np.float32); g=f.mean(1)
    thr=np.percentile(g,97); sel=f[g>=thr]; cols.append(sel.mean(0) if len(sel) else np.zeros(3))
C=np.array(cols); hx=lambda c:'#%02x%02x%02x'%tuple(int(v) for v in c)
med=np.median(C,0); 
# 按配乐时间轴（BGA 从 0.16s 起）每 10 秒一个代表色
rep=[(s,hx(C[int((s+0.16)*10):int((s+10.16)*10)].mean(0))) for s in range(0,278,10)]
json.dump(dict(dt=0.1,median=hx(med),colors=[hx(c) for c in C],per10s=rep),open('work/bga_accent.json','w'))
print('median',hx(med),'mean',hx(C.mean(0)));print(rep)
