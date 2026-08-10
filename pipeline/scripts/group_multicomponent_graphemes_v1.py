from __future__ import annotations
import argparse,csv
from itertools import combinations
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
def parse():
 p=argparse.ArgumentParser(description="Promote multi-component Brahmi graphemes, including three-dot e-matra.")
 p.add_argument("--input-dir",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--image",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--line-step",type=int,default=105);return p.parse_args()
def main():
 a=parse();rest=np.asarray(Image.open(a.input_dir/"restored_dark_ocr.png").convert("L"));mp=a.input_dir/"final_keep_mask.png"
 if not mp.exists():mp=a.input_dir/"keep_mask.png"
 mask=np.asarray(Image.open(mp).convert("L"));binary=(mask>=96).astype(np.uint8);binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)
 n,lab,stats,cent=cv2.connectedComponentsWithStats(binary,8)
 with a.manifest.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
 main_labels=set()
 for row in rows:
  cx=(int(row["x0"])+int(row["x1"]))//2;cy=(int(row["y0"])+int(row["y1"]))//2;v=int(lab[np.clip(cy,0,lab.shape[0]-1),np.clip(cx,0,lab.shape[1]-1)])
  if v:main_labels.add(v)
 small=[]
 for i in range(1,n):
  x,y,w,h,area=map(int,stats[i])
  if i in main_labels or area<25 or area>240 or max(w,h)>20 or min(w,h)<5:continue
  fill=area/max(w*h,1);aspect=w/max(h,1)
  if fill<.45 or not .65<=aspect<=1.5:continue
  dark=1-rest[y:y+h,x:x+w].astype(np.float32)/255.;support=binary[y:y+h,x:x+w]>0
  if not support.any() or float(dark[support].mean())<.40:continue
  small.append(dict(label=i,x0=x,y0=y,x1=x+w,y1=y+h,area=area,cx=float(cent[i,0]),cy=float(cent[i,1]),size=max(w,h)))
 line_values={}
 for row in rows:
  cy=(int(row["y0"])+int(row["y1"]))/2;li=int(round(cy/a.line_step));line_values.setdefault(li,[]).append(cy)
 lines={li:float(np.median(v)) for li,v in line_values.items() if len(v)>=4}
 main_centers=[((int(r["x0"])+int(r["x1"]))/2,(int(r["y0"])+int(r["y1"]))/2,int(r["x1"])-int(r["x0"])) for r in rows]
 median_width=float(np.median([q[2] for q in main_centers])) if main_centers else 60.0
 cell=44;grid={}
 for i,c in enumerate(small):grid.setdefault((int(c["cx"]//cell),int(c["cy"]//cell)),[]).append(i)
 candidates=[]
 for i,c in enumerate(small):
  gx,gy=int(c["cx"]//cell),int(c["cy"]//cell);near=set()
  for dx in (-1,0,1):
   for dy in (-1,0,1):near.update(grid.get((gx+dx,gy+dy),[]))
  for j,k in combinations(sorted(q for q in near if q>i),2):
   m=[c,small[j],small[k]];areas=np.array([q["area"] for q in m],np.float32);sizes=np.array([q["size"] for q in m],np.float32);xs=np.array([q["cx"] for q in m]);ys=np.array([q["cy"] for q in m])
   mean=max(2.,float(sizes.mean()));xspan=float(xs.max()-xs.min());yspan=float(ys.max()-ys.min())
   if areas.max()/max(areas.min(),1)>1.8 or xspan>mean*6.5 or yspan>mean*6.5:continue
   yorder=np.argsort(ys);yox=xs[yorder];yoy=ys[yorder];ygaps=np.diff(yoy)
   chevron=yspan>=mean*1.5 and abs(float(yox[0]-yox[2]))<=max(4.0,mean*.38) and abs(float(yox[1]-(yox[0]+yox[2])/2))>=mean*1.15 and min(ygaps)>=mean*.55 and max(ygaps)/max(min(ygaps),1)<=1.65
   vertical=xspan<=max(6.0,mean*.72) and yspan>=mean*1.5
   if chevron:
    pass
   elif vertical:
    order=np.argsort(ys);oy=ys[order];ox=xs[order];gaps=np.diff(oy)
    if min(gaps)<mean*.65 or max(gaps)/max(min(gaps),1)>1.40 or float(np.std(ox))>max(2.2,mean*.24):continue
   else:
    if xspan<mean*1.25 or yspan<mean*1.25:continue
    order=np.argsort(xs);ox=xs[order];oy=ys[order];dx=np.diff(ox);dy=np.diff(oy)
    if min(dx)<mean*.55 or not ((dy[0]>0 and dy[1]>0) or (dy[0]<0 and dy[1]<0)):continue
    steps=np.hypot(dx,dy)
    if min(steps)<mean*.75 or max(steps)/max(min(steps),1)>1.40:continue
    slope,intercept=np.polyfit(ox,oy,1);residual=float(np.max(np.abs(oy-(slope*ox+intercept))))
    if not .50<=abs(float(slope))<=2.0 or residual>max(2.0,mean*.25):continue
   cy=float(ys.mean());li=int(round(cy/a.line_step));line_y=lines.get(li)
   cx=float(xs.mean());neighbors=[q for q in main_centers if abs(q[1]-cy)<=a.line_step*.34]
   left=[cx-q[0] for q in neighbors if q[0]<cx];right=[q[0]-cx for q in neighbors if q[0]>cx]
   max_neighbor_gap=max(110.0,median_width*2.15)
   if not left or not right or min(left)>max_neighbor_gap or min(right)>max_neighbor_gap:continue
   if line_y is None or abs(cy-line_y)>a.line_step*.35:continue
   d=[float(np.hypot(m[u]["cx"]-m[v]["cx"],m[u]["cy"]-m[v]["cy"])) for u,v in ((0,1),(0,2),(1,2))]
   if min(d)<mean*.65 or max(d)>mean*6.5:continue
   score=float(np.std(d)/max(np.mean(d),1))+abs(cy-line_y)/a.line_step+(areas.max()/areas.min()-1)*.08
   candidates.append((score,(i,j,k)))
 used=set();groups=[]
 for score,ids in sorted(candidates):
  if used.intersection(ids):continue
  m=[small[q] for q in ids];x0=min(q["x0"] for q in m);y0=min(q["y0"] for q in m);x1=max(q["x1"] for q in m);y1=max(q["y1"] for q in m);cy=sum(q["cy"]*q["area"] for q in m)/sum(q["area"] for q in m)
  groups.append(dict(line_index=int(round(cy/a.line_step))+1,x0=x0,y0=y0,x1=x1,y1=y1,grouping_reason="i_three_dot",source_labels=";".join(str(q["label"]) for q in m),group_score=f"{score:.6f}"));used.update(ids)
 for row in rows:row.setdefault("grouping_reason","connected_component");row.setdefault("source_labels","");row.setdefault("group_score","")
 combined=rows+groups;combined.sort(key=lambda r:(int(r["line_index"]),int(r["x0"])))
 for i,row in enumerate(combined,1):row["index"]=str(i)
 a.output.mkdir(parents=True,exist_ok=True);fields=["index","line_index","x0","y0","x1","y1","grouping_reason","source_labels","group_score"]
 with (a.output/"prompt_manifest.csv").open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(combined)
 overlay=Image.open(a.image).convert("RGB").resize((rest.shape[1],rest.shape[0]),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(overlay);font=ImageFont.load_default()
 for row in combined:
  b=tuple(int(row[k]) for k in ("x0","y0","x1","y1"));special=row["grouping_reason"]=="i_three_dot";draw.rectangle(b,outline=(255,145,0) if special else (0,210,110),width=3 if special else 2);draw.text((b[0]+1,b[1]+1),row["index"],fill=(150,65,0) if special else (0,105,55),font=font)
 overlay.save(a.output/"grouped_overlay.png");print(f"base_prompts={len(rows)} small_components={len(small)} e_matra_groups={len(groups)} total={len(combined)}");print(f"output={a.output.resolve()}")
if __name__=="__main__":main()
