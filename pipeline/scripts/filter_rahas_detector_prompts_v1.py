from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_rahas_character_presence_v1 import Net,features
def parse():
 p=argparse.ArgumentParser();p.add_argument("--image",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--threshold",type=float,default=.55);p.add_argument("--batch-size",type=int,default=128);return p.parse_args()
def prep(crop,s=96):
 a=np.asarray(crop.convert("L"));ink=a<245
 if ink.any():y,x=np.where(ink);a=a[y.min():y.max()+1,x.min():x.max()+1]
 scale=76/max(a.shape);a=cv2.resize(a,(max(3,int(a.shape[1]*scale)),max(3,int(a.shape[0]*scale))),interpolation=cv2.INTER_CUBIC)
 canvas=np.full((s,s),255,np.uint8);h,w=a.shape;y=(s-h)//2;x=(s-w)//2;canvas[y:y+h,x:x+w]=a
 return features(canvas)
def main():
 a=parse();device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ck=torch.load(a.checkpoint,map_location=device,weights_only=False);net=Net().to(device);net.load_state_dict(ck["model"]);net.eval()
 page=Image.open(a.image).convert("RGB")
 with a.manifest.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
 tensors=[]
 for r in rows:tensors.append(prep(page.crop(tuple(int(r[k]) for k in ("x0","y0","x1","y1")))))
 scores=[]
 with torch.no_grad():
  for i in range(0,len(tensors),a.batch_size):scores.extend(torch.sigmoid(net(torch.stack(tensors[i:i+a.batch_size]).to(device))).cpu().tolist())
 kept=[]
 for r,s in zip(rows,scores):
  r["presence_score"]=f"{s:.6f}"
  if s>=a.threshold:kept.append(r)
 kept.sort(key=lambda r:(int(r["line_index"]),int(r["x0"])));a.output.mkdir(parents=True,exist_ok=True);overlay=page.copy();draw=ImageDraw.Draw(overlay);font=ImageFont.load_default()
 for i,r in enumerate(kept,1):
  r["index"]=str(i);b=tuple(int(r[k]) for k in ("x0","y0","x1","y1"));draw.rectangle(b,outline=(0,205,105),width=2);draw.text((b[0]+2,b[1]+2),f"{i:03d}",fill=(0,95,50),font=font)
 fields=list(kept[0].keys()) if kept else list(rows[0].keys())+["presence_score"]
 with (a.output/"prompt_manifest.csv").open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(kept)
 overlay.save(a.output/"filtered_overlay.png");print(f"input={len(rows)} kept={len(kept)} threshold={a.threshold} output={a.output.resolve()}")
if __name__=="__main__":main()
