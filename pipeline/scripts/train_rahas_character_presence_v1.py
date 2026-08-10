from __future__ import annotations
import argparse,random
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset,DataLoader
EXT={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
def parse():
 p=argparse.ArgumentParser();p.add_argument("--characters",type=Path,default=Path("datasets/character_references"));p.add_argument("--output",type=Path,default=Path("pipeline/checkpoints/rahas_character_presence_v1"))
 p.add_argument("--epochs",type=int,default=8);p.add_argument("--samples",type=int,default=12000);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--seed",type=int,default=1217)
 return p.parse_args()
def paths(root):return sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in EXT)
def normalize_glyph(path,r,s=96):
 a=np.asarray(Image.open(path).convert("L"));ink=a<245
 if not ink.any():return np.full((s,s),255,np.uint8)
 y,x=np.where(ink);a=a[y.min():y.max()+1,x.min():x.max()+1];target=r.randint(42,82);scale=target/max(a.shape)
 a=cv2.resize(a,(max(3,int(a.shape[1]*scale)),max(3,int(a.shape[0]*scale))),interpolation=cv2.INTER_CUBIC)
 if r.random()<.5:a=cv2.GaussianBlur(a,(3,3),r.uniform(.1,1.1))
 if r.random()<.4:
  k=np.ones((r.choice([2,3]),r.choice([2,3])),np.uint8);a=cv2.erode(a,k) if r.random()<.5 else cv2.dilate(a,k)
 canvas=np.full((s,s),r.uniform(238,255),np.float32);h,w=a.shape;y0=(s-h)//2+r.randint(-6,6);x0=(s-w)//2+r.randint(-6,6)
 ink=(255-a)/255.;strength=r.uniform(.16,1.0);canvas[y0:y0+h,x0:x0+w]-=ink*strength*r.uniform(130,245)
 return np.clip(canvas,0,255).astype(np.uint8)
def negative(r,g,s=96):
 c=np.full((s,s),r.uniform(238,255),np.float32);c+=g.normal(0,r.uniform(1,7),(s,s))
 mode=r.randrange(5)
 if mode<3:
  centers=[(r.randrange(s),r.randrange(s)) for _ in range(r.randint(1,5))]
  for _ in range(r.randint(8,100)):
   cx,cy=r.choice(centers);x=int(np.clip(g.normal(cx,r.uniform(4,22)),0,s-1));y=int(np.clip(g.normal(cy,r.uniform(4,22)),0,s-1));cv2.circle(c,(x,y),r.choice([1,1,2,3,4]),r.uniform(20,210),-1)
 elif mode==3:
  for _ in range(r.randint(1,7)):
   x,y=r.randrange(s),r.randrange(s);cv2.line(c,(x,y),(int(np.clip(x+r.randint(-28,28),0,s-1)),int(np.clip(y+r.randint(-28,28),0,s-1))),r.uniform(20,190),r.choice([1,2,3]),cv2.LINE_AA)
 else:
  for _ in range(r.randint(0,6)):cv2.circle(c,(r.randrange(s),r.randrange(s)),r.randint(2,15),r.uniform(20,180),-1)
 return np.clip(c,0,255).astype(np.uint8)
def features(a):
 gray=a.astype(np.float32)/255.;dark=1-gray;contrast=cv2.GaussianBlur(dark,(0,0),1.2)
 return torch.from_numpy(np.stack([gray,dark,contrast]).astype(np.float32))
class Data(Dataset):
 def __init__(self,p,n,seed):self.p=p;self.n=n;self.seed=seed
 def __len__(self):return self.n
 def __getitem__(self,i):
  r=random.Random(self.seed+i);g=np.random.default_rng(self.seed+i);positive=i%2==0
  a=normalize_glyph(r.choice(self.p),r) if positive else negative(r,g)
  return features(a),torch.tensor(float(positive))
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.BatchNorm2d(24),nn.GELU(),nn.Conv2d(24,48,3,2,1),nn.BatchNorm2d(48),nn.GELU(),nn.Conv2d(48,96,3,2,1),nn.BatchNorm2d(96),nn.GELU(),nn.Conv2d(96,128,3,2,1),nn.BatchNorm2d(128),nn.GELU(),nn.AdaptiveAvgPool2d(1));self.head=nn.Linear(128,1)
 def forward(self,x):return self.head(self.body(x).flatten(1)).squeeze(1)
def main():
 a=parse();p=paths(a.characters);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");net=Net().to(device);opt=torch.optim.AdamW(net.parameters(),3e-4,weight_decay=1e-4);lossfn=nn.BCEWithLogitsLoss()
 tr=Data(p,a.samples,a.seed);va=Data(p,2000,a.seed+100000);tl=DataLoader(tr,a.batch_size,shuffle=True,num_workers=2,pin_memory=True);vl=DataLoader(va,a.batch_size,num_workers=2,pin_memory=True);a.output.mkdir(parents=True,exist_ok=True);best=0.
 for e in range(a.epochs):
  net.train();losses=[]
  for x,y in tl:
   x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);loss=lossfn(net(x),y);loss.backward();opt.step();losses.append(float(loss))
  net.eval();ok=n=0
  with torch.no_grad():
   for x,y in vl:
    pred=torch.sigmoid(net(x.to(device))).cpu()>=.5;ok+=int((pred==y.bool()).sum());n+=len(y)
  acc=ok/n;state={"model":net.state_dict(),"epoch":e,"val_accuracy":acc,"args":vars(a)}
  torch.save(state,a.output/"latest.pt")
  if acc>best:best=acc;torch.save(state,a.output/"best.pt")
  print(f"epoch={e+1}/{a.epochs} loss={np.mean(losses):.4f} val_accuracy={acc:.4f}",flush=True)
 print(f"device={device} glyphs={len(p)} best_accuracy={best:.4f} output={a.output.resolve()}")
if __name__=="__main__":main()
