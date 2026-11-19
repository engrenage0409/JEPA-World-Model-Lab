"""PCA projection of target vs imagined latent trajectories (not generated video)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from jepa.data import ClipDataset
from jepa.models import MaskedJEPA,ActionDynamics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--encoder",default="checkpoints/masked_jepa.pt")
    p.add_argument("--dynamics",default="checkpoints/action_dynamics.pt")
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--out",default="results/latent_trajectory.png")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    e=torch.load(a.encoder,map_location="cpu",weights_only=True)
    d=torch.load(a.dynamics,map_location="cpu",weights_only=True)
    enc=MaskedJEPA(**e["config"]).to(device);enc.load_state_dict(e["model"]);enc.eval()
    dyn=ActionDynamics(**d["config"]).to(device);dyn.load_state_dict(d["model"]);dyn.eval()
    example=ClipDataset([Path(a.data)/d["val_episodes"][0]],steps=a.steps)[0]
    with torch.inference_mode():
        frames=example["frames"].to(device)
        latent=enc.target(frames).mean(1)
        pred=dyn.rollout(latent[:1],example["actions"].unsqueeze(0).to(device))[0]
    real=latent[1:].cpu().numpy();imagined=pred.cpu().numpy()
    points=np.concatenate([real,imagined],axis=0)
    centered=points-points.mean(0,keepdims=True)
    _,_,v=np.linalg.svd(centered,full_matrices=False)
    xy=centered@v[:2].T
    xy=(xy-xy.min(0))/(np.ptp(xy,axis=0)+1e-9)
    xy=xy*420+40
    image=Image.new("RGB",(520,520),"white")
    draw=ImageDraw.Draw(image)
    for i,color in ((0,"blue"),(len(real),"red")):
        for j in range(len(real)):
            x,y=xy[i+j]
            draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
            if j: draw.line([tuple(xy[i+j-1]),(float(x),float(y))],fill=color,width=2)
    draw.text((8,8),"Blue = target latents; Red = action-predicted latents",fill="black")
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);image.save(out)
    print("Saved",out)

if __name__=="__main__":main()
