"""Train a small JEPA-style masked future-patch representation model."""
import argparse,json,random
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from jepa.data import split_episodes,ClipDataset
from jepa.models import MaskedJEPA,random_patch_mask

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--dim",type=int,default=128)
    p.add_argument("--mask-ratio",type=float,default=0.5)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--ema",type=float,default=0.996)
    p.add_argument("--out",default="checkpoints/masked_jepa.pt")
    a=p.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    tr,va=split_episodes(a.data)
    train=DataLoader(ClipDataset(tr,a.steps),batch_size=a.batch_size,shuffle=True)
    val=DataLoader(ClipDataset(va,a.steps),batch_size=a.batch_size)
    model=MaskedJEPA(dim=a.dim).to(device)
    opt=torch.optim.AdamW(list(model.encoder.parameters())+list(model.predictor.parameters()),lr=a.lr)
    best=float("inf");out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        metrics={}
        for name,loader in [("train",train),("val",val)]:
            model.train(name=="train")
            values=[]
            for batch in loader:
                frames=batch["frames"].to(device)
                # Sample a random (context,next-frame) pair in each sequence.
                t=random.randrange(a.steps)
                with torch.set_grad_enabled(name=="train"):
                    tokens=model.encoder.grid**2
                    mask=random_patch_mask(frames.shape[0],tokens,a.mask_ratio,device)
                    loss,_,_=model(frames[:,t],frames[:,t+1],mask)
                    if name=="train":
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        model.update_target(a.ema)
                values.append(loss.item())
            metrics[name]=sum(values)/len(values)
        print(json.dumps({"epoch":epoch+1,**metrics}))
        if metrics["val"]<best:
            best=metrics["val"]
            torch.save({"model":model.state_dict(),"config":{"dim":a.dim},"epoch":epoch+1,
                        "train_episodes":[x.name for x in tr],"val_episodes":[x.name for x in va],
                        "val_masked_feature_mse":best},out)
    print("Saved:",out,"best validation masked loss:",best)

if __name__=="__main__":main()
