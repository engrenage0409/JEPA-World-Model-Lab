"""Train action-conditioned feature dynamics against a frozen EMA JEPA target."""
import argparse,json,random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from jepa.data import split_episodes,ClipDataset
from jepa.models import MaskedJEPA,ActionDynamics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--encoder",default="checkpoints/masked_jepa.pt")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--out",default="checkpoints/action_dynamics.pt")
    a=p.parse_args()
    random.seed(42);torch.manual_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(a.encoder,map_location="cpu",weights_only=True)
    jepa=MaskedJEPA(**ck["config"]).to(device)
    jepa.load_state_dict(ck["model"]);jepa.eval()
    jepa.requires_grad_(False)
    tr=[Path(a.data)/p for p in ck["train_episodes"]]
    va=[Path(a.data)/p for p in ck["val_episodes"]]
    train=DataLoader(ClipDataset(tr,a.steps),batch_size=a.batch_size,shuffle=True)
    val=DataLoader(ClipDataset(va,a.steps),batch_size=a.batch_size)
    dynamics=ActionDynamics(dim=ck["config"]["dim"]).to(device)
    opt=torch.optim.AdamW(dynamics.parameters(),lr=a.lr)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);best=float("inf")
    for epoch in range(a.epochs):
        metrics={}
        for name,loader in (("train",train),("val",val)):
            dynamics.train(name=="train")
            values=[]
            for batch in loader:
                frames=batch["frames"].to(device)
                actions=batch["actions"].to(device)
                with torch.no_grad():
                    B,T,C,H,W=frames.shape
                    features=jepa.target(frames.reshape(B*T,C,H,W)).mean(1).reshape(B,T,-1)
                with torch.set_grad_enabled(name=="train"):
                    prediction=dynamics.rollout(features[:,0],actions)
                    loss=F.mse_loss(prediction,features[:,1:])
                    if name=="train":
                        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                values.append(loss.item())
            metrics[name]=sum(values)/len(values)
        print(json.dumps({"epoch":epoch+1,**metrics}))
        if metrics["val"]<best:
            best=metrics["val"]
            torch.save({"model":dynamics.state_dict(),"config":{"dim":ck["config"]["dim"]},
                        "encoder":a.encoder,"epoch":epoch+1,
                        "val_latent_mse":best,"val_episodes":[p.name for p in va]},out)
    print("Saved:",out,"validation rollout latent MSE:",best)

if __name__=="__main__":main()
