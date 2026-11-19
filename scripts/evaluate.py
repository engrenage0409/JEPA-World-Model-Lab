"""Compare open-loop multi-step action-conditioned latent prediction on held-out episodes."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from jepa.data import ClipDataset
from jepa.models import MaskedJEPA,ActionDynamics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--encoder",default="checkpoints/masked_jepa.pt")
    p.add_argument("--dynamics",default="checkpoints/action_dynamics.pt")
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--out",default="results/latent_eval.json")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    enc=torch.load(a.encoder,map_location="cpu",weights_only=True)
    dyn=torch.load(a.dynamics,map_location="cpu",weights_only=True)
    model=MaskedJEPA(**enc["config"]).to(device);model.load_state_dict(enc["model"])
    predictor=ActionDynamics(**dyn["config"]).to(device);predictor.load_state_dict(dyn["model"])
    model.eval();predictor.eval()
    paths=[Path(a.data)/x for x in dyn["val_episodes"]]
    dataset=ClipDataset(paths,a.steps)
    loader=DataLoader(dataset,batch_size=4)
    sums=torch.zeros(a.steps);count=0
    with torch.inference_mode():
        for sample in loader:
            frames,actions=sample["frames"].to(device),sample["actions"].to(device)
            B,T,C,H,W=frames.shape
            latent=model.target(frames.reshape(B*T,C,H,W)).mean(1).reshape(B,T,-1)
            pred=predictor.rollout(latent[:,0],actions)
            sums+=(pred-latent[:,1:]).square().mean(-1).sum(0).cpu()
            count+=B
    result={"windows":count,"step_mse":(sums/count).tolist(),
            "warning":"Latent MSE is NOT image reconstruction quality or downstream control success."}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
