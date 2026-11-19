"""Latent-goal CEM demonstration; not a learned CarRacing controller."""
import argparse,json
from pathlib import Path
import torch
from jepa.data import ClipDataset
from jepa.models import MaskedJEPA,ActionDynamics
from jepa.planning import cem_plan

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    p.add_argument("--encoder",default="checkpoints/masked_jepa.pt")
    p.add_argument("--dynamics",default="checkpoints/action_dynamics.pt")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    e=torch.load(a.encoder,map_location="cpu",weights_only=True)
    d=torch.load(a.dynamics,map_location="cpu",weights_only=True)
    j=MaskedJEPA(**e["config"]).to(device);j.load_state_dict(e["model"]);j.eval()
    dyn=ActionDynamics(**d["config"]).to(device);dyn.load_state_dict(d["model"]);dyn.eval()
    sample=ClipDataset([Path(a.data)/d["val_episodes"][0]],steps=6)[0]
    with torch.inference_mode():
        start=j.target(sample["frames"][:1].to(device)).mean(1)[0]
        goal=j.target(sample["frames"][-1:].to(device)).mean(1)[0]
        actions,metrics=cem_plan(dyn,start,goal,horizon=6)
    print(json.dumps({"candidate_actions":actions.cpu().tolist(),**metrics},indent=2))

if __name__=="__main__":main()
