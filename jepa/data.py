"""Reuse world-model-lab CarRacing episodes: observations[T+1,96,96,3], actions[T,3].
No episode split leakage; T context frames plus T prediction targets.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def load_episode(path):
    with np.load(path,allow_pickle=False) as a:
        required=("observations","actions","rewards","dones")
        if any(key not in a for key in required): raise ValueError(f"{path}: missing expected arrays")
        images,actions,rewards,dones=[a[key].copy() for key in required]
    t=len(actions)
    if images.shape!=(t+1,96,96,3) or images.dtype!=np.uint8:
        raise ValueError(f"{path}: expected uint8 observations [T+1,96,96,3]")
    if actions.shape!=(t,3) or rewards.shape!=(t,) or dones.shape!=(t,):
        raise ValueError(f"{path}: inconsistent transition alignment")
    return images,actions.astype(np.float32),rewards.astype(np.float32),dones.astype(bool)

def split_episodes(directory,seed=42,val_fraction=0.2):
    paths=sorted(Path(directory).glob("*.npz"))
    if len(paths)<2: raise ValueError("Need at least two episode files")
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(paths))
    n=max(1,min(len(paths)-1,round(val_fraction*len(paths))))
    reserved=set(order[:n].tolist())
    return [p for i,p in enumerate(paths) if i not in reserved],[p for i,p in enumerate(paths) if i in reserved]

class ClipDataset(Dataset):
    """For each sample, frames [L+1,3,96,96], actions [L,3], no terminal crossing."""
    def __init__(self,paths,steps=4,stride=4):
        if steps<1 or stride<1: raise ValueError("steps and stride must be positive")
        self.episodes=[load_episode(p) for p in paths]
        self.windows=[]
        for e,(_,actions,_,dones) in enumerate(self.episodes):
            for start in range(0,len(actions)-steps+1,stride):
                if dones[start:start+steps-1].any(): continue
                self.windows.append((e,start))
        if not self.windows: raise ValueError("No windows found; reduce --steps")
        self.steps=steps
    def __len__(self): return len(self.windows)
    def __getitem__(self,index):
        e,i=self.windows[index]
        imgs,actions,_,_=self.episodes[e]
        frames=np.ascontiguousarray(imgs[i:i+self.steps+1].transpose(0,3,1,2))
        return {"frames":torch.from_numpy(frames).float()/255.0,
                "actions":torch.tensor(actions[i:i+self.steps].copy())}
