"""Small spatial patch-token JEPA. Not V-JEPA/V-JEPA2 official architecture.

Masked objective: online encoder observes context frame; predictor reconstructs
EMA target encoder features for masked TARGET patches from next frame.
AC objective: action-conditioned dynamics predicts future pooled target features.
"""
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

class PatchEncoder(nn.Module):
    def __init__(self,dim=128,patch=16,image_size=96,depth=2,heads=4):
        super().__init__()
        if image_size%patch: raise ValueError("image size must divide patch")
        self.dim=dim
        self.patch=patch
        self.grid=image_size//patch
        self.patchify=nn.Conv2d(3,dim,patch,patch)
        self.position=nn.Parameter(torch.randn(1,self.grid**2,dim)*0.02)
        layer=nn.TransformerEncoderLayer(dim,heads,dim*4,dropout=0.0,
                                         batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,depth)
        self.norm=nn.LayerNorm(dim)
    def forward(self,image):
        x=self.patchify(image).flatten(2).transpose(1,2)+self.position
        return self.norm(self.blocks(x))

class MaskedPredictor(nn.Module):
    def __init__(self,dim=128,depth=2,heads=4):
        super().__init__()
        self.mask_token=nn.Parameter(torch.zeros(1,1,dim))
        layer=nn.TransformerEncoderLayer(dim,heads,dim*4,dropout=0.0,
                                         batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,depth)
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim,dim)
    def forward(self,context,mask):
        if context.shape[:2]!=mask.shape: raise ValueError("mask [B,N] must match tokens")
        hidden=torch.where(mask.unsqueeze(-1),self.mask_token.expand_as(context),context)
        return self.head(self.norm(self.blocks(hidden)))

class MaskedJEPA(nn.Module):
    def __init__(self,dim=128,patch=16,depth=2):
        super().__init__()
        self.encoder=PatchEncoder(dim,patch,depth=depth)
        self.target=deepcopy(self.encoder)
        self.target.requires_grad_(False)
        self.predictor=MaskedPredictor(dim,depth=depth)
    @torch.no_grad()
    def update_target(self,momentum=0.996):
        for target,online in zip(self.target.parameters(),self.encoder.parameters()):
            target.lerp_(online,1-momentum)
    def train(self,mode=True):
        super().train(mode);self.target.eval();return self
    def forward(self,context_frame,target_frame,mask):
        context=self.encoder(context_frame)
        with torch.no_grad(): targets=self.target(target_frame)
        pred=self.predictor(context,mask)
        # Only masked patch predictions are supervised. L2 on normalized features.
        per_patch=(F.normalize(pred,dim=-1)-F.normalize(targets,dim=-1)).square().mean(-1)
        return per_patch[mask].mean(),pred,targets

def random_patch_mask(batch,tokens,ratio,device):
    if not 0<ratio<=1: raise ValueError("mask ratio must be within (0,1]")
    k=max(1,min(tokens,round(tokens*ratio)))
    noise=torch.rand(batch,tokens,device=device)
    indices=noise.argsort(-1)[:,:k]
    mask=torch.zeros(batch,tokens,device=device,dtype=torch.bool)
    mask.scatter_(1,indices,True)
    return mask

class ActionDynamics(nn.Module):
    """Residual MLP latent transition on mean-pooled JEPA tokens; single-step rollout."""
    def __init__(self,dim=128,action_dim=3,hidden=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim+action_dim,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,dim))
    def forward(self,z,action):
        return z+self.net(torch.cat([z,action],-1))
    def rollout(self,start,actions):
        z=start;future=[]
        for t in range(actions.shape[1]):
            z=self(z,actions[:,t])
            future.append(z)
        return torch.stack(future,1)
