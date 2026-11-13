# JEPA World Model experiment roadmap

### Phase 0 — reusable data
- [x] Load CarRacing episodes from world-model-lab, episode-level holdout, sequential windows
- [ ] Collect more varied demonstrations; five random episodes are insufficient for meaningful representation benchmarks

### Phase 1 — JEPA-inspired masked feature prediction
- [x] Small ViT patch encoder; EMA target encoder; masked future-patch predictor
- [x] Mask-only normalized feature regression
- [ ] Complete GPU training and inspect collapse / validation curves
- [ ] Proper V-JEPA spatiotemporal tubelet masks and blockwise context encoder

### Phase 2 — Action-conditioned JEPA
- [x] Frozen teacher features; MLP action-conditioned state transition
- [x] Multi-step rollout loss; per-horizon validation MSE
- [x] Latent trajectory PCA plot
- [ ] Action-conditioned token-level dynamics with richer video/state handling
- [ ] Strong baseline: MLP/no action vs action; varied seeds and datasets

### Phase 3 — latent planning
- [x] CEM latent-goal optimizer demo using provided goal embedding
- [ ] Learned task reward/cost, calibrated transition errors and closed-loop environment interactions
- [ ] Closed-loop planning success rate, comparisons to MPC/TD-MPC and random policies

### Phase 4 — further research
- [ ] Official V-JEPA 2 evaluation / adapters
- [ ] Multi-modal sensors and proprioception
- [ ] State uncertainty and temporal consistency tests

**Terminology:** This is a small JEPA-inspired experiment, not official V-JEPA 2-AC.
