# Primary papers and implementation references

1. [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243). Masked target-block feature prediction and EMA teacher.
2. [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). Predict space-time target representations, not future RGB frames.
3. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). Video pretraining and action-conditioned modeling.
4. [Official V-JEPA](https://github.com/facebookresearch/jepa) and [V-JEPA 2](https://github.com/facebookresearch/vjepa2). See primary code for real transformer masking, position encodings and training details.
5. [DreamerV3](https://arxiv.org/abs/2301.04104). Contrast pixel/world-model reconstruction with representation-space JEPA objectives.
6. [TD-MPC2](https://arxiv.org/abs/2310.16828). Compare action-conditioned latent dynamics and model-predictive control.
7. [Model Predictive Control and CEM overview](https://arxiv.org/abs/1805.12114). Sample and optimize action trajectories.

## What this repository does NOT implement

- The official pretrained V-JEPA / V-JEPA 2 architectures or training data scale.
- True blockwise spatiotemporal context masking over video tubelets.
- Multi-view robot proprioception, real-world action calibration, or reward-model-guided planning.
- Accurate control of CarRacing or a photorealistic video generator.

This baseline uses next-frame patch target features and mean-pooled action-conditioned latent dynamics.
