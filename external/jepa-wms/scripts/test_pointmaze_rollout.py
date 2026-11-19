
"""
JEPA-WM PointMaze — Latent Rollout Smoke Test

Tests:
1. Load official pretrained JEPA-WM.
2. Encode RGB observations and proprioceptive states.
3. Align visual context and action conditioning.
4. Perform one-step action-conditioned latent rollout.
5. Validate tensor shapes and numerical stability.

Synthetic inputs are used only for interface testing.
This is NOT an official PointMaze planning evaluation.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict


def main():
    # ============================================
    # 1. Configuration
    # ============================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 1
    OBS_FRAMES = 2
    IMAGE_SIZE = 224
    PROPRIO_DIM = 4
    HORIZON = 1

    torch.manual_seed(42)

    print("\n========== JEPA-WM PointMaze Smoke Test ==========")
    print("Device:", device)

    # ============================================
    # 2. Load pretrained model
    # ============================================
    print("\n[1] Loading official JEPA-WM...")

    model, _ = torch.hub.load(
        ".",
        "jepa_wm_pointmaze",
        source="local",
    )

    model = model.to(device).eval()

    print("Model:", type(model).__name__)
    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    print("Context window:", model.ctxt_window)
    print("Action dimension:", model.action_dim)

    # ============================================
    # 3. Create synthetic observations
    # ============================================
    print("\n[2] Creating observations...")

    # RGB images: [B, T, C, H, W]
    # model.encode() expects pixel values in [0, 255].
    visual = torch.randint(
        low=0,
        high=256,
        size=(
            BATCH_SIZE,
            OBS_FRAMES,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=torch.uint8,
        device=device,
    )

    # PointMaze proprio: [B, T, 4]
    proprio = torch.zeros(
        BATCH_SIZE,
        OBS_FRAMES,
        PROPRIO_DIM,
        dtype=torch.float32,
        device=device,
    )

    obs = {
        "visual": visual,
        "proprio": proprio,
    }

    print("Visual observation:", visual.shape)
    print("Proprio observation:", proprio.shape)

    # ============================================
    # 4. Encode observations
    # ============================================
    print("\n[3] Encoding observations...")

    with torch.no_grad():
        z_full = model.encode(obs)

    print("Full visual latent:", z_full["visual"].shape)
    print("Full proprio latent:", z_full["proprio"].shape)

    # ============================================
    # 5. Align context with one action
    # ============================================
    print("\n[4] Preparing one-frame context...")

    # The current AdaLN predictor requires the
    # number of visual timesteps and action
    # conditioning timesteps to match.
    #
    # We use the last observation as the context
    # for this one-step smoke test.
    #
    # Keep TensorDict type so model.unroll()
    # returns both visual and proprio features.

    z_context = TensorDict(
        {
            "visual": z_full["visual"][:, -1:],
            "proprio": z_full["proprio"][:, -1:],
        },
        batch_size=[BATCH_SIZE, 1],
        device=device,
    )

    print("Context visual:", z_context["visual"].shape)
    print("Context proprio:", z_context["proprio"].shape)

    assert z_context["visual"].shape[1] == 1
    assert z_context["proprio"].shape[1] == 1

    # ============================================
    # 6. Create action sequence
    # ============================================
    print("\n[5] Creating action sequence...")

    # Shape: [Horizon, Batch, ModelActionDim]
    #
    # PointMaze has 2-dimensional raw actions.
    # The loaded model expects 10-dimensional
    # packed action features.
    #
    # Zero action is only used for smoke testing.

    actions = torch.zeros(
        HORIZON,
        BATCH_SIZE,
        model.action_dim,
        dtype=torch.float32,
        device=device,
    )

    print("Actions:", actions.shape)

    # ============================================
    # 7. Latent rollout
    # ============================================
    print("\n[6] Running action-conditioned rollout...")

    with torch.no_grad():
        predicted = model.unroll(
            z_context,
            act_suffix=actions,
        )

    pred_visual = predicted["visual"]
    pred_proprio = predicted["proprio"]

    print("Predicted visual:", pred_visual.shape)
    print("Predicted proprio:", pred_proprio.shape)

    # model.unroll() returns time-major tensors:
    # [context + horizon, B, ...]

    expected_steps = 1 + HORIZON

    assert pred_visual.shape[0] == expected_steps
    assert pred_visual.shape[1] == BATCH_SIZE

    assert pred_proprio.shape[0] == expected_steps
    assert pred_proprio.shape[1] == BATCH_SIZE

    # ============================================
    # 8. Validate numerical stability
    # ============================================
    print("\n[7] Checking numerical stability...")

    assert torch.isfinite(pred_visual).all(), (
        "Visual rollout contains NaN or Inf"
    )

    assert torch.isfinite(pred_proprio).all(), (
        "Proprio rollout contains NaN or Inf"
    )

    print("Visual numerical stability: PASS")
    print("Proprio numerical stability: PASS")

    # ============================================
    # 9. Inspect predicted latent
    # ============================================
    print("\n[8] Inspecting latent transition...")

    # Current visual latent:
    # [B, V, H, W, D]
    z_current = z_context["visual"][:, -1]

    # Predicted next visual latent:
    # [B, V, H, W, D]
    z_next = pred_visual[-1]

    print("Current latent:", z_current.shape)
    print("Predicted next latent:", z_next.shape)

    # These metrics measure changes between
    # consecutive latent representations.
    #
    # They are NOT prediction accuracy metrics
    # because no ground-truth next observation
    # is provided.

    mse = F.mse_loss(z_next, z_current)

    cosine = F.cosine_similarity(
        z_next.flatten(1),
        z_current.flatten(1),
        dim=1,
    ).mean()

    print(f"Latent change MSE: {mse.item():.6f}")
    print(f"Latent cosine similarity: {cosine.item():.6f}")

    # ============================================
    # 10. GPU memory
    # ============================================
    if device == "cuda":
        print("\n[9] GPU memory...")

        allocated = (
            torch.cuda.memory_allocated() / 1024**3
        )
        reserved = (
            torch.cuda.memory_reserved() / 1024**3
        )
        peak = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        print(f"Allocated: {allocated:.3f} GiB")
        print(f"Reserved: {reserved:.3f} GiB")
        print(f"Peak allocated: {peak:.3f} GiB")

    # ============================================
    # 11. Summary
    # ============================================
    print("\n========== TEST SUMMARY ==========")
    print("Official pretrained model: PASS")
    print("Visual encoding: PASS")
    print("Proprio encoding: PASS")
    print("Action-conditioned rollout: PASS")
    print("Numerical stability: PASS")
    print("==================================")
    print("JEPA-WM PointMaze Smoke Test SUCCESS!")


if __name__ == "__main__":
    main()
