#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from torch import Tensor
import torch
import torch.nn.functional as F
from einops import rearrange

def center_aligned_scale_invariant_loss(
        pred_depth_map: Tensor,
        target_depth_map: Tensor,
        mask: Tensor,
        channel_dependent: bool,
        eps: float = 1e-8,
):
    # The registring always considers all channels (all objects).
    aligned_pred_depth_map = register_image(
        moving_image=pred_depth_map,
        fixed_image=target_depth_map,
        mask=mask,
        channel_dependent=False,
        eps=eps,
    )
    return scale_invariant_loss(
        pred_depth_map=aligned_pred_depth_map,
        target_depth_map=target_depth_map,
        mask=mask,
        channel_dependent=channel_dependent,
        eps=eps,
    )

def scale_invariant_loss(
        pred_depth_map: Tensor,
        target_depth_map: Tensor,
        mask: Tensor,
        channel_dependent: bool,
        eps: float = 1e-8,
):
    """
    Scale-invariant loss for depth maps.

    Args:
        pred_depth_map (Tensor): Predicted depth map. (B, C, ...)
        target_depth_map (Tensor): Target depth map. (B, C, ...)
        mask (Tensor): Mask to apply on the depth maps. (B, C, ...)
        eps (float): Small value to avoid division by zero.

    Returns:
        Tensor: Computed scale-invariant loss.
    """
    if pred_depth_map.ndim < 3:
        raise RuntimeError(
            f"Expected at least 3 dimensions, got {pred_depth_map.ndim}."
        )
    if pred_depth_map.shape != target_depth_map.shape:
        raise RuntimeError(
            f"Shape mismatch: {pred_depth_map.shape} vs {target_depth_map.shape}"
        )
    if pred_depth_map.shape != mask.shape:
        raise RuntimeError(
            f"Shape mismatch: {pred_depth_map.shape} vs {mask.shape}")

    if channel_dependent:
        agg_dim = tuple(range(2, mask.ndim))
        agg_pattern = 'b c ... -> b c (...)'
    else:
        agg_dim = tuple(range(1, mask.ndim))
        agg_pattern = 'b ... -> b (...)'

    with torch.no_grad():
        # (B, C, 1) or (B, 1)
        T = torch.sum(
            mask.to(pred_depth_map.dtype), dim=agg_dim,
        ).unsqueeze_(-1) + eps
        mask = mask.to(torch.bool)
        mask = rearrange(mask, agg_pattern)
        target_depth_map = rearrange(
            target_depth_map, agg_pattern)

    pred_depth_map = rearrange(
        pred_depth_map, agg_pattern)
    g = torch.zeros_like(pred_depth_map)
    g[mask] = (
            torch.log(F.relu(pred_depth_map[mask]) + eps)
            - torch.log(target_depth_map[mask] + eps)
    )

    term2 = torch.square(torch.sum(g / T, dim=-1))
    term1 = torch.sum(torch.square(g) / T, dim=-1) - term2
    si_loss = term1 + 0.15 * term2
    with torch.no_grad():
        si_loss.clamp_(eps)
    si_loss = torch.sqrt(si_loss)
    return si_loss.mean()

def register_image(
        moving_image: Tensor,
        fixed_image: Tensor,
        mask: Tensor,
        channel_dependent: bool,
        eps: float = 1e-8,
):
    if moving_image.ndim < 3:
        raise RuntimeError(
            f"Expected at least 3 dimensions, got {moving_image.ndim}."
        )
    if moving_image.shape != fixed_image.shape:
        raise RuntimeError(
            f"Shape mismatch: {moving_image.shape} vs {fixed_image.shape}"
        )
    if moving_image.shape != mask.shape:
        raise RuntimeError(
            f"Shape mismatch: {moving_image.shape} vs {mask.shape}")

    if channel_dependent:
        agg_dim = tuple(range(2, mask.ndim))
    else:
        agg_dim = tuple(range(1, mask.ndim))

    mask = mask.to(moving_image.dtype)
    T = torch.sum(mask, dim=agg_dim, keepdim=True) + eps  # (B, 1, 1, 1)
    diff = (torch.mul(fixed_image, mask) - torch.mul(moving_image, mask))
    shift = torch.sum(torch.div(diff, T), dim=agg_dim, keepdim=True)
    return moving_image + shift


@torch.no_grad()
def main():
    shape = (2, 3, 32, 32)
    pred_depth_map = torch.exp(torch.rand(shape, dtype=torch.float32))
    target_depth_map = torch.exp(torch.rand(shape, dtype=torch.float32))

    # Mask for indicating valid pixels
    mask = torch.ones(shape, dtype=torch.bool)
    mask[:, :, 10:20, 10:20] = False

    loss = scale_invariant_loss(
        pred_depth_map=pred_depth_map,
        target_depth_map=target_depth_map,
        mask=mask,
        channel_dependent=False,
    )
    print(
        f'Scale-invariant independent loss: '
        f'{loss.item()}'
    )

    loss = scale_invariant_loss(
        pred_depth_map=pred_depth_map,
        target_depth_map=target_depth_map,
        mask=mask,
        channel_dependent=True,
    )
    print(
        f'Scale-invariant dependent loss: '
        f'{loss.item()}'
    )

    """
        When using center-aligned scale-invariant (CASI) loss,
        the model can not estimate the shift.
        Thus, to perform 3D reconstruction from the depth map,
        we need to either assume a fixed shift 
        or calculate the shift from other prediction.
        For example, we can train another model using SI loss and calculate the predicted shift (mean of predicted depth).
    """
    loss = center_aligned_scale_invariant_loss(
        pred_depth_map=pred_depth_map,
        target_depth_map=target_depth_map,
        mask=mask,
        channel_dependent=False,
    )
    print(
        f'Center-aligned Scale-invariant independent loss: '
        f'{loss.item()}'
    )

    loss = center_aligned_scale_invariant_loss(
        pred_depth_map=pred_depth_map,
        target_depth_map=target_depth_map,
        mask=mask,
        channel_dependent=True,
    )
    print(
        f'Center-aligned Scale-invariant dependent loss: '
        f'{loss.item()}'
    )


if __name__ == '__main__':
    main()