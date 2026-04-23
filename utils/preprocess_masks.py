import torch
from typing import List

def split_tensor_to_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    if len(tensor.shape) != 4 or tensor.shape[1] != 1 or tensor.shape[2] != 512 or tensor.shape[3] != 512:
        raise ValueError("Input tensor must have shape (X, 1, 512, 512).")
    split_tensors = torch.chunk(tensor, tensor.shape[0], dim=0)
    masks_pred = [t.squeeze(0) for t in split_tensors]
    return masks_pred

def discretize_mask(masks_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return torch.gt(masks_logits, threshold).float()

def assemble_raw_masks(raw_masks: List[torch.Tensor]) -> List[torch.Tensor]:
    masks = []
    for r_m in raw_masks:
        r_m = discretize_mask(r_m)
        r_m = torch.sum(r_m, dim=0, keepdim=True)
        masks.append(torch.clamp(r_m, max=1.0))
    return masks


def process_masks(masks_pred: torch.Tensor) -> List[torch.Tensor]:
    masks_pred = split_tensor_to_list(masks_pred[0])
    masks_pred = assemble_raw_masks(masks_pred)
    masks_pred = torch.stack(masks_pred, dim=0)
    combined_mask = torch.sum(masks_pred, dim=0, keepdim=True)
    masks_pred = [combined_mask]
    return masks_pred