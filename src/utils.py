from typing import Dict

import torch


def clone_tensors(tensors: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """
    Clones all tensors in dictionary.

    Args:
        tensors (dict): Dictionary of tensors with string keys.

    Returns:
        New dictionary with cloned tensors.
    """
    return {idx: tensor.clone() for idx, tensor in tensors.items()}
