import json
from pathlib import Path
from typing import Dict, Union

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


def tensors_to_float(tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Converts all single-value tensors in dictionary to float values.

    Args:
        tensors (dict): Dictionary of tensors with single value.

    Returns:
        dict
    """
    assert isinstance(tensors, dict), f"Input argument must be dict! Found {type(tensors)}"

    output = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            output[k] = v.item()
        else:
            output[k] = v

    return output


def save_json(obj: Dict, filepath: Union[str, Path]) -> None:
    """
    Saves dict object to given path.

    Args:
        obj (dict): Object to serialize as json.
        filepath (str or Path): Save location for json file.

    Returns:
        None
    """
    with open(filepath, 'w') as fp:
        json.dump(obj, fp)
