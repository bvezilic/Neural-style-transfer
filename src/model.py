from collections import OrderedDict
from typing import List, Dict, Callable

import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from src.utils import clone_tensors


class VGG19(nn.Module):
    """
    Pretrained VGG19. Uses hooks to extract outputs from different layers.

    Args:
        content_layers_ids (list): List of layer ids from which to extract output
        style_layers_ids (list): List of layer ids from which to extract output
    """

    def __init__(self, content_layers_ids: List[int], style_layers_ids: List[int]):
        super(VGG19, self).__init__()

        self.pretrained_vgg19 = models.vgg19(pretrained=True).features

        self.content_layers_ids = content_layers_ids
        self.style_layers_ids = style_layers_ids

        # Cache attributes for temporary storing conv layers output (tensor)
        self._content_features = OrderedDict({})
        self._style_features = OrderedDict({})

        # Register hooks to obtain various outputs
        for idx, module in enumerate(self.pretrained_vgg19.children()):
            if idx in self.content_layers_ids:
                module.register_forward_hook(self._get_content_activation(idx))

            if idx in self.style_layers_ids:
                module.register_forward_hook(self._get_style_activation(idx))

    def _get_content_activation(self, idx: int) -> Callable:
        """
        Stores i-th layer activation to `self._content_features` dictionary.

        Args:
            idx (int): VGG19 layer index

        Returns:
            Callable: PyTorch hook function
        """

        def hook(module, input, output) -> None:
            self._content_features[idx] = output

        return hook

    def _get_style_activation(self, idx: int) -> Callable:
        """
        Stores i-th layer activation to `self._style_features` dictionary.

        Args:
            idx (int): VGG19 layer index

        Returns:
            Callable: PyTorch hook function
        """

        def hook(module, input, output) -> None:
            self._style_features[idx] = output

        return hook

    def clear_features(self) -> None:
        """
        Clears stored features (outputs of conv layers).
        """
        self._content_features = OrderedDict({})
        self._style_features = OrderedDict({})

    def forward(self, input_image: Tensor) -> (Tensor, Dict[int, Tensor], Dict[int, Tensor]):
        """
        Runs forward pass through pretrained vgg19.

        Args:
            input_image (Tensor): Normalized image of shape (B, C, H, W)

        Returns:
            Outputs of conv layers picked up by hooks.

            x (Tensor): Output of last feature layer from vgg19. Tensor of shape (B, 512, 8, 8)
            content_features (dict): Outputs of conv layers at indices (keys of dicts)
            style_features (dict): Outputs of conv layers at indices (keys of dicts)
        """
        x = self.pretrained_vgg19(input_image)

        content_features = clone_tensors(self._content_features)
        style_features = clone_tensors(self._style_features)

        self.clear_features()

        return x, content_features, style_features
