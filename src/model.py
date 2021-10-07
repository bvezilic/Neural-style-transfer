from collections import OrderedDict
from typing import List, Dict, Callable

import torch.nn as nn
import torchvision.models as models
from torch import Tensor


class VGG19(nn.Module):
    def __init__(self,
                 content_layers: List[int],
                 style_layers: List[int]):
        super(VGG19, self).__init__()

        self.pretrained_vgg19 = models.vgg19(pretrained=True).features

        self.content_layers = content_layers
        self.style_layers = style_layers

        self.content_features = OrderedDict({})
        self.style_features = OrderedDict({})

        # Register hooks to obtain various inputs
        for idx, module in enumerate(self.pretrained_vgg19.children()):
            if idx in self.content_layers:
                module.register_forward_hook(self._get_content_activation(idx))

            if idx in self.style_layers:
                module.register_forward_hook(self._get_style_activation(idx))

    def _get_content_activation(self, idx: int) -> Callable:
        def hook(module, input, output) -> None:
            self.content_features[idx] = output

        return hook

    def _get_style_activation(self, idx: int) -> Callable:
        def hook(module, input, output) -> None:
            self.style_features[idx] = output

        return hook

    def clear_features(self) -> None:
        """
        Clears stored features (outputs of conv layers) from hooks.
        """
        self.content_features = OrderedDict({})
        self.style_features = OrderedDict({})

    @staticmethod
    def clone_features(features: Dict[int, Tensor]) -> Dict[int, Tensor]:
        """
        Clones all tensors in dictionary.

        Args:
            features (dict): Dictionary of tensors with string keys.

        Returns:
            New dictionary with cloned tensors.
        """
        return {idx: feature.clone() for idx, feature in features.items()}

    def forward(self, input_image: Tensor) -> (Tensor, Dict[int, Tensor], Dict[int, Tensor]):
        """
        Runs forward pass through pretrained vgg19.

        Args:
            input_image (Tensor): Normalized image of shape (B, C, H, W)

        Returns:
            Outputs of conv layers picked up by hooks.

            x (Tensor):
            content_features (dict):
            style_features (dict):
        """
        x = self.pretrained_vgg19(input_image)

        content_features = self.clone_features(self.content_features)
        style_features = self.clone_features(self.style_features)

        self.clear_features()

        return x, content_features, style_features
