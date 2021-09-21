from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class TotalLoss(nn.Module):
    def __init__(self,
                 content_features: List[Tensor],
                 style_features: List[Tensor],
                 alpha: float = 1.,
                 beta: float = 1000.):
        super(TotalLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.content_losses = [ContentLoss(feature) for feature in content_features]
        self.style_losses = [StyleLoss(feature) for feature in style_features]

    def forward(self, input_content_features: List[Tensor], input_style_features: List[Tensor]) -> Tensor:

        assert len(self.content_losses) == len(input_content_features), \
            f"Mismatched lengths of content features: Expected {len(self.content_losses)} got {len(input_content_features)}"
        assert len(self.style_losses) == len(input_style_features), \
            f"Mismatched lengths of style features: Expected {len(self.style_losses)} got {len(input_style_features)}"

        total_content_loss = 0
        total_style_loss = 0

        for content_loss, input_content_feature in zip(self.content_losses, input_content_features):
            total_content_loss += content_loss(input_content_feature)

        for style_loss, input_style_feature in zip(self.style_losses, input_style_features):
            total_style_loss += style_loss(input_style_feature) / len(input_style_features)  # Normalization factor W

        total_loss = self.alpha * total_content_loss + self.beta * total_style_loss
        return total_loss


class ContentLoss(nn.Module):
    def __init__(self, content_feature: Tensor):
        super(ContentLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.register_buffer('content_feature', content_feature.detach())

    def forward(self, input_features: Tensor) -> Tensor:
        return self.mse(input_features, self.content_feature)


class StyleLoss(nn.Module):
    def __init__(self, style_feature: Tensor, normalize_gram_matrix=False):
        super(StyleLoss, self).__init__()

        self.normalize_gram_matrix = normalize_gram_matrix
        self.mse = nn.MSELoss()
        self.register_buffer('style_gram', gram_matrix(style_feature.detach(), normalize=self.normalize_gram_matrix))

    def forward(self, input_feature: Tensor) -> Tensor:
        input_gram = gram_matrix(input_feature, normalize=self.normalize_gram_matrix)
        return self.mse(input_gram, self.style_gram)


def gram_matrix(feature_maps: Tensor, normalize: bool = False) -> Tensor:
    B, C, H, W = feature_maps.size()

    assert B == 1, f"Batch size must be 1! Got B={B}"

    feature_maps = feature_maps.squeeze(0)  # Remove batch_size
    features = feature_maps.view(C, H * W)
    g = torch.mm(features, features.t())

    if normalize:
        g = g / g.numel()

    return g
