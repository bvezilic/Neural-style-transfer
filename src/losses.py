import torch
import torch.nn as nn
from torch import Tensor


class TotalLoss:
    """
    Computes total loss as described in paper: https://arxiv.org/abs/1508.06576

    ``total_loss = alpha * content_loss + beta * style_loss``

    Args:
        content_features (dict): Dictionary where key is layer index in vgg19 and value is feature maps obtained at that
                                 index for content image.
        style_features (dict): Dictionary where key is layer index in vgg19 and value is feature maps obtained at that
                               index for style image.
        alpha (float): Coefficient for content loss.
        beta (float): Coefficient for style loss.
        normalize_gram_matrix (bool): Default to True.
    """

    def __init__(self, content_features: dict[int, Tensor], style_features: dict[int, Tensor], alpha: float = 1.,
                 beta: float = 1000., normalize_gram_matrix: bool = True):
        super(TotalLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.normalize_gram_matrix = normalize_gram_matrix

        self.content_losses = {
            layer_idx: ContentLoss(feature) for layer_idx, feature in content_features.items()
        }
        self.style_losses = {
            layer_idx: StyleLoss(feature, normalize_gram_matrix=self.normalize_gram_matrix)
            for layer_idx, feature in style_features.items()
        }

    def __call__(self, input_content_features: dict[int, Tensor], input_style_features: dict[int, Tensor]) \
            -> dict[str, float | Tensor]:
        """
        Computes total loss for generated image.

        Args:
            input_content_features (dict): Dictionary where key is layer index in vgg19 and value is feature maps
                                           obtained at that index for generated image.
            input_style_features (dict): Dictionary where key is layer index in vgg19 and value is feature maps obtained
                                         at that index for generated image.

        Returns:
            Tensor: Sum of weighted style and content loss
        """
        assert len(self.content_losses) == len(input_content_features), \
            f"Mismatched lengths of content features: Expected {len(self.content_losses)} got {len(input_content_features)}"
        assert len(self.style_losses) == len(input_style_features), \
            f"Mismatched lengths of style features: Expected {len(self.style_losses)} got {len(input_style_features)}"

        losses = {
            'total_content_loss': 0,
            'total_style_loss': 0,
        }

        for content_layer_idx in input_content_features:
            content_loss = self.content_losses[content_layer_idx]
            input_content_feature = input_content_features[content_layer_idx]

            loss = content_loss(input_content_feature)
            losses['total_content_loss'] += loss
            losses[f'content_loss_{content_layer_idx}'] = loss.item()

        for style_layer_idx in input_style_features:
            style_loss = self.style_losses[style_layer_idx]
            input_style_feature = input_style_features[style_layer_idx]

            loss = style_loss(input_style_feature)
            losses['total_style_loss'] += loss
            losses[f'style_loss_{style_layer_idx}'] = loss.item()

        total_content_loss = losses['total_content_loss']
        total_style_loss = losses['total_style_loss'] / len(input_style_features)  # Normalization factor w

        losses['total_loss'] = self.alpha * total_content_loss + self.beta * total_style_loss

        return losses


class ContentLoss:
    """
    Computes content loss for one (conv) layer.

    Args:
        content_feature (Tensor): Feature map from i-th conv layer of content image.
    """

    def __init__(self, content_feature: Tensor):
        super(ContentLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.content_feature = content_feature.detach()

    def __call__(self, input_feature: Tensor) -> Tensor:
        """
        Computes mean-squared error between outputs of i-th layer from content image and generated image.

        Args:
            input_feature (Tensor): Feature maps from i-th conv layer of generated image.

        Returns:
            Tensor: Mean-squared error/loss
        """
        return self.mse(input_feature, self.content_feature)


class StyleLoss:
    """
    Computes style loss for one (conv) layer.

    Args:
        style_feature (Tensor): Feature maps from i-th conv layer of style image.
    """

    def __init__(self, style_feature: Tensor, normalize_gram_matrix=False):
        super(StyleLoss, self).__init__()

        self.normalize_gram_matrix = normalize_gram_matrix
        self.mse = nn.MSELoss()
        self.style_gram = gram_matrix(style_feature.detach(), normalize=self.normalize_gram_matrix)

    def __call__(self, input_feature: Tensor) -> Tensor:
        """
        Computes mean-squared error between:

        `style_feature`: Gram matrix of feature maps from i-th conv layer of style image
        and
        `input_feature`: Gram matrix of feature maps from i-th conv layer of generated image

        Args:
            input_feature: Feature maps from i-th conv layer of generated image.

        Returns:
            Tensor: Mean-squared error/loss
        """
        input_gram = gram_matrix(input_feature, normalize=self.normalize_gram_matrix)
        return self.mse(input_gram, self.style_gram)


def gram_matrix(feature_maps: Tensor, normalize: bool = False) -> Tensor:
    """
    Computes Gram matrix as inner product of feature maps (F) as F * F^T.

    Args:
        feature_maps (Tensor): Output tensor from conv layer of shape (B=batch, C=channel, H=height, W=width)
        normalize (bool): Normalize Gram matrix by its number of elements.

    Returns:
        Tensor: Gram matrix
    """
    assert isinstance(feature_maps, torch.Tensor), f"Input tensor should be a torch tensor. Got {type(feature_maps)}"
    assert feature_maps.is_floating_point(), f"Input  tensor should be a float tensor. Got {feature_maps.dtype}"
    assert feature_maps.ndim >= 3, f"Expected tensor to be a tensor image of size (..., C, H, W), got " \
                                   f"tensor.size()={feature_maps.size()}"

    if feature_maps.ndim == 4:
        B, C, H, W = feature_maps.size()
        assert B == 1, f'Batch size must be 1! Got B={B}'
    elif feature_maps.ndim == 3:
        C, H, W = feature_maps.size()

    features = feature_maps.view(C, H * W)
    g = torch.mm(features, features.t())

    if normalize:
        g = g / g.numel()

    return g
