import torch


class Denormalize(object):
    """Denormalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will denormalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] * std[channel]) + mean[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return denormalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def denormalize(tensor, mean, std, inplace=False):
    """
    Denormalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will denormalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] * std[channel]) + mean[channel]``

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace (bool): Perform operation on same tensor.

    Returns:
        Tensor: Denormalized Tensor image.
    """
    assert isinstance(tensor, torch.Tensor), f"Input tensor should be a torch tensor. Got {type(tensor)}"
    assert tensor.is_floating_point(), f"Input  tensor should be a float tensor. Get {tensor.dtype}"
    assert tensor.ndim >= 3, f"Expected tensor to be a tensor image of size (..., C, H, W), got " \
                            f"tensor.size()={tensor.size()}"

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    tensor.mul_(std).add_(mean)
    return tensor

