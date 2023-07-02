from pathlib import Path
from typing import Callable

import PIL
import click
import torch.cuda
import torch.optim as optim
import tqdm
from PIL import Image

from src.losses import TotalLoss
from src.model import VGG19
from src.utils import tensors_to_float


@click.command()
@click.option('-content', '--content_image_path', type=click.Path(),
              help='Path to content image.')
@click.option('-style', '--style_image_path', type=click.Path(),
              help='Path to style image.')
@click.option('-ig', '--init_generated', type=click.Choice(['random', 'content'], case_sensitive=False),
              default='content', help='Initialized generated image from content image or random values.')
@click.option('-a', '--alpha', type=click.FLOAT, default=1.,
              help='Weight for content image.')
@click.option('-b', '--beta', type=click.FLOAT, default=1000.,
              help='Weight for style image.')
@click.option('-i', '--iterations', type=click.INT, default=10,
              help='Number of iterations.')
@click.option('--gram-norm/--no-gram-norm', type=click.BOOL, default=True,
              help='Flag whether to normalize gram matrix.')
@click.option('-o', '--output_image_path', type=click.Path(),
              help='Path to output (generated) image.')
def run_neural_transfer(
        content_image_path: str,
        style_image_path: str,
        init_generated: str,
        alpha: float,
        beta: float,
        iterations: int,
        gram_norm: bool,
        output_image_path: str,
) -> None:
    """
    Runs neural style transfer based on provided content and style image.

    Args:
        content_image_path (str): Path to content image.
        style_image_path (str): Path to style image.
        init_generated (str): Initialized generated image from content image or random values.
        alpha (float): Weight for content image.
        beta (float): Weight for style image.
        iterations (int): Number of iterations.
        gram_norm (bool): Flag whether to normalize gram matrix.
        output_image_path (str): Save location of generated image.

    Returns:
        None: Outputs generated image to `output_image_path`.
    """
    # Setup model for experiments
    content_layer_ids = [21]  # conv4_2
    style_layer_ids = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

    model = VGG19(
        content_layer_ids=content_layer_ids,
        style_layer_ids=style_layer_ids,
    )

    # Load the images
    content_image = PIL.Image.open(content_image_path)
    style_image = PIL.Image.open(style_image_path)

    # Run style transfer and plot the results
    losses, output_img = style_transfer(
        model=model,
        content_image=content_image,
        style_image=style_image,
        alpha=alpha,
        beta=beta,
        iterations=iterations,
        init_output=init_generated,
        normalize_gram_matrix=gram_norm
    )

    # Save image
    save_path = Path(output_image_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(save_path)
    print(f"Generated image saved to '{save_path}'")


def style_transfer(model: VGG19, content_image: PIL.Image, style_image: PIL.Image, alpha: float, beta: float,
                   iterations: int, init_output: str = 'content', normalize_gram_matrix: bool = True,
                   _progress_callback: Callable = None) -> (dict, PIL.Image):
    """
    Run style transfer.

    Args:
        model: VGG19 model for NST
        content_image: PIL.Image
        style_image: PIL.Image
        alpha: Weight for content image
        beta: Weight for style image
        iterations: Number of iterations
        init_output: Initialization for generated image. Random or from content image
        normalize_gram_matrix: Bool whether to normalize gram matrix
        _progress_callback: Internal callback function for updating progress bar

    Returns:
        losses: Dictionary with losses
        output_image: PIL.Image
    """
    # Setup device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Preprocess inputs
    content_tensor = model.preprocess(content_image)
    style_tensor = model.preprocess(style_image)

    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)

    # Obtain content and style features
    with torch.no_grad():
        _, content_features, _ = model(content_tensor)
        _, _, style_features = model(style_tensor)

    loss_criterion = TotalLoss(
        content_features=content_features,
        style_features=style_features,
        alpha=alpha,
        beta=beta,
        normalize_gram_matrix=normalize_gram_matrix
    )

    # Setup initial output image
    if init_output == "random":
        mean = torch.tensor(model.preprocess.mean)
        std = torch.tensor(model.preprocess.std)

        mean = mean.view(-1, 1, 1).expand_as(content_tensor)
        std = std.view(-1, 1, 1).expand_as(content_tensor)

        output_image = torch.normal(mean, std)
    elif init_output == "content":
        output_image = content_tensor.clone()
    else:
        raise ValueError(f"Unavailable option {init_output}! Must be either `random` or `content`.")

    output_image = output_image.contiguous().to(device)

    # Setup optimizer
    output_image.requires_grad_(True)
    optimizer = optim.LBFGS(params=[output_image])

    # FINAL RUN
    losses = []
    for i in tqdm.tqdm(range(iterations), desc="Running style transfer"):
        def closure():
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            _, content_features, style_features = model(output_image)

            # Compute loss
            loss = loss_criterion(
                input_content_features=content_features,
                input_style_features=style_features
            )
            total_loss = loss['total_loss']

            # Compute gradients
            total_loss.backward()

            # Save losses
            losses.append(tensors_to_float(loss))

            return total_loss

        optimizer.step(closure=closure)

        if _progress_callback:
            _progress_callback(step=i, total=iterations)

    # POST-PROCESSING
    output_image = model.postprocess(output_image)

    return losses, output_image


if __name__ == '__main__':
    run_neural_transfer()
