from pathlib import Path

import click
import torch.cuda
import torch.optim as optim
import tqdm
import yaml
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Lambda, ToPILImage

from src.losses import TotalLoss
from src.model import VGG19
from src.transforms import Denormalize
from src.utils import tensors_to_float, save_json


@click.command()
@click.option('-p', '--params', type=click.Path(),
              help='Path to params.yaml file')
@click.option('-content', '--content_image_path', type=click.Path(),
              help='Path to content image.')
@click.option('-style', '--style_image_path', type=click.Path(),
              help='Path to style image.')
@click.option('-o', '--output_image_path', type=click.Path(),
              help='Path to output (generated) image.')
@click.option('-l', '--losses_path', type=click.Path(),
              help='Path to losses.json file.')
def run_neural_transfer(
        content_image_path: str,
        style_image_path: str,
        output_image_path: str,
        losses_path: str,
        params: str,
) -> None:
    """
    Runs neural style transfer based on provided content and style image.

    Args:
        content_image_path (str): Path to content image.
        style_image_path (str): Path to style image.
        output_image_path (str): Save location of generated image.
        losses_path (str): Save location for losses.json file.
        params (str): Path to params.yaml file.

    Returns:
        None: Outputs generated image to `output_image_path`.
    """
    # LOAD PARAMS
    with open(params, 'r') as fp:
        params = yaml.safe_load(fp)

    # LOAD IMAGES
    print(f"Loading images...\n"
          f"Content image: {content_image_path}\n"
          f"Style image: {style_image_path}\n")
    images = {
        'content_image': Image.open(content_image_path),
        'style_image': Image.open(style_image_path),
        'input_image': Image.open(content_image_path),  # Use content image as starting point
    }

    # PREPROCESS
    mean = params['preprocess'].get('mean')
    std = params['preprocess'].get('std')
    resize = params['preprocess'].get('resize')

    preprocess = Compose([
        ToTensor(),
        Resize(size=resize),
        Normalize(mean=mean, std=std),
        Lambda(lambda x: x.unsqueeze(0)),  # Add batch size
    ])
    images = {name: preprocess(image) for name, image in images.items()}

    # MODEL
    content_layers_ids = params['vgg19'].get('content_layers_ids')
    style_layers_ids = params['vgg19'].get('style_layers_ids')

    model = VGG19(
        content_layers_ids=content_layers_ids,
        style_layers_ids=style_layers_ids,
    )

    # DEVICE SETUP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()  # Set immediately to eval mode
    images = {name: image.to(device) for name, image in images.items()}

    # LOSS FUNCTION
    with torch.no_grad():
        _, content_features, _ = model(images['content_image'])
        _, _, style_features = model(images['style_image'])

    alpha = params['train']['loss'].get('alpha')
    beta = params['train']['loss'].get('beta')
    normalize_gram_matrix = params['train']['loss'].get('normalize_gram_matrix', False)

    loss_criterion = TotalLoss(
        content_features=content_features,
        style_features=style_features,
        alpha=alpha,
        beta=beta,
        normalize_gram_matrix=normalize_gram_matrix
    )

    # OPTIMIZER
    images['input_image'].requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS(params=[images['input_image']])

    # FINAL RUN
    iterations = params['train'].get('iterations')
    all_losses = {'losses': []}
    for i in tqdm.tqdm(range(iterations)):
        def closure():
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            _, content_features, style_features = model(images['input_image'])

            # Compute loss
            losses = loss_criterion(
                input_content_features=content_features,
                input_style_features=style_features
            )
            total_loss = losses['total_loss']

            # Compute gradients
            total_loss.backward()

            # Log losses
            log_losses = tensors_to_float(losses)
            all_losses['losses'].append(log_losses)

            return total_loss

        optimizer.step(closure=closure)
        print(f"Iteration: {i * optimizer.defaults['max_iter']}, Loss: {all_losses['losses'][-1]['total_loss']}")

    # Log all losses to json
    save_json(all_losses, losses_path)

    # POSTPROCESSING
    postprocessing = Compose([
        Denormalize(mean=mean, std=std),
        Lambda(lambda x: x.squeeze(0)),  # Removes batch dim
        Lambda(lambda x: x.clamp(0, 1)),
        ToPILImage()
    ])
    final_image = postprocessing(images['input_image'])  # final_image: PIL.Image

    # SAVE GENERATED IMAGE
    save_path = Path(output_image_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(save_path)
    print(f"Generated image saved to '{save_path}'")


if __name__ == '__main__':
    run_neural_transfer()
