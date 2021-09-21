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


@click.command()
@click.option('-content', '--content_image_path', type=click.Path(), help='Path to content image')
@click.option('-style', '--style_image_path', type=click.Path(), help='Path to content image')
@click.option('-p', '--params', type=click.Path(), help='Path to params.yaml file')
def run_neural_transfer(
        content_image_path: str,
        style_image_path: str,
        params: str,
) -> None:
    """
    Runs neural style transfer based on provided content and style image.

    Args:
        content_image_path (str): Path to content image.
        style_image_path (str): Path to style image.
        params (str): Path to params.yaml file.

    Returns:
        None - Outputs generated.jpg image to images/* directory.
    """
    # LOAD PARAMS
    with open(params, 'r') as fp:
        params = yaml.safe_load(fp)

    # LOAD IMAGES
    print("Loading images...")
    images = {
        'content_image': Image.open(content_image_path),
        'style_image': Image.open(style_image_path),
        'input_image': Image.open(content_image_path),  # Use content image as starting point
    }

    # PREPROCESS
    mean = params.get('preprocess').get('mean')
    std = params.get('preprocess').get('std')
    resize = params.get('preprocess').get('resize')

    preprocess = Compose([
        ToTensor(),
        Resize(size=resize),
        Normalize(mean=mean, std=std),
        Lambda(lambda x: x.unsqueeze(0)),  # Add batch size
    ])
    images = {name: preprocess(image) for name, image in images.items()}

    # MODEL
    content_layers = params.get('vgg19').get('content_layers')
    style_layers = params.get('vgg19').get('style_layers')

    model = VGG19(
        content_layers=content_layers,
        style_layers=style_layers,
    )
    print(model)

    # DEVICE SETUP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()  # Set immediately to eval mode
    images = {name: image.to(device) for name, image in images.items()}

    # LOSS FUNCTION
    with torch.no_grad():
        _, content_features, _ = model(images['content_image'])
        _, _, style_features = model(images['style_image'])

    alpha = params.get('train').get('loss').get('alpha')
    beta = params.get('train').get('loss').get('beta')
    loss_criterion = TotalLoss(
        content_features=content_features.values(),
        style_features=style_features.values(),
        alpha=alpha,
        beta=beta,
    )

    # OPTIMIZER
    images['input_image'].requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.Adam(params=[images['input_image']])

    # FINAL RUN
    iterations = params.get('train').get('iterations')
    for i in tqdm.tqdm(range(iterations)):
        # Forward pass
        _, content_features, style_features = model(images['input_image'])
        # Compute loss
        loss = loss_criterion(
            input_content_features=content_features.values(),
            input_style_features=style_features.values()
        )
        # Compute gradients
        loss.backward()
        # Update input image
        optimizer.step()
        # Clear gradients
        optimizer.zero_grad()

        if i % 50:
            print(f"Iteration: {i+1}, Loss: {loss.item()}")

    # SAVE GENERATED IMAGE
    postprocessing = Compose([
        Denormalize(mean=mean, std=std),
        Lambda(lambda x: x.squeeze(0)),  # Removes batch dim
        Lambda(lambda x: x.clamp(0, 1)),
        ToPILImage()
    ])

    final_image = postprocessing(images['input_image'])
    final_image.save('images/generated.jpg')


if __name__ == '__main__':
    run_neural_transfer()
