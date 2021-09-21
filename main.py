import torch.cuda
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Lambda, ToPILImage
import torch.optim as optim

import tqdm
from src.losses import TotalLoss
from src.model import VGG19
from src.transforms import Denormalize
from config import IMAGE_DIR


def run_neural_transfer(
        content_image_path: str,
        style_image_path: str,
):
    # LOAD IMAGES
    print("Loading images...")
    images = {
        'content_image': Image.open(content_image_path),
        'style_image': Image.open(style_image_path),
        'input_image': Image.open(content_image_path),  # Use content image as starting point
    }

    # PREPROCESS
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_size = (256, 256)

    preprocess = Compose([
        ToTensor(),
        Resize(size=new_size),
        Normalize(mean=mean, std=std),
        Lambda(lambda x: x.unsqueeze(0)),  # Add batch size
    ])
    images = {name: preprocess(image) for name, image in images.items()}

    # MODEL
    content_layers = [
        21   # conv4_2
    ]
    style_layers = [
        0,   # conv1_1
        5,   # conv2_1
        10,  # conv3_1
        19,  # conv4_1
        28,  # conv5_1
    ]
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
    _, content_features, _ = model(images['content_image'])
    _, _, style_features = model(images['style_image'])
    loss_criterion = TotalLoss(
        content_features=content_features.values(),
        style_features=style_features.values(),
        alpha=1.,
        beta=1000
    )

    # OPTIMIZER
    images['input_image'].requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.Adam(params=[images['input_image']])

    # FINAL RUN
    iterations = 1000
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

    # SAVE INPUT IMAGE
    postprocessing = Compose([
        Denormalize(mean=mean, std=std),
        Lambda(lambda x: x.squeeze(0)),  # Removes batch dim
        Lambda(lambda x: x.clamp(0, 1)),
        ToPILImage()
    ])

    final_image = postprocessing(images['input_image'])
    final_image.save('images/generated.jpg')


if __name__ == '__main__':
    run_neural_transfer(
        content_image_path=IMAGE_DIR / 'dancing.jpg',
        style_image_path=IMAGE_DIR / 'picasso.jpg'
    )
