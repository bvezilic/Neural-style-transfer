import gradio as gr
import torch.cuda
import torch.optim as optim
import yaml
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Lambda, ToPILImage

from src.losses import TotalLoss
from src.model import VGG19
from src.transforms import Denormalize

# LOAD PARAMS
with open('params.yaml', 'r') as fp:
    params = yaml.safe_load(fp)

mean = params['preprocess'].get('mean')
std = params['preprocess'].get('std')
resize = params['preprocess'].get('resize')

# PREPROCESSING
preprocess = Compose([
    ToTensor(),
    Resize(size=resize),
    Normalize(mean=mean, std=std),
    Lambda(lambda x: x.unsqueeze(0)),  # Add batch size
])

# POSTPROCESSING
postprocessing = Compose([
    Denormalize(mean=mean, std=std),
    Lambda(lambda x: x.squeeze(0)),  # Removes batch dim
    Lambda(lambda x: x.clamp(0, 1)),
    ToPILImage()
])

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


def run_style_transfer(content_image, style_image, iterations, alpha, beta, progress=gr.Progress()):
    # PRE-PROCESS
    content_image = preprocess(content_image).to(device)
    style_image = preprocess(style_image).to(device)
    generated_image = content_image.clone().to(device)

    # LOSS FUNCTION
    with torch.no_grad():
        _, content_features, _ = model(content_image)
        _, _, style_features = model(style_image)

    loss_criterion = TotalLoss(
        content_features=content_features,
        style_features=style_features,
        alpha=alpha,
        beta=beta,
        normalize_gram_matrix=True
    )

    # OPTIMIZER
    generated_image.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS(params=[generated_image])

    # FINAL RUN
    iterations = iterations
    for _ in progress.tqdm(range(iterations), desc="Applying style"):
        def closure():
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            _, content_features, style_features = model(generated_image)

            # Compute loss
            losses = loss_criterion(
                input_content_features=content_features,
                input_style_features=style_features
            )
            total_loss = losses['total_loss']

            # Compute gradients
            total_loss.backward()

            return total_loss

        optimizer.step(closure=closure)

    # POST-PROCESS
    generated_image = postprocessing(generated_image)

    return generated_image


examples = [
    ['images/Tuebingen_Neckarfront.jpg', 'images/Van_Gogh_Starry_Night.jpg', None, None, None],
    ['images/Tuebingen_Neckarfront.jpg', 'images/The_Scream.jpg', None, None, None],
    ['images/Vassily_Kandinsky_Composition_7.jpg', 'images/Tuebingen_Neckarfront.jpg', None, None, None],
]

# DEMO APP
demo = gr.Interface(fn=run_style_transfer,
                    inputs=[
                        gr.Image(shape=(224, 224), type="pil", label="Content Image"),
                        gr.Image(shape=(224, 224), type="pil", label="Style Image"),
                        gr.Number(value=10, label="Iterations", precision=0),
                        gr.Number(value=1., label="Alpha"),
                        gr.Number(value=1000., label="Beta")
                    ],
                    outputs=gr.Image(shape=(224, 224), type='pil', label="Output Image"),
                    examples=examples,
                    title="Style transfer")
demo.queue().launch()
