import gradio as gr

from src.model import VGG19
from src.style_transfer import style_transfer

# LOAD MODEL
content_layer_ids = [21]  # conv4_2
style_layer_ids = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

model = VGG19(
    content_layer_ids=content_layer_ids,
    style_layer_ids=style_layer_ids,
)


def style_transfer_fn(content_image, style_image, iterations, alpha, beta, init_output, progress=gr.Progress()):
    def progress_callback(step, total):
        progress((step, total))

    loss, output_img = style_transfer(
        model=model,
        content_image=content_image,
        style_image=style_image,
        alpha=alpha,
        beta=beta,
        iterations=iterations,
        init_output=init_output,
        normalize_gram_matrix=True,
        _progress_callback=progress_callback
    )

    return output_img


# Add examples to Demo app
examples = [
    ['images/Tuebingen_Neckarfront.jpg', 'images/Van_Gogh_Starry_Night.jpg', 10, 1., 1000., "content"],
    ['images/Tuebingen_Neckarfront.jpg', 'images/The_Scream.jpg', 10, 1., 1000., "content"],
    ['images/Tuebingen_Neckarfront.jpg', 'images/Vassily_Kandinsky_Composition_7.jpg', 10, 1., 1000., "content"],
]

# DEMO APP
demo = gr.Interface(
    fn=style_transfer_fn,
    description="""
    # Neural Style Transfer
    
    Demo app for NST (Neural Style Transfer).
    """,
    inputs=[
        gr.Image(shape=(224, 224), type="pil", label="Content Image"),
        gr.Image(shape=(224, 224), type="pil", label="Style Image"),
        gr.Number(value=10, label="Iterations", precision=0,
                  info="Number of iterations on LBFGS optimizer"),
        gr.Number(value=1., label="Alpha", info="Content weight"),
        gr.Number(value=1000., label="Beta", info="Style weight"),
        gr.Radio(choices=["random", "content"], value="content", label="Initialization",
                 info="Starting point of output image", )
    ],
    outputs=gr.Image(shape=(224, 224), type='pil', label="Output Image"),
    examples=examples,
    title="Style transfer"
)
demo.queue().launch()
