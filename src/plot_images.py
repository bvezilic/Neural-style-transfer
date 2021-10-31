from typing import List

import click
from PIL import Image

from src.utils import plot_images


@click.command()
@click.option('-i', '--image', type=click.Path(), multiple=True, help='Path to image.')
@click.option('-o', '--output_plot', type=click.Path(), help='Save location of the plot.')
@click.option('-t', '--title', type=str, default='', help='Title of plot.')
def save_plot_images(image: List[str], output_plot: str, title: str) -> None:
    """
    Plots multiple images together in a grid and saves to file (.jpg).

    Args:
        image (str or Path): Path to image.
        output_plot (str or Path): Save location of the plot.
        title (str): Title of plot.

    Returns:
        None
    """
    pil_images = [Image.open(img) for img in image]
    fig = plot_images(pil_images, title=title)
    fig.savefig(output_plot)


if __name__ == '__main__':
    save_plot_images()
