from pathlib import Path

import click
import requests

content_images_url = {
    'Tuebingen_Neckarfront_Andreas_Praefcke.jpg': "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Tuebingen_Neckarfront.jpg/640px-Tuebingen_Neckarfront.jpg"
}

style_images_url = {
    'Shipwreck_of_the_Minotaur_William_Turner.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Shipwreck_of_the_Minotaur_William_Turner.jpg/640px-Shipwreck_of_the_Minotaur_William_Turner.jpg',
    'Starry_Night_Van_Gogh.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/606px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
    'The_Scream.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/603px-The_Scream.jpg',
    'Femme_nue_assise_Picasso.jpg': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg/589px-Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg',
    'Composition_7_Vassily_Kandinsky.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/640px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
}


@click.command()
@click.option('-t', '--target_dir', type=click.Path(), help='Path to target/save directory')
def download_images(target_dir: str) -> None:
    target_dir = Path(target_dir)
    print(f"Target (save) directory set to: {target_dir}")

    for name, url in content_images_url.items():
        print(f"Downloading image `{name}` on URL: {url}")
        response = requests.get(url)
        (target_dir / name).write_bytes(response.content)

    for name, url in style_images_url.items():
        print(f"Downloading image `{name}` on URL: {url}")
        response = requests.get(url)
        (target_dir / name).write_bytes(response.content)


if __name__ == '__main__':
    download_images()
