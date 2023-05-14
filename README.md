# Neural-style-transfer

PyTorch's implementation of Neural Style Transfer. Learning repository with several experiments and playing around.

## Usage

### Gradio app

Run demo app with `python app.py`

## Quick overview

![neural-style-transfer](https://user-images.githubusercontent.com/16206648/139554170-1c63cd04-c83f-4ea8-bb12-da05b87d5c9a.jpg)

## Experiments

Summary of experiments are given below. All experiments will be on their respective branches and reproducable. :)

### Gram matrix normalization

Some of the online resources didn't take gram matrix normalization into consideration, not even paper mentions it.
However, pytorch tutorial says it's really important, so let's compare the results.

### Low losses in later layers

I've noticed in experiments that style losses in later layers are really low. Could it be that the reason for that is
that I am using content image as starting point for generation?

### Adam vs LBFGS optimizer

Some are using Adam (probably due to simplicity), others say that LBFGS produces much better results, so let's compare
them.
