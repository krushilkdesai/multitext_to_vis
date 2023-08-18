# !pip install googletrans==3.1.0a0
# !pip install --upgrade diffusers transformers -q

from googletrans import Translator
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import os
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



# commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")




def get_translation(text,dest_lang):
  translator = Translator()
  translated_text = translator.translate(text, dest=dest_lang)
  return translated_text.text


class CFG:
    # device = "cpu"
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)


    # generator = torch.Generator(device='cpu')
    # dataset = MNIST(root=".", download=True, transform=ToTensor())
    # dt = DataLoader(dataset, batch_size=8, generator=generator, shuffle=True)


    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (900,900)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12


image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)


def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image


translation = get_translation("india flag","en")
generate_image(translation, image_gen_model)

