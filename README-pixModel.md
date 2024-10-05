# PixModel.ipynb

## Overview

`PixModel.ipynb` demonstrates a pipeline for transforming simple sketches into high-resolution images using various machine learning techniques. The notebook includes functionalities for generating sketches, converting them into detailed images using Stable Diffusion, enhancing the images using super-resolution techniques, and further improving them with Real-ESRGAN.

## Setup

### Dependencies

Install the required Python packages:

```bash
!pip install --upgrade diffusers transformers accelerate
!pip install torch torchvision diffusers transformers quickdraw scipy ftfy accelerate xformers
!pip install --upgrade torch torchvision


## Import Libraries

import torch
import numpy as np
from PIL import Image, ImageDraw
from quickdraw import QuickDrawData
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms

## Setup Environment
### Check if running in Google Colab and set up device:
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Initialize Stable Diffusion Pipeline
### Load and configure the Stable Diffusion model:
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to(device)

pipe.enable_attention_slicing()


## Functions
get_sketch(category, size=(512, 512))
### Fetches a sketch of a specified category and resizes it to fit the given dimensions.
get_image_from_sketch(sketch_image, prompt)
### Generates a photorealistic image from the given sketch using the Stable Diffusion model.
super_resolution(image, scale_factor=2)
### Enhances the resolution of the generated image using bilinear interpolation.
enhance_with_realesrgan(image_path, output_path="enhanced_image.png")
### Enhances the image using Real-ESRGAN.

