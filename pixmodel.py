# -*- coding: utf-8 -*-
"""PixModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lxbim29jWdgaoyp3otT7YnMDpaltJFRo
"""

!pip install --upgrade diffusers transformers accelerate
!pip install torch torchvision diffusers transformers quickdraw scipy ftfy accelerate xformers

!pip install --upgrade torch torchvision

import torch
import numpy as np
from PIL import Image, ImageDraw
from quickdraw import QuickDrawData
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qd = QuickDrawData()

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to(device)

pipe.enable_attention_slicing()

def get_sketch(category, size=(512, 512)):
    try:
        sketch = qd.get_drawing(category)
        max_ratio = 5.0
        aspect_ratio = max(sketch.image.width, sketch.image.height) / min(sketch.image.width, sketch.image.height)
        if aspect_ratio > max_ratio:
            print(f"Warning: Sketch aspect ratio ({aspect_ratio:.2f}) is too extreme, skipping.")
            return None

        # white background image
        image = Image.new('RGB', size, color='white')

        # scaling factor to fit
        scale = min(size[0] / sketch.image.width, size[1] / sketch.image.height)


        scaled_width = int(sketch.image.width * scale)
        scaled_height = int(sketch.image.height * scale)

        # Calculate position to center the sketch
        left = (size[0] - scaled_width) // 2
        top = (size[1] - scaled_height) // 2

        draw = ImageDraw.Draw(image)
        for stroke in sketch.strokes:
            scaled_stroke = [(int(x * scale) + left, int(y * scale) + top) for x, y in stroke]
            draw.line(scaled_stroke, fill='black', width=2)

        return image

    except KeyError:
        print(f"Error: Category '{category}' not found.")
        return None

def get_image_from_sketch(sketch_image, prompt):
    # Preprocess the image
    sketch_image = sketch_image.convert("RGB")

    # Generate image
    image = pipe(prompt=prompt, image=sketch_image, strength=0.75, guidance_scale=7.5).images[0]

    return image

import torch
import torchvision.transforms as transforms
from PIL import Image

def super_resolution(image, scale_factor=2):
    # Convert PIL Image to tensor
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(image).unsqueeze(0)

    # Move tensor to the available device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)

    # Upscale the image using bilinear interpolation
    upscaled = torch.nn.functional.interpolate(
        img_tensor,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=True
    )

    to_pil = transforms.ToPILImage()
    upscaled_image = to_pil(upscaled.squeeze().cpu().clamp(0, 1))

    return upscaled_image

category = "house"
sketch_image = get_sketch(category)
if sketch_image is not None:
    sketch_image.save(f"sketch_{category}.png")
    print(f"Initial sketch saved as {category}_sketch.png")
    prompt = f"A photorealistic image of a detailed {category}, inspired by a sketch."
    try:
        generated_image = get_image_from_sketch(sketch_image, prompt)
        generated_image.save(f"{category}_pix2pix.png")
        if generated_image:
            enhanced_image = super_resolution(generated_image, scale_factor=2)
            enhanced_image.save(f"enhanced_{category}.png")
            print(f"Enhanced image saved as enhanced_{category}.png")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Error: Out of memory. Try reducing image size or complexity.")
        else:
            print(f"An error occurred: {e}")
else:
    print("Failed to generate sketch.")

#Using ERSGAN
!pip install realesrganimport torch
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from PIL import Image

model_path = '/opt/anaconda3/envs/diffuser/lib/python3.10/site-packages/realesrgan/weights/realesr-general-x4v3.pth'
netscale = 4

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True
)

if torch.cuda.is_available():
    upsampler = upsampler.cuda()

def enhance_with_realesrgan(image_path, output_path="enhanced_image.png"):
    try:
        img = Image.open(image_path).convert('RGB')

        with torch.no_grad():
            output, _ = upsampler.enhance(img)

        output.save(output_path)
        print(f"Enhanced image saved to {output_path}")

    except Exception as e:
        print(f"Error during Real-ESRGAN enhancement: {e}")

'''
To run:
enhance_with_realesrgan(image_path)
'''

