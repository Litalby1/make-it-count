from typing import Union, List
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from PIL import ImageOps
from skimage import filters


# Display images
def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True,
                downscale_rate=None) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)

    if downscale_rate:
        pil_img = pil_img.resize((int(pil_img.size[0] // downscale_rate), int(pil_img.size[1] // downscale_rate)))

    if display_image:
        display(pil_img)
    return pil_img

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def show_mask(masks_list, out_images, tgt_size, display_image=True):
    plt.figure(figsize=(5,5))
    images = []

    for index_in_batch, masks in enumerate(masks_list):
        attn_image = masks[tgt_size].view(tgt_size,tgt_size).float()
        attn_image = show_image_relevance(attn_image, out_images[index_in_batch])
        attn_image = attn_image.astype(np.uint8)
        attn_image = np.array(Image.fromarray(attn_image).resize((tgt_size, tgt_size)))
        images.append(attn_image)

    images = view_images(np.stack(images, axis=0), display_image=False)

    if display_image:
        plt.imshow(images)
        plt.show()

    return images

def get_dynamic_threshold(tensor):
    return filters.threshold_otsu(tensor.cpu().numpy())

def attn_map_to_binary(attention_map, scaler=1.):
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask

def plot_object_attention_map(object_attention_map, output_dir, cross_attention_dim, i):
    object_attention_map_reshaped = object_attention_map.reshape(-1, 1).detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(object_attention_map_reshaped.reshape(cross_attention_dim, cross_attention_dim), cmap='gray')
    plt.axis('off')
    plt.savefig(f"{output_dir}/object_attention_map_{i}.png", bbox_inches='tight')

def concat_images(images, size=512):
    # Open images and resize them
    width = height = size
    images = [ImageOps.fit(image, (size, size), Image.LANCZOS)
              for image in images]

    # Create canvas for the final image with total size
    shape = (math.isqrt(len(images)), math.ceil(len(images)/math.isqrt(len(images))))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)

    return image