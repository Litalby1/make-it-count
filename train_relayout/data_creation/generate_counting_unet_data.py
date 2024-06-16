import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import os
import diffusers
import torch
import yaml
import io
import json

from pipeline.self_counting_sdxl_pipeline import SelfCountingSDXLPipeline
import numpy as np
import random
import matplotlib.pyplot as plt
from diffusers.utils.torch_utils import randn_tensor
from utils.attention_utils import attn_map_to_binary, show_mask

from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import inflect

def set_seed(seed: int):
    """
    Set the seed for reproducibility in PyTorch, NumPy, and Python's random module.

    Parameters:
    - seed (int): The seed value to use for all random number generators.
    """
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Make CuDNN backend deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

def read_yaml(file_path):
    with open(file_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data

def generate_latents(pipe, seed, device):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    shape = (1, pipe.unet.config.in_channels, 128, 128)
    latents = randn_tensor(shape, generator=g, device=device, dtype=torch.float16)

    return latents, g

def get_agg_self_attn(self_step_store, t, layers=None, cross_mask=None):
    if layers is None:
        layers = list(self_step_store[t].keys())
        
    self_attn_feats_mean = torch.cat([self_step_store[t][x] for x in layers]).mean(dim=0)

    if cross_mask is not None:
        self_attn_feats_mean[~cross_mask, :] = 0
        self_attn_feats_mean[:, ~cross_mask] = 0
    
    return self_attn_feats_mean

def db_scan(self_attn_feats_mean, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', min_cluster_size=10, display_image=True):
    X = self_attn_feats_mean.cpu().numpy()
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(X)
    cluster_labels = clustering.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in clustering.labels_ else 0)

    small_clusters = [i for i in range(n_clusters) if (clustering.labels_ == i).sum() < min_cluster_size]
    
    for i in small_clusters:
        cluster_labels[clustering.labels_ == i] = -1
        n_clusters -= 1
    
    # Re number clusters
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    for i, cluster in enumerate(set(cluster_labels)):
        if cluster == -1:
            continue
        cluster_labels[cluster_labels == cluster] = i

    # Display the clustering in the original shape
    plt.imshow(cluster_labels.reshape(32,32))
    plt.title(f'Clustering with DBSCAN')

    # Save the image to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    cluster_img = Image.open(buf).convert('RGB')

    if display_image:
        plt.show()
    else:
        plt.close()

    return cluster_labels, n_clusters, cluster_img

def generate_imgs_masks(pipe, prompt, seed, num_inference_steps, db_eps=0.1):
    # set seed
    latents, g = generate_latents(pipe, seed, device)

    # Run pipeline
    out = pipe(prompt=[prompt], 
                    num_inference_steps=num_inference_steps,
                    perform_counting=False,
                    generator=g, 
                    latents=latents).images
    img = out[0]

    # Get Cross Attention
    cross_map = torch.stack([torch.cat(list(x.values())).mean(dim=0) for x in pipe.attention_store.cross_step_store.values()]).mean(dim=0)
    cross_map = cross_map[:, 5] # Get the object token
    cross_mask = torch.from_numpy(attn_map_to_binary(cross_map, 1.)).to(cross_map.device).bool().view(-1)
    cross_mask_img = show_mask([{32: cross_mask}], out, 32, display_image=False)


    # Get Self Attention
    self_attn_feats_mean = get_agg_self_attn(pipe.attention_store.self_step_store, 25, layers=['up_52'], cross_mask=cross_mask)
    self_attn_feats_mean = (self_attn_feats_mean + self_attn_feats_mean.T) / 2

    scores = []
    # Clustering
    for eps in db_eps:
        np_clusters, n_dbscan_clusters, cluster_img = db_scan(self_attn_feats_mean, eps=eps, min_samples=10, min_cluster_size=15,
                                                            metric='cosine', algorithm='auto', display_image=False)

        # Calculate Shilluette Score
        if n_dbscan_clusters > 1:
            shilluette_score = silhouette_score(self_attn_feats_mean.cpu().numpy(), np_clusters, metric='cosine')
        else:
            shilluette_score = 0

        scores.append(shilluette_score)

    best_eps = db_eps[np.argmax(scores)]
    np_clusters, n_dbscan_clusters, cluster_img = db_scan(self_attn_feats_mean, eps=best_eps, min_samples=10, min_cluster_size=15,
                                                        metric='cosine', algorithm='auto', display_image=False)
    print(f'Best Epsilon: {best_eps}')


    size = (256, 256)
    img = img.resize(size)
    cross_mask_img = cross_mask_img.resize(size)
    cluster_img = cluster_img.resize(size)

    return n_dbscan_clusters, img, cross_mask_img, cluster_img, np_clusters


def parse_arguments():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="train_relayout/output_unet_data")

    return parser.parse_args()

if __name__ == "__main__": 
    args = parse_arguments()

    config = read_yaml("pipeline/pipeline_config.yaml")
    pipe = SelfCountingSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16",
        use_onnx=False
    )
    pipe.counting_config = config['counting_model']

    # device
    config['pipeline']['device'] = f'cuda:{args.gpu_id}'
    device = torch.device(f'cuda:{args.gpu_id}')
    pipe.to(device)

    if config['counting_model']['use_ddpm']:
        print('Using DDPM as scheduler.')
        pipe.scheduler = diffusers.DDPMScheduler.from_config(pipe.scheduler.config)

    p = inflect.engine()

    out_dir = args.output_dir
    # make sure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    NUM_OBJECTS_TO_RUN = 10000

    start_seed = args.start_seed
    random.seed(start_seed)
    seeds = [random.randint(1, 10000000) for _ in range(NUM_OBJECTS_TO_RUN)]

    object_list = ['car', 'airplane', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'backpack', 'tie', 'ball', 'glove', 'cup', 'bowl', 'apple', 'donut', 'phone', 'clock']
    numbers = ['two', 'three', 'four', 'five', 'six']
    suffixes = ['on the grass', 'on the road', 'on the ground']

    one_up = {'two': 'three', 'three': 'four', 'four': 'five', 'five': 'six', 'six': 'seven', 'seven': 'eight', 'eight': 'nine'}

    p_use_suffix = 0.5

    for i, seed in enumerate(seeds):
        random.seed(seed)
        
        object = random.choice(object_list)
        number = random.choice(numbers)

        if random.random() < p_use_suffix:
            suffix = ' ' + random.choice(suffixes)
        else:
            suffix = ''

        object_plural = p.plural(object)

        prompt = f'A photo of {number} {object_plural}{suffix}'
        prompt_2 = f'A photo of {one_up[number]} {object_plural}{suffix}'

        print(prompt)

        n_dbscan_clusters, img, cross_mask_img, cluster_img, np_clusters = generate_imgs_masks(pipe, prompt, seed, config["counting_model"]["num_inference_steps"], db_eps=np.arange(0.1, 0.2, 0.01))
        n_dbscan_clusters_2, img_2, cross_mask_img_2, cluster_img_2, np_clusters2 = generate_imgs_masks(pipe, prompt_2, seed, config["counting_model"]["num_inference_steps"], db_eps=np.arange(0.1, 0.2, 0.01))

        print(f'seed: {seed}. {prompt}: {n_dbscan_clusters} objects, and {n_dbscan_clusters_2} objects')

        if n_dbscan_clusters + 1 == n_dbscan_clusters_2:
            print('**************************** Found Good example! ****************************')

            dir_name = f'{object}_{number}_{seed}_clusters={n_dbscan_clusters}-{n_dbscan_clusters_2}'

            dir_path = os.path.join(out_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)

            img.save(os.path.join(dir_path, 'img1.png'))
            cluster_img.save(os.path.join(dir_path, 'cluster_img1.png'))
            cross_mask_img.save(os.path.join(dir_path, 'cross_mask_img1.png'))
            np.save(os.path.join(dir_path, 'cluster1.npy'), np_clusters)

            img_2.save(os.path.join(dir_path, 'img2.png'))
            cluster_img_2.save(os.path.join(dir_path, 'cluster_img2.png'))
            cross_mask_img_2.save(os.path.join(dir_path, 'cross_mask_img2.png'))
            np.save(os.path.join(dir_path, 'cluster2.npy'), np_clusters2)

            json_data = {
                'seed': seed,
                'prompt': prompt,
                'prompt_2': prompt_2,
                'n_clusters': n_dbscan_clusters,
                'n_clusters_2': n_dbscan_clusters_2,
                'object': object,
                'object_plural': object_plural,
                'number': number
            }

            with open(os.path.join(dir_path, 'metadata.json'), 'w') as f:
                json.dump(json_data, f)