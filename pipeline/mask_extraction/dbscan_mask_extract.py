import torch
import yaml
import io
import numpy as np
import random
import matplotlib.pyplot as plt
from diffusers.utils.torch_utils import randn_tensor
from utils.attention_utils import attn_map_to_binary, show_mask

from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

def compute_pca(features, n_components=3, normalize=False):
    H, W, C = features.shape
    
    # min max normalization
    if normalize:
        features = (features - features.min()) / (features.max() - features.min())

    features = features.reshape(W*H, -1).cpu().numpy()
    pca = PCA(n_components=n_components)
    pca_featurs = pca.fit_transform(features)
    pca_featurs = pca_featurs.reshape(H, W, n_components)

    return pca_featurs

def show_pca(pca_features_list, titles=None, images_per_row=5):
    images_per_row = min(images_per_row, len(pca_features_list))

    titles = titles or ['Image'] * len(pca_features_list)
    num_features = len(pca_features_list)
    cmap = plt.get_cmap('inferno')
    
    images = []

    num_rows = (num_features + images_per_row - 1) // images_per_row  # Ceiling division to ensure all images fit
    
    # Create a figure with subplots in a row
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(6*images_per_row, 6*num_rows), squeeze=False)  # Ensures axes is always a 2D array

    for i, pca_features in enumerate(pca_features_list):
        normalized_pca_result = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        images.append(Image.fromarray((normalized_pca_result * 255).astype(np.uint8)))

        # Calculate row and column index for the current image
        row_idx, col_idx = divmod(i, images_per_row)
        
        ax = axes[row_idx, col_idx]
        ax.imshow(normalized_pca_result, cmap=cmap)
        # ax.set_title(titles[i])
        ax.axis('off')  # Optionally turn off the axis

    # Hide any unused subplots
    for j in range(i+1, num_rows * images_per_row):
        row_idx, col_idx = divmod(j, images_per_row)
        axes[row_idx, col_idx].axis('off')
        
    plt.tight_layout()  # Adjust layout so titles and images don't overlap
    plt.show()
    
    return images


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


def db_scan(self_attn_feats_mean, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', min_cluster_size=10,
            display_image=True):
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
    plt.imshow(cluster_labels.reshape(32, 32))
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


def save_pca_features_to_image(pca_features, filename):
    # Normalize the PCA features to the range [0, 255]
    pca_features -= pca_features.min()  # Make the minimum zero
    pca_features /= pca_features.max()  # Scale to [0, 1]
    pca_features = (pca_features * 255).astype(np.uint8)  # Scale to [0, 255]

    # Convert to PIL Image and save
    img = Image.fromarray(pca_features)
    img.save(filename)

def generate_imgs_masks(pipe, prompt, seed, num_inference_steps, db_eps=0.1):
    # set seed
    latents, g = generate_latents(pipe, seed, pipe.device)

    # Run pipeline
    out = pipe(prompt=[prompt],
               num_inference_steps=num_inference_steps,
               perform_counting=False,
               generator=g,
               latents=latents).images
    img = out[0]

    # Get Cross Attention
    cross_map = torch.stack([torch.cat(list(x.values())).mean(dim=0) for x in pipe.attention_store.cross_step_store.values()]).mean(dim=0)
    cross_map = cross_map[:, pipe.attention_store.object_token_idx] # Get the object token
    cross_mask = torch.from_numpy(attn_map_to_binary(cross_map, 1.)).to(cross_map.device).bool().view(-1)
    cross_mask_img = show_mask([{32: cross_mask}], out, 32, display_image=False)
    # Get Self Attention
    self_attn_feats_mean = get_agg_self_attn(pipe.attention_store.self_step_store, 25, layers=['up_52'],
                                             cross_mask=cross_mask)

    self_attn_feats_mean = (self_attn_feats_mean + self_attn_feats_mean.T) / 2

    scores = []
    # Clustering
    for eps in db_eps:
        np_clusters, n_dbscan_clusters, cluster_img = db_scan(self_attn_feats_mean, eps=eps, min_samples=10,
                                                              min_cluster_size=15,
                                                              metric='cosine', algorithm='auto', display_image=False)

        # Calculate Shilluette Score
        if n_dbscan_clusters > 1:
            shilluette_score = silhouette_score(self_attn_feats_mean.cpu().numpy(), np_clusters, metric='cosine')
        else:
            shilluette_score = 0

        scores.append(shilluette_score)

    best_eps = db_eps[np.argmax(scores)]
    np_clusters, n_dbscan_clusters, cluster_img = db_scan(self_attn_feats_mean, eps=best_eps, min_samples=10,
                                                          min_cluster_size=15,
                                                          metric='cosine', algorithm='auto', display_image=False)
    print(f'Best Epsilon: {best_eps}')

    size = (256, 256)
    cross_mask_img = cross_mask_img.resize(size)
    cluster_img = cluster_img.resize(size)
    return n_dbscan_clusters, img, cross_mask_img, cluster_img, np_clusters


# TODO: insert config into pipe
def dbscan_extract_mask(prompt, pipe, config, seed):
    n_dbscan_clusters, img, cross_mask_img, cluster_img, np_clusters = generate_imgs_masks(pipe, prompt, seed,
                                                                                           config["counting_model"][
                                                                                               "num_inference_steps"],
                                                                                           db_eps=np.arange(0.1, 0.2,
                                                                                                            0.01))

    return np_clusters, n_dbscan_clusters, img