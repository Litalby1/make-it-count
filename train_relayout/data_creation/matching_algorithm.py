import argparse
import os
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np


def iou(mask1, mask2):
    """Calculate the Intersection over Union (IoU) of two binary masks."""
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    return intersection / union if union != 0 else 0


def save_as_tensors(matched_pairs, unmatched_mask, output_path):
    """Save matched pairs and the unmatched mask as separate .pt files."""
    matched_batch1 = torch.tensor([m[0] for m in matched_pairs])
    matched_batch2 = torch.tensor([m[1] for m in matched_pairs])

    print(f"matched_batch2 shape: {matched_batch2.shape}")

    if unmatched_mask is not None:
        unmatched_mask_tensor = torch.tensor(unmatched_mask)

        print(f"unmatched_mask_tensor shape: {unmatched_mask_tensor.shape}")

        # Ensure consistency in reshaping
        if unmatched_mask_tensor.ndim < matched_batch2.ndim:
            # Add necessary dimensions
            while unmatched_mask_tensor.ndim < matched_batch2.ndim:
                unmatched_mask_tensor = unmatched_mask_tensor.unsqueeze(0)

        print(f"Adjusted unmatched_mask_tensor shape: {unmatched_mask_tensor.shape}")

        matched_batch2 = torch.cat((matched_batch2, unmatched_mask_tensor))
    data_dict = {
        "input": matched_batch1,
        "target": matched_batch2
    }
    torch.save(data_dict, output_path)


def compute_bounding_box(mask):
    """Convert a binary mask into a bounding box with (x, y, w, h) format."""
    if mask.ndim == 4:
        mask = mask.squeeze()  # Handle cases with an additional dimension

    coords = np.column_stack(np.where(mask))  # Extract non-zero coords
    if len(coords) == 0:
        return None

    min_coords, max_coords = np.min(coords, axis=0), np.max(coords, axis=0)
    w, h = max_coords - min_coords + 1
    return (*min_coords + [w // 2, h // 2], w, h)


def bbox_distance(bbox1, bbox2):
    """Calculate the distance between two bounding boxes."""
    if bbox1 is None or bbox2 is None:
        return float('inf')  # Treat absence of a bbox as an infinite distance

    # Distance between centers
    center_dist = np.sqrt((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2)

    # Difference in size (Euclidean distance)
    size_dist = np.sqrt((bbox1[2] - bbox2[2]) ** 2 + (bbox1[3] - bbox2[3]) ** 2)

    return center_dist + size_dist


def match_bboxes_hungarian(batch1, batch2):
    """Match bounding boxes between two batches using the Hungarian algorithm."""
    bboxes1 = [compute_bounding_box(mask) for mask in batch1]
    bboxes2 = [compute_bounding_box(mask) for mask in batch2]

    # Create a cost matrix based on bbox distances
    cost_matrix = np.array([
        [bbox_distance(b1, b2) if b1 and b2 else float('inf') for b1 in bboxes1]
        for b2 in bboxes2
    ])

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_indices, col_indices].sum()

    # Create matched pairs and resize them to 32x32
    matched_pairs = [
        (batch1[r], batch2[c]) for r, c in zip(col_indices, row_indices)
    ]

    # Select an unmatched mask and resize it
    unmatched_mask = [batch2[c] for c in range(len(batch2)) if c not in row_indices]

    if unmatched_mask:
        unmatched_mask = unmatched_mask[0]  # Use one left out mask

    return matched_pairs, unmatched_mask, total_cost


def run_matching(batch1, batch2, output_path):
    # Find the best permutation and unmatched mask
    matched_pairs, unmatched_mask, total_cost = match_bboxes_hungarian(batch1, batch2)

    grade_per_pairs = total_cost/len(matched_pairs)

    if grade_per_pairs > 9:  # Threshold for filtered out pairs of images that don't have similar layout scene
        return

    save_as_tensors(matched_pairs, unmatched_mask, output_path)


def create_masks_from_file(file_path):
    if os.path.exists(file_path):
        array = np.load(file_path).reshape(32, 32)
        unique_values = np.unique(array[array > -1])
        masks = np.zeros((len(unique_values), 32, 32), dtype=int)
        for idx, value in enumerate(unique_values):
            masks[idx, :, :] = (array == value)
        return masks
    return None  # Return None if the file does not exist

parser = argparse.ArgumentParser(description='Process some file paths.')
parser.add_argument('--data_dir', help='Base path for data')
parser.add_argument('--output_dir', help='Output directory for matched pairs')

if __name__ == "__main__":
    args = parser.parse_args()
    base_path = args.data_dir
    out_dir = args.output_dir

    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for sample_idx, subdir in enumerate(subdirs):
        try:
            subdir_path = os.path.join(base_path, subdir)
            masks1 = create_masks_from_file(os.path.join(subdir_path, 'cluster1.npy'))
            masks2 = create_masks_from_file(os.path.join(subdir_path, 'cluster2.npy'))

            # Create same index for both input and target pairs masks in the batch
            output_path = os.path.join(out_dir, f'{subdir}_{sample_idx}.pt')

            # Run the matching algorithm
            run_matching(masks1, masks2, output_path)
        except:
            continue

