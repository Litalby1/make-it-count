import numpy as np
import torch
from scipy.ndimage import label, find_objects
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch.nn.functional as F


def from_channels(channels_mask):
    desired_number_of_clusters = channels_mask.size(0)
    channels_mask = channels_mask.flatten(start_dim=1)

    combined_mask = torch.zeros(1024, dtype=torch.float16)

    for i in range(desired_number_of_clusters):
        combined_mask[channels_mask[i] == 1] = i + 1

    combined_mask = combined_mask.view(32, 32)
    return combined_mask


def to_channels(masks):
    num_of_masks = int(masks.max().item())
    torch_object_masks = torch.zeros((num_of_masks, 32, 32), dtype=torch.float16)

    for i in range(num_of_masks):
        torch_object_masks[i][masks.detach() == i + 1] = 1

    return torch_object_masks


def remove_sparse_blobs(grid, min_blob_size=10):
    grid = grid.reshape(32, 32)
    unique_masks = np.unique(grid)
    unique_masks = unique_masks[unique_masks >= 0]
    # Create a copy of the grid to modify
    updated_grid = grid.copy()
    # Process each unique mask
    for mask_id in unique_masks:
        # Create a binary mask for the current id
        binary_mask = (grid == mask_id)
        # Label connected components in the binary mask
        labeled_array, num_features = label(binary_mask)
        # Boolean to track if all blobs are less than 10 pixels
        all_blobs_small = True
        # Check each labeled feature
        for i in range(1, num_features + 1):
            blob_size = np.sum(labeled_array == i)
            if blob_size >= min_blob_size:
                all_blobs_small = False
                break
        # If all blobs in this mask are smaller than 10 pixels, mark those regions as -1
        if all_blobs_small:
            updated_grid[binary_mask] = -1

    # Re number clusters
    unique_masks = np.unique(updated_grid)
    unique_masks = unique_masks[unique_masks >= 0]
    n_cluster_new = len(unique_masks)

    
    for i, cluster in enumerate(unique_masks):
        updated_grid[updated_grid == cluster] = i

    return torch.tensor(updated_grid).flatten(), n_cluster_new


def cut_bbox(boxes, image_size):
    fig, ax = plt.subplots()
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)

    # Plot each bounding box in a random color
    base_color = (0, 0, 0)
    for i, box in enumerate(boxes):
        color = base_color if i == 0 else np.random.rand(3, )
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)

    # Calculate and plot the bounding box that encloses all boxes
    relevant_boxes = boxes[1:]
    x1_all = min(box[0] for box in relevant_boxes)
    y1_all = min(box[1] for box in relevant_boxes)
    x2_all = max(box[2] for box in relevant_boxes)
    y2_all = max(box[3] for box in relevant_boxes)

    old_center = (image_size // 2, image_size // 2)

    dist_left = old_center[1] - x1_all
    dist_right = x2_all - old_center[1]
    dist_up = old_center[0] - y1_all
    dist_down = y2_all - old_center[0]
    print(f'dist_left: {dist_left}, dist_right: {dist_right}, dist_up: {dist_up}, dist_down: {dist_down}')

    final_rect_half_w = max(dist_left, dist_right, image_size // 4)
    final_rect_half_h = max(dist_up, dist_down, image_size // 4)
    final_half = max(final_rect_half_h, final_rect_half_w)
    final_rect = [image_size // 2 - final_half, image_size // 2 - final_half,
                  image_size // 2 + final_half, image_size // 2 + final_half]

    rect_all = patches.Rectangle((x1_all, y1_all), x2_all - x1_all, y2_all - y1_all, linewidth=2, edgecolor='blue',
                                 facecolor='none')
    ax.add_patch(rect_all)

    final_rect_patch = patches.Rectangle((final_rect[0], final_rect[1]), 2 * final_half, 2 * final_half,
                                         linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(final_rect_patch)

    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.axis('scaled')
    plt.title('Bounding Boxes')
    # plt.show()

    return relevant_boxes, final_rect


def get_bounding_boxes(masks):
    """
    Extract bounding boxes from a set of binary masks.

    Args:
    masks (torch.Tensor): A tensor of shape (num_masks, height, width)

    Returns:
    List[tuple]: A list of bounding boxes in the format (x1, y1, x2, y2)
    """
    boxes = []
    for i in range(masks.shape[0]):
        mask = masks[i]
        # Find indices where there's a 1
        nonzero = torch.nonzero(mask, as_tuple=False)
        if nonzero.size(0) == 0:
            continue  # No object in this mask
        ymin, xmin = torch.min(nonzero, dim=0).values
        ymax, xmax = torch.max(nonzero, dim=0).values
        # Append bounding box coordinates in format [x1, y1, x2, y2]
        boxes.append((xmin.item(), ymin.item(), xmax.item(), ymax.item()))
    return boxes


def find_tight_bbox(tensor):
    # Project along channels to find any mask presence
    projection = torch.any(tensor > 0, dim=1)[0]  # Assuming batch size of 1

    # Find indices where there's any mask
    non_zero_indices = projection.nonzero(as_tuple=True)
    y_min, y_max = non_zero_indices[0].min().item(), non_zero_indices[0].max().item()
    x_min, x_max = non_zero_indices[1].min().item(), non_zero_indices[1].max().item()

    return (y_min, x_min, y_max, x_max)


def make_square_crop(tensor, bbox):
    y_min, x_min, y_max, x_max = bbox
    height = y_max - y_min
    width = x_max - x_min

    # Determine the size of the square (the max of height and width)
    square_size = max(height, width)

    # Calculate padding to make the image square
    padding_top = (square_size - height) // 2
    padding_bottom = square_size - height - padding_top
    padding_left = (square_size - width) // 2
    padding_right = square_size - width - padding_left

    # Crop the tensor to the bounding box
    cropped_tensor = tensor[..., y_min:y_max, x_min:x_max]

    # Pad to make square
    square_tensor = F.pad(cropped_tensor, (padding_left, padding_right, padding_top, padding_bottom), "constant", 0)

    return square_tensor