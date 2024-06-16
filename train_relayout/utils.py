import numpy as np
from matplotlib import pyplot as plt
import wandb
import torch


def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt)
    union = np.clip(pred + gt, 0, 1)

    # Avoid division by zero by checking if the union has no True values
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)


def plot_masks(inputs_np, targets_np, outputs_np, is_wandb, weight_vector,
               extra_mask_intersection, extra_mask_median, epoch, sample_index):
    """Generate a combined plot for inputs, targets, and predictions."""
    # Assuming 'inputs_np', 'targets_np', 'outputs_np' and 'weight_vector' are predefined
    num_predictions = max(inputs_np.shape[1], targets_np.shape[1], outputs_np.shape[1])

    # Define a base colormap
    base_cmap = plt.cm.get_cmap('tab20', num_predictions)  # 'tab20' provides up to 20 distinct colors
    colors = base_cmap(np.linspace(0, 1, num_predictions))

    fig, axes = plt.subplots(3, num_predictions + 1, figsize=(16, 9))

    for i in range(num_predictions):
        color = colors[i % len(colors)]

        for row, data in enumerate([inputs_np, targets_np, outputs_np]):
            if i < data.shape[1]:
                mask = np.where(data[0, i] > 0, 1, 0)
                axes[row, i].imshow(mask, cmap='gray', interpolation='nearest')
                axes[row, i].set_title(f"{'Input' if row == 0 else 'Target' if row == 1 else 'Prediction'} {i}")
                axes[row, i].axis("off")

    # Initialize the aggregated mask with a dark background
    for row, data in enumerate([inputs_np, targets_np, outputs_np]):
        aggregate_rgba = np.zeros((data.shape[2], data.shape[3], 4))

        for i in range(data.shape[1]):
            if weight_vector[0][i].item() > 0:
                mask = np.where(data[0, i] > 0, 1, 0)
                color = colors[i % len(colors)]
                rgba_layer = np.zeros((data.shape[2], data.shape[3], 4))
                rgba_layer[:, :, :3] = color[:3] * mask[:, :, np.newaxis]  # Apply color to the mask
                rgba_layer[:, :, 3] = mask * 0.7  # Set alpha for visible areas
                aggregate_rgba += rgba_layer

        # Normalize and clip RGB channels to prevent them from summing to white
        max_color_intensity = 0.75  # Reduce this if colors are too bright or increase for more intensity
        aggregate_rgba[:, :, :3] = np.clip(aggregate_rgba[:, :, :3], 0, max_color_intensity)
        aggregate_rgba[:, :, 3] = np.clip(aggregate_rgba[:, :, 3], 0, 1)  # Ensure alpha is within [0, 1]

        axes[row, -1].imshow(aggregate_rgba)
        axes[row, -1].set_title(
            f"Aggregate\nInt.Score: {extra_mask_intersection:.2f}\nMed.Score: {extra_mask_median:.2f}")
        axes[row, -1].axis("off")

    axes[0, -2].axis('off')
    plt.tight_layout()
    if is_wandb:
        wandb.log({f"Segmentation_Masks": wandb.Image(fig)}, step=sample_index + 1000000)


def avg_extra_mask_intersection(predictions, targets, weight_vector):
    """Computes the average intersection-over-union (IoU) score for an 'extra' object mask in each batch of predictions
     against an aggregate target mask composed of weighted masks."""
    total_score = 0
    for batch_idx in range(predictions.shape[0]):
        extra_object = predictions[batch_idx, -1]
        # Aggregate all other masks into one binary mask
        aggregate_mask = torch.zeros_like(targets[batch_idx, 0])

        for mask_idx in range(predictions.shape[1] - 1):
            if weight_vector[batch_idx][mask_idx] > 0:
                aggregate_mask += targets[batch_idx, mask_idx]
        # clip the aggregate mask to ensure it is binary
        aggregate_mask = torch.clamp(aggregate_mask, 0, 1)
        intersection = extra_object * aggregate_mask.cpu().numpy()  # Use multiplication for logical AND
        score = intersection.sum() / (extra_object.sum()) if extra_object.sum() > 0 else 0  # Prevent division by zero
        total_score += score  # Add the score for this batch

    avg_score = total_score / predictions.shape[0]
    return avg_score


def median_extra_object(predictions, targets, weight_vector):
    """Calculates the average ratio of the size of the 'extra' object mask to the median size of target masks in each
     batch, providing a scale comparison of the 'extra' object relative to typical objects in the dataset."""
    distance_from_median = 0
    for batch_idx in range(predictions.shape[0]):
        extra_object = predictions[batch_idx, -1]
        medians = []
        for mask_idx in range(predictions.shape[1] - 1):
            if weight_vector[batch_idx][mask_idx] > 0:
                medians.append(torch.sum(targets[batch_idx, mask_idx]))

        medians_tensor = torch.tensor(medians)
        median_of_mask = torch.median(medians_tensor).item()
        sum_extra_object = np.sum(extra_object)
        # Calculate the distance from the median
        min_val = min(median_of_mask, sum_extra_object)
        max_val = max(median_of_mask, sum_extra_object)
        distance_from_median += (min_val / max_val)

    avg_score = distance_from_median / predictions.shape[0]
    return avg_score


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        """
        Initializes the EarlyStopper instance.

        Parameters:
            patience (int): The number of epochs with no improvement after which training will be stopped.
            min_delta (float): The minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def check_stop(self, validation_loss):
        """
        Determines whether training should be stopped early.

        Parameters:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if validation_loss < self.min_validation_loss - self.min_delta:
            # Reset counter if there is a significant improvement
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            # Increment counter if no improvement
            print("early stopping counter: ", self.counter)
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
