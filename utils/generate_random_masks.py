import torch
import random
import matplotlib.pyplot as plt
import numpy as np

# Squares
def place_square(mask, k, value, padding, bounding_boxes):
    """
    Try to place a KxK square in the mask with the given value.
    The function returns True if the square was successfully placed, otherwise False.
    """
    H, W = mask.shape
    
    for _ in range(10000):  # Attempt up to 1000 times to find a valid placement
        x = random.randint(0, W - k - 2*padding)
        y = random.randint(0, H - k - 2*padding)
        
        bounding_box = (x / W, y / H, (x+k+padding) / W, (y+k+padding) / H)
        
        # Check if the area is free
        if torch.all(mask[y:y+k+2*padding, x:x+k+2*padding] == 0):
            mask[y+padding:y+k+padding, x+padding:x+k+padding] = value
            bounding_boxes.append(bounding_box)
            return True
    return False

def generate_squares_mask(n, k, mask_size=(32,32), padding=1, max_attempts=100):
    """
    Generate a 128x128 mask with N non-overlapping KxK squares.
    """
    for attempt in range(max_attempts):
        mask = torch.zeros(mask_size, dtype=torch.int)
        bounding_boxes = []
        for value in range(1, n + 1):
            if not place_square(mask, k, value, padding, bounding_boxes):
                break  # Breaks from the for-loop if placement failed
        else:
            # If the loop completes without breaking, all squares were placed successfully
            return mask, bounding_boxes
    raise ValueError(f"Unable to place all squares after {max_attempts} attempts.")

# Circles
def circles_do_not_overlap(center1, center2, min_distance):
    """Check if two circles do not overlap given their centers and minimum distance."""
    x1, y1 = center1
    x2, y2 = center2
    distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return distance_squared >= min_distance ** 2

def place_centers(n, radius, mask_size, padding):
    """Place centers ensuring they don't overlap."""
    centers = []
    max_attempts = 10000
    min_distance = 2 * radius + 2 * padding  # Minimum distance between centers

    for _ in range(n):
        for attempt in range(max_attempts):
            x_center = random.randint(radius + padding, mask_size[1] - radius - padding - 1)
            y_center = random.randint(radius + padding, mask_size[0] - radius - padding - 1)
            new_center = (x_center, y_center)

            # Check the new center against all existing centers
            if all(circles_do_not_overlap(center, new_center, min_distance) for center in centers):
                centers.append(new_center)
                break
        else:
            return None # Could not place all centers
    
    return centers

def draw_circles(mask, centers, radius, value_start=1):
    """Draw circles on the mask based on the centers."""
    for i, (x_center, y_center) in enumerate(centers, start=value_start):
        for y in range(mask.size(0)):
            for x in range(mask.size(1)):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    mask[y, x] = i

def generate_circles_mask(n, radius, mask_size=(128, 128), padding=1, max_attempts=100):
    mask = torch.zeros(mask_size, dtype=torch.int)

    for attempt in range(max_attempts):
        centers = place_centers(n, radius, mask_size, padding)
        if centers is not None:
            draw_circles(mask, centers, radius)
            return mask
        
    return None # Could not place all circles

# Display the mask
def show_mask(mask, save_path=None):
    """
    Display the mask with different colors for each square.
    """
    plt.figure(figsize=(8, 8))  # Set figure size
    # Use 'viridis' or any other colormap. Ensure there are enough colors for all squares.
    plt.imshow(mask, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Show color scale
    plt.axis('off')  # Hide axis

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def show_mask_list(masks, titles=None, save_path=None):
    """
    Display a list of masks with different colors for each square.
    """
    if titles is None:
        titles = [f"Mask {i+1}" for i in range(len(masks))]

    fig, axs = plt.subplots(1, len(masks), figsize=(4 * len(masks), 4))  # Set figure size
    for ax, mask, title in zip(axs, masks, titles):
        ax.imshow(mask, cmap='viridis', interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')  # Hide axis

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def generate_random_masks_factory(shape='circle', number_clusters=2):
    if shape == 'circle':
        # calculate so the area of all circles will be ~400
        radius = int(np.sqrt(400/(np.pi*number_clusters)))
        mask = generate_circles_mask(number_clusters, radius, mask_size=(32, 32), padding=1, max_attempts=200)

        if mask is None:
            radius -= 1
            mask = generate_circles_mask(number_clusters, radius, mask_size=(32, 32), padding=1, max_attempts=200)
        
        return mask
    elif shape == 'square':
        return generate_squares_mask(number_clusters, 8, mask_size=(32, 32), padding=1, max_attempts=200)