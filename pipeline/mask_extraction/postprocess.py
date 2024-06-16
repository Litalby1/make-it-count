import cv2
import numpy as np
import torch
from pipeline.mask_extraction.utils_masks import to_channels, from_channels


def calculate_ratio(convex_hull, image):
    # Create a mask from the convex hull
    hull_image = np.zeros_like(image)
    cv2.fillConvexPoly(hull_image, convex_hull, 255)
    overlap = cv2.bitwise_and(image, hull_image)
    overlap_area = np.sum(overlap > 0)
    hull_area = np.sum(hull_image > 0)
    return 1 - (overlap_area / hull_area)


def blob_merger(binary_mask):
    # Convert to binary
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # Find contours
    contours_data = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_data) == 3:
        _, contours, _ = contours_data
    else:
        contours, _ = contours_data

    # Filter contours that are large enough to have a meaningful area
    filtered_contours = [cnt for cnt in contours if cnt.shape[0] >= 1]

    # Sort contours by area
    filtered_contours.sort(key=cv2.contourArea, reverse=True)

    # Process each contour and merge
    union_image = np.zeros_like(binary_mask)
    main_blob = filtered_contours[0]  # Start with the largest blob
    merged = False  # Flag to track if a merge has occurred

    # Draw the largest blob first
    cv2.drawContours(union_image, [main_blob], -1, (255), thickness=cv2.FILLED)

    # Try to merge the main blob with the next blobs
    for contour in filtered_contours[1:]:
        temp_image = np.copy(union_image)
        cv2.drawContours(temp_image, [contour], -1, (255), thickness=cv2.FILLED)
        combined_contour = np.vstack((main_blob, contour))  # Temporarily combine for the new convex hull
        convex_hull = cv2.convexHull(combined_contour)
        ratio = calculate_ratio(convex_hull, temp_image) # Calculate the ratio of the overlap area to the convex hull area
        if ratio <= 0.03:  # Adjusted ratio threshold for demonstration
            main_blob = combined_contour
            merged = True  # Set the flag to True if a merge occurs
            cv2.drawContours(union_image, [contour], -1, (255), thickness=cv2.FILLED)

    # Draw the final convex hull if a merge occurred
    if merged:
        convex_hull = cv2.convexHull(main_blob)
        cv2.fillConvexPoly(union_image, convex_hull, (255))
    else:
        # If no merge happened, draw the original largest blob
        cv2.drawContours(union_image, [filtered_contours[0]], -1, (255), thickness=cv2.FILLED)

    return union_image


def apply_dilation(binary_mask, kernel_size=3):
    """Applies erosion to the binary mask to create a margin."""
    if isinstance(binary_mask, torch.Tensor):  # Ensure it's a NumPy array for OpenCV functions
        binary_mask = binary_mask.cpu().numpy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
    return dilated_mask


def remove_intersections_from_largest(mask_outputs, buffer_size=3):
    num_masks = mask_outputs.size(0)
    mask_array = mask_outputs.view(num_masks, 32, 32)  # Assume each mask is 32x32 for easier handling

    # Apply dilation for each mask to handle touching borders
    dilated_masks = torch.stack([torch.from_numpy(apply_dilation(mask.numpy(), kernel_size=buffer_size)).to(mask_outputs.device) for mask in mask_array])

    for i in range(num_masks):
        for j in range(i + 1, num_masks):
            # Calculate intersection using dilated masks
            intersection = dilated_masks[i] & dilated_masks[j]
            if intersection.sum() > 0:  # Check if there is an intersection or touching border
                area_i = mask_array[i].sum()
                area_j = mask_array[j].sum()

                # Determine which mask is larger and remove the intersection from it
                if area_i > area_j:
                    mask_array[i][intersection.bool()] = 0  # Remove the intersection from the larger mask
                else:
                    mask_array[j][intersection.bool()] = 0

    return mask_array.view(num_masks, 32, 32)  # Reshape back to the original dimensions


def run_postprocess(mask_outputs):
    mask_outputs = to_channels(mask_outputs)

    for m in range(mask_outputs.size(0)):
        try:
            merged_mask = blob_merger(mask_outputs[m, :].reshape(32, 32).numpy())
            merged_mask = merged_mask / 255
            mask_outputs[m, :, :] = torch.tensor(merged_mask, dtype=torch.int64)
        except:
            continue

    seperated_masks = remove_intersections_from_largest(mask_outputs).view(-1, 1024)
    combined_mask = from_channels(seperated_masks)

    return combined_mask