import os
import torch
from torch.utils.data import Dataset
import numpy as np


class RelayoutDataset(Dataset):
    def __init__(self, data_dirs, config):
        self.total_input_dirs = []
        self.total_output_dirs = []
        self.config = config

        for dir in data_dirs:
            # Get lists of files from directories
            filtered_input_files = []
            filtered_output_files = []
            data_samples = [os.path.join(dir, f) for f in sorted(os.listdir(dir))]
            for i, sample_name in enumerate(data_samples):
                data_dict = torch.load(os.path.join(dir, sample_name))
                masks_input = data_dict["input"]
                masks_target = data_dict["target"]
                # note we only support masks input with exist number of objects such at least one object
                # and no more than 9.
                if (len(masks_input) <= 9) and (len(masks_input) > 0):
                    filtered_input_files.append(masks_input)
                    filtered_output_files.append(masks_target)

            self.total_input_dirs.extend(filtered_input_files)
            self.total_output_dirs.extend(filtered_output_files)


    def __len__(self):
        return len(self.total_input_dirs)

    def __getitem__(self, idx):
        masks_input = self.total_input_dirs[idx]
        masks_target = self.total_output_dirs[idx]

        # Random horizontal flip
        if torch.rand(1) < 0.5 and self.config["augmentation"]["horizontal_flip"]:
            masks_input = torch.flip(masks_input, dims=[2])  # Flip along the width axis
            masks_target = torch.flip(masks_target, dims=[2])

        # Padding masks input to 9 channels
        input_num = len(masks_input)
        if input_num < 9:
            padding_input = np.zeros((9 - input_num, 32, 32))
            masks_input = np.concatenate((masks_input, padding_input), axis=0)

        # Padding masks target to 10 channels
        target_num = len(masks_target)
        if target_num <= 10:
            padding_target = np.zeros((9 - input_num, 32, 32))
            unmatched_mask = np.expand_dims(masks_target[-1, :, :], axis=0)
            masks_target = np.concatenate((masks_target[:-1, :, :], padding_target, unmatched_mask), axis=0)

        input_tensor = torch.tensor(masks_input, dtype=torch.float32)
        output_tensor = torch.tensor(masks_target, dtype=torch.float32)

        # Optional: weight differently exist objects vs extra predicted object
        weight_vector = torch.tensor(([self.config["loss"]["mask_channel_weights"]] * input_num +
                         [0] * (9 - input_num) + [self.config["loss"]["last_channel_weights"]]), dtype=torch.float32)

        if self.config["augmentation"]["shuffle_channels"]:
            # Determine the number of active channels (those not padded)
            masks_input_shuffled = torch.zeros_like(input_tensor)
            masks_target_shuffled = torch.zeros_like(output_tensor)
            weight_vector_shuffled = torch.zeros_like(weight_vector)

            # Shuffle indices for input, target and weighted vector
            shuffled_indices = torch.randperm(9)[:input_num]
            masks_input_shuffled[shuffled_indices] = input_tensor[:input_num, :, :]
            masks_target_shuffled[shuffled_indices] = output_tensor[:input_num, :, :]
            masks_target_shuffled[-1, :, :] = output_tensor[-1, :, :]
            weight_vector_shuffled[shuffled_indices] = weight_vector[:input_num]
            weight_vector_shuffled[-1] = weight_vector[-1]

            # Replace original batches with shuffled
            input_tensor, output_tensor, weight_vector = masks_input_shuffled, masks_target_shuffled, weight_vector_shuffled

        return input_tensor, output_tensor, weight_vector