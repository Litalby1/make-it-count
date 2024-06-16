import torch
import torch.nn.functional as F
from pipeline.mask_extraction.postprocess import blob_merger
from pipeline.mask_extraction.utils_masks import find_tight_bbox, make_square_crop


def relayout_undergeneration(input_mask, number_of_clusters_founded, desired_number_of_clusters, config, in_channels=9, out_channels=10):
    gradual_padding = config["mask_creation"]["dbscan_mask"]["gradual_padding"]
    p = 0  # padding
    unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
           in_channels=in_channels, out_channels=out_channels, init_features=128, pretrained=False)

    checkpoint = torch.load(config["mask_creation"]["dbscan_mask"]["unet_checkpoint_path"], map_location=torch.device(config['pipeline']['device']))
    unet_model.load_state_dict(checkpoint)
    unet_model.eval()
    input_mask = input_mask.unsqueeze(0)
    zeros = torch.zeros(input_mask.shape[0], in_channels-number_of_clusters_founded, input_mask.shape[2], input_mask.shape[3])
    input_mask = torch.cat((input_mask, zeros), 1)

    # add padding to the input mask
    padding = (p, p, p, p)
    input_mask = F.pad(input_mask, padding, mode='constant', value=0)

    with torch.no_grad():
        # Iterate over the number of clusters to be added, and add them one by one
        for i in range(desired_number_of_clusters - number_of_clusters_founded):

            if gradual_padding and (i % 2 == 0):
                p += 8
                input_mask = F.pad(input_mask, (p, p, p, p), mode='constant', value=0)

            output = unet_model(input_mask.float())
            binary_outputs = torch.where(output > 0.5, 1, 0)

            out_mask = binary_outputs.clone()
            out_mask[:, number_of_clusters_founded + i, :, :] = binary_outputs[:, -1, :, :]

            out_mask = out_mask.squeeze(0)
            for m in range(out_mask.size(0)):
                try:
                    merged_mask = blob_merger(out_mask[m, :].reshape(input_mask.size(-1), input_mask.size(-1)).numpy())
                    merged_mask = merged_mask / 255
                    out_mask[m, :, :] = torch.tensor(merged_mask, dtype=torch.int64)
                except:
                    continue

            out_mask = out_mask.unsqueeze(0)
            input_mask = out_mask[:, :in_channels, :, :]

    if gradual_padding:
        bbox = find_tight_bbox(out_mask)
        square_mask = make_square_crop(out_mask, bbox)
        square_mask_float = square_mask.float()
        resized_tensor = F.interpolate(square_mask_float, size=(32, 32), mode='nearest')
        binary_tensor = (resized_tensor > 0.5).int()
        out_mask = binary_tensor

    mask_outputs = out_mask[0, :desired_number_of_clusters].view(desired_number_of_clusters, -1)
    return mask_outputs


def relayout_overgeneration(torch_object_masks, num_objects_to_remove):
    sum_per_channel = torch.sum(torch_object_masks, dim=(1, 2))

    # Remove the objects with the smallest sum
    _, indices = torch.sort(sum_per_channel, descending=True)
    indices = indices[:-num_objects_to_remove]

    return torch_object_masks[indices]








