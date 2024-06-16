import torch
from pipeline.mask_extraction.dbscan_mask_extract import dbscan_extract_mask
from pipeline.mask_extraction.postprocess import run_postprocess
from pipeline.mask_extraction.relayout import relayout_undergeneration, relayout_overgeneration
from pipeline.mask_extraction.utils_masks import from_channels, to_channels, remove_sparse_blobs


def relayout(sdxl_pipe, prompt, required_object_num, config, seed):
    # Run vanilla + dbscan
    obj_num_match = False

    vanilla_masks, n_dbscan_clusters, vanilla_img = dbscan_extract_mask(prompt, sdxl_pipe, config, seed)

    vanilla_masks, number_object_remain = remove_sparse_blobs(vanilla_masks)

    n_dbscan_clusters = number_object_remain

    vanilla_masks = torch.tensor(vanilla_masks, dtype=torch.float16).view(32, 32)
    vanilla_masks += 1

    if n_dbscan_clusters == 0:  # If no object is found, add a single object in the center
        vanilla_masks[13:19, 13:19] = 1
        n_dbscan_clusters = 1

    if n_dbscan_clusters == required_object_num:
        correct_number_mask = vanilla_masks
        obj_num_match = True

    else:
        torch_object_masks = to_channels(vanilla_masks)

        if n_dbscan_clusters > required_object_num:
            object_masks = relayout_overgeneration(torch_object_masks, n_dbscan_clusters - required_object_num)

        elif n_dbscan_clusters < required_object_num:
            object_masks = relayout_undergeneration(torch_object_masks, number_of_clusters_founded=n_dbscan_clusters,
                                        desired_number_of_clusters=required_object_num, config=config)

        correct_number_mask = from_channels(object_masks)

    post_process_mask = run_postprocess(correct_number_mask)

    return vanilla_masks, correct_number_mask, post_process_mask, vanilla_img, obj_num_match



