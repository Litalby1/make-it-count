import torch
import torch.nn.functional as F

def object_layout_loss(object_attention_map, config, desired_mask, attnstore):
    foreground_mask = (desired_mask != 0).to(dtype=object_attention_map.dtype)
    loss_cross = 0
    # Cross Attention Loss
    if config['cross_loss_step_range'][0] <= attnstore.curr_step_index <= config['cross_loss_step_range'][1] \
        and config['cross_loss_weight'] > 0:
        if config['cross_attn_loss_type'] == 'bce_logits':
            object_attention_map = (object_attention_map - torch.min(object_attention_map)) / (torch.max(object_attention_map) - torch.min(object_attention_map))
            loss_cross = F.binary_cross_entropy_with_logits(object_attention_map, foreground_mask, pos_weight=torch.tensor(config['cross_attn_bce_pos_wt']).to(object_attention_map.device))
        elif config['cross_attn_loss_type'] == 'bce':
            min_value, max_value = 0.05, 0.95
            object_attention_map = (object_attention_map - torch.min(object_attention_map)) * (max_value - min_value) / (torch.max(object_attention_map) - torch.min(object_attention_map)) + min_value
            loss_cross = F.binary_cross_entropy(object_attention_map, foreground_mask)

    loss = config['cross_loss_weight'] * loss_cross

    return loss
