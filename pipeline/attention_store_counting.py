from typing import List
import numpy as np
import torch
from collections import defaultdict
from pipeline.mask_extraction.utils_masks import to_channels

class CrossAndSelfAttentionStore:
    @staticmethod
    def get_empty_step_store(save_timesteps=None):
        d = defaultdict(list)
        for t in save_timesteps:
            d[t] = {}
        return d

    @staticmethod
    def get_empty_store():
        return {}
    def __init__(self, attn_res, save_timesteps, loss, masking_dict, desired_mask, object_token_idx):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.curr_step_index = 0
        self.count_ts = 0

        if save_timesteps == "all":
            self.save_timesteps = list(range(50))
        else:
            self.save_timesteps = save_timesteps

        self.attn_res = (attn_res, attn_res)
        self.loss = loss
        self.masking_dict = masking_dict

        self.desired_mask = desired_mask
        self.object_token_idx = object_token_idx

        if desired_mask is not None:
            self.channels_desired_mask = to_channels(self.desired_mask).to(device=self.desired_mask.device).bool()

        # Step store to save attention at defined steps, only when not training (self.loss = False)
        self.cross_step_store = self.get_empty_step_store(self.save_timesteps)
        self.self_step_store = self.get_empty_step_store(self.save_timesteps)
        
        # Attention store to save attention ONLY for the current step, 
        self.cross_attention_store = self.get_empty_store()
        self.self_attention_store = self.get_empty_store()

        self.all_cross_attention = {}
        self.all_self_attention = {}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads):
        if (self.save_timesteps is None) or (self.curr_step_index in self.save_timesteps) or self.loss:
            if (attn.shape[1] == np.prod(self.attn_res)) and (self.cur_att_layer >= 0):
                
                if (not self.loss) and (is_cross):
                    guided_attn = attn[attn.size(0)//2:]
                else:
                    guided_attn = attn

                if is_cross:
                    guided_attn = guided_attn.reshape([guided_attn.shape[0]//attn_heads, attn_heads, *guided_attn.shape[1:]]).mean(dim=1)

                    self.cross_attention_store[place_in_unet] = guided_attn
                    if self.curr_step_index in self.save_timesteps and not self.loss:
                        self.cross_step_store[self.curr_step_index][place_in_unet] = guided_attn
                else:
                    guided_attn = guided_attn.reshape([guided_attn.shape[0]//attn_heads, attn_heads, *guided_attn.shape[1:]]).mean(dim=1)

                    self.self_attention_store[place_in_unet] = guided_attn
                    if self.curr_step_index in self.save_timesteps and not self.loss:
                        self.self_step_store[self.curr_step_index][place_in_unet] = guided_attn

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.all_cross_attention = self.cross_attention_store
        self.cross_attention_store = self.get_empty_store()
        self.all_self_attention = self.self_attention_store
        self.self_attention_store = self.get_empty_store()


    def aggregate_attention(self, from_where: List[str], get_cross=True) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        if get_cross:
            attention_maps = self.all_cross_attention
        else:
            attention_maps = self.all_self_attention
        
        for layer, curr_map in attention_maps.items():
            if any([x in layer for x in from_where]):
                curr_map_reshape = curr_map.reshape(-1, self.attn_res[0], self.attn_res[1], curr_map.shape[-1])
                out.append(curr_map_reshape)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

