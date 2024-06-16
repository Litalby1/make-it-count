from __future__ import annotations
import torch
from diffusers.models.attention import Attention

class CountingProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.curr_step = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Store Attention
        self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads)

        # Self Attention Masking
        attn_dim = self.attnstore.attn_res[0]
        if self.attnstore.loss and self.attnstore.masking_dict['enable'] and \
            (attention_probs.shape[0] == 40 and attention_probs.shape[2] == attn_dim**2
             and self.attnstore.masking_dict['start_step'] <= self.attnstore.curr_step_index <=
             self.attnstore.masking_dict['end_step']
             and 'up' in self.place_in_unet):
                
            max_blob_index = torch.max(self.attnstore.desired_mask).int().item()
            attention_probs = attention_probs.view(40, attn_dim, attn_dim, attn_dim**2)
           
            blob_coordinates = torch.zeros((attn_dim, attn_dim), device=attention_probs.device)

            for j in range(0, max_blob_index + 1):
                    current_blob_mask = self.attnstore.desired_mask == j
                    indices = current_blob_mask.nonzero(as_tuple=False)
                    if len(indices) > 0:
                        # Update blob_coordinates tensor
                        blob_coordinates[indices[:, 0], indices[:, 1]] = 1

            # Flatten the blob_coordinates for easier indexing
            blob_coordinates_flat = blob_coordinates.view(-1).bool()

            # Find indices of the blob to modify in attention_probs
            blobs_indexes = (self.attnstore.desired_mask == 0).nonzero(as_tuple=False)

            for idx in blobs_indexes:
                x, y = idx[0], idx[1]
                # Zero out the selected positions across all 40 channels for this blob
                attention_probs[:, x, y, blob_coordinates_flat] = 0

            attention_probs = attention_probs.reshape(40, attn_dim**2, attn_dim**2)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states