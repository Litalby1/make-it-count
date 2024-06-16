import torch.nn as nn
import torch


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=None, smooth=0, intersection_penalty=0.5, dice_penalty=0.5, existing_mask_weights=1,
                 extra_mask_weight=1, intersection_loss=False):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = weights if weights else []
        self.intersection_penalty = intersection_penalty
        self.intersection_loss = intersection_loss
        self.dice_penalty = dice_penalty
        self.existing_mask_weights = existing_mask_weights
        self.extra_mask_weight = extra_mask_weight

    def dice_coeff(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        return (2 * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)

    def forward(self, predict, target, weights=None):
        self.weights = weights
        if self.weights.size(1) != predict.size(1):
            raise ValueError(f"Weights size mismatch: got {len(self.weights)} weights, expected {predict.size(1)}.")

        total_loss = 0
        total_dice_loss = 0
        total_intersection_loss = 0

        for j in range(predict.size(0)):
            for i in range(predict.size(1)):
                if weights[j][i] > 0:
                    dice_score = self.dice_coeff(predict[j, i, :, :], target[j, i, :, :])
                    # Dice loss between predicted and target mask
                    dice_loss = (1 - dice_score) * self.dice_penalty
                    total_loss += dice_loss
                    total_dice_loss += dice_loss
                    if self.intersection_loss:
                        for k in range(i + 1, predict.size(1)):
                            if weights[j][k] > 0:
                                # Intersection loss between all pairs of predicted masks
                                intersection_score = self.dice_coeff(predict[j, i, :, :], predict[j, k, :, :])
                                intersection_loss = self.intersection_penalty * intersection_score
                                total_loss += intersection_loss
                                total_intersection_loss += intersection_loss
                    else:
                        total_intersection_loss = torch.tensor(0)

        return total_loss, total_dice_loss, total_intersection_loss

