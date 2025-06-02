import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from metrics import *


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    JI_score = 0
    ind = 0

    for batch in dataloader:
        ind += 1
        images = batch['image']
        true_masks = batch['mask']
        images_patch = batch['image_patch']
        true_masks_patch = batch['mask_patch']
        adj = batch['adj']
        images = images.to(device=device, dtype=torch.float32)
        images_patch = images_patch.to(device=device, dtype=torch.float32)
        images_patch = images_patch.view([images_patch.shape[1], images_patch.shape[2], images_patch.shape[3], images_patch.shape[4]])
        mask_type = torch.float32
        true_masks = true_masks.to(device=device, dtype=mask_type)
        true_masks_patch = true_masks_patch.to(device=device, dtype=mask_type)
        true_masks = true_masks.view([-1, 1, true_masks.shape[1], true_masks.shape[2]])
        true_masks_patch = true_masks_patch.view([-1, 1, true_masks_patch.shape[2], true_masks_patch.shape[3]])
        adj = adj.to(device=device)
        adj = adj.view([adj.shape[1], adj.shape[2]])

        with torch.no_grad():
            mask_pred, patches_pred, _ = net(images, images_patch, adj)
            mask_pred = (mask_pred > 0.5).float()
            dice_score += calDSI(mask_pred, true_masks)

    net.train()

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, JI_score / num_val_batches
