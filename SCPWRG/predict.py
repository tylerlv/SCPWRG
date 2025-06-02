import argparse
import logging
import os
import torch.nn as nn
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask
from metrics import *
from Network.model import *
from medpy.metric import binary


dir_img = Path('./test/imgs')
dir_mask = Path('./test/masks')

file = open('Metrices.txt', 'a+')


def predict_img(net, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
    val_set = BasicDataset(dir_img, dir_mask, scale_factor)
    data_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    num_test_batches = len(data_loader)

    Dice_score = 0
    JI_score = 0
    ASSD_score = 0
    HD95_score = 0
    Sensitivity = 0
    Specificity = 0

    for j, batch in enumerate(data_loader, 1):
        images = batch['image'].to(device=device, dtype=torch.float32)
        images_patch = batch['image_patch'].to(device=device, dtype=torch.float32)
        images_patch = images_patch.view(images_patch.shape[1], images_patch.shape[2], images_patch.shape[3], images_patch.shape[4])

        mask_type = torch.float32
        true_masks = batch['mask'].to(device=device, dtype=mask_type)
        true_masks_patch = batch['mask_patch'].to(device=device, dtype=mask_type)
        true_masks = true_masks.view(-1, 1, true_masks.shape[1], true_masks.shape[2])
        true_masks_patch = true_masks_patch.view(-1, 1, true_masks_patch.shape[2], true_masks_patch.shape[3])

        adj = batch['adj'].to(device=device)
        adj = adj.view(adj.shape[1], adj.shape[2])

        masks_pred, patches_pred, _ = net(images, images_patch, adj)

        pred = (masks_pred > out_threshold).cpu().detach().numpy()
        true_masks_np = true_masks.cpu().detach().numpy()

        Dice_score += binary.dc(pred, true_masks_np)
        JI_score += binary.jc(pred, true_masks_np)
        ASSD_score += binary.assd(pred, true_masks_np)
        HD95_score += binary.hd95(pred, true_masks_np)
        Sensitivity += binary.sensitivity(pred, true_masks_np)
        Specificity += binary.specificity(pred, true_masks_np)

        logging.info(f'Processed batch {j}/{num_test_batches}')

    Dice_score /= num_test_batches
    JI_score /= num_test_batches
    ASSD_score /= num_test_batches
    HD95_score /= num_test_batches
    Sensitivity /= num_test_batches
    Specificity /= num_test_batches

    file.write(
        f"Dice: {Dice_score}\n"
        f"JI: {JI_score}\n"
        f"ASSD: {ASSD_score}\n"
        f"HD95: {HD95_score}\n"
        f"Sensitivity: {Sensitivity}\n"
        f"Specificity: {Specificity}\n"
    )
    logging.info(f"Metrics - Dice: {Dice_score}, JI: {JI_score}, ASSD: {ASSD_score}, HD95: {HD95_score}, Sensitivity: {Sensitivity}, Specificity: {Specificity}")


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoint/checkpoint_epoch501.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', default=False,
                        help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling')

    return parser.parse_args()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    net = Model(1, 1, 8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)

    checkpoint_path = f'./checkpoints/checkpoint_epoch500.pth'
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logging.info(f'Model loaded from {checkpoint_path}')
    file.write(f'{epoch} - Model loaded\n')
    predict_img(net=net, device=device, scale_factor=args.scale)

    file.close()