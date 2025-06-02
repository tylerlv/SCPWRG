import torch
from skimage.util.shape import view_as_blocks
import argparse
import logging
import os
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from Network.model import Model
from utils.loss import DiceLoss


dir_img_train = Path('./train/Img')
dir_mask_train = Path('./train/Mask')
dir_checkpoint = Path('./checkpoints/')


def view_as_block_torch(arr_in, block_shape):
    arr_in = arr_in.cpu().detach().numpy()
    blocks = view_as_blocks(arr_in, block_shape).reshape([-1, 1, *block_shape])
    arr_out = torch.from_numpy(blocks).to(device)
    return arr_out


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    dataset = BasicDataset(dir_img_train, dir_mask_train, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCELoss()
    criterion_Dice = DiceLoss()
    criterion_MSE = nn.MSELoss()

    global_step = 0

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0

        for batch in train_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            images_patch = batch['image_patch'].to(device=device, dtype=torch.float32)
            images_patch = images_patch.view(images_patch.shape[1], images_patch.shape[2], images_patch.shape[3], images_patch.shape[4])

            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            true_masks_patch = batch['mask_patch'].to(device=device, dtype=torch.float32)
            true_masks = true_masks.view(-1, 1, true_masks.shape[1], true_masks.shape[2])
            true_masks_patch = true_masks_patch.view(-1, 1, true_masks_patch.shape[2], true_masks_patch.shape[3])

            multi_masks = images.clone()

            adj = batch['adj'].to(device=device)
            adj = adj.view(adj.shape[1], adj.shape[2])

            masks_pred, patches_pred, multi_pred = net(images, images_patch, adj)

            masks_pred_patches = view_as_block_torch(masks_pred.view(256, 256), block_shape=(32, 32))

            lambda_1 = 0.1
            eta = 1

            loss_ws = criterion_Dice(masks_pred, true_masks)
            loss_rec = criterion_MSE(multi_pred, multi_masks)
            loss_ps = criterion(patches_pred, true_masks_patch)

            loss_aux = loss_rec + eta * loss_ps

            loss = loss_ws + lambda_1 * loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            division_step = (n_train // batch_size)
            if division_step > 0 and epoch % 10 == 1 and global_step % division_step == 0:
                val_score = evaluate(net, val_loader, device)
                logging.info(f'Validation Dice score: {val_score}')
                scheduler.step(val_score)

        logging.info(f'Epoch {epoch} Loss: {epoch_loss / n_train:.6f}')

        if save_checkpoint and epoch % 10 == 1:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            checkpoint_path = dir_checkpoint / f'checkpoint_epoch{epoch}.pth'
            torch.save(net.state_dict(), str(checkpoint_path))
            logging.info(f'Checkpoint {epoch} saved at {checkpoint_path}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0, help='Percent of data used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = Model(3, 1, 64)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Training interrupted. Model saved as INTERRUPTED.pth')
        raise