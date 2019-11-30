#!/usr/bin/env python3
"""
Usage:
    ./debug.py <cmd> [--which=<n>]
"""
from docopt import docopt

import torch

from dataset import CocoDataset
from config import INPUT_SZ, CELL_SZS
from train import img_transform, calc_loss

if __name__ == '__main__':
    args = docopt(__doc__)
    which = 0
    if args['--which']:
        which = int(args['--which'])

    if args['<cmd>'] == 'visloss':
        from train import calc_loss
        from model import EfficientDet
        model = EfficientDet().cuda()
        ds = CocoDataset(is_train=True, img_size=INPUT_SZ, 
            img_transform=img_transform)
        img, boxes, classes, score_masks = ds[which]
        img = img.cuda()
        preds = model(img[None])
        for i, p in enumerate(preds):
            z_obj = torch.zeros_like(p[0])
            z_obj[:,0] = -100
            z_cls = torch.zeros_like(p[1])
            sm = torch.from_numpy(score_masks[i][None])
            coord_loss, obj_loss, cls_loss = calc_loss((z_obj, z_cls),
                boxes[i][None].cuda(), classes[i][None].cuda(), sm,
                apply_mean=False)
            print(i, coord_loss, obj_loss, cls_loss)

            print(coord_loss.shape)
            import matplotlib.pyplot as plt

            fig = plt.figure()
            plt.imshow(coord_loss.cpu().numpy()[0], cmap='viridis')
            plt.colorbar()
            fig.savefig(f'coord{i}.png')

            fig = plt.figure()
            plt.imshow(obj_loss.cpu().numpy()[0], cmap='viridis')
            plt.colorbar()
            fig.savefig(f'obj{i}.png')

            fig = plt.figure()
            plt.imshow(cls_loss.cpu().numpy()[0], cmap='viridis')
            plt.colorbar()
            fig.savefig(f'cls{i}.png')

            print('AA', p[0][:,0].shape, boxes[i][None].cuda()[:,0].shape)

