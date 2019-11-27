#!/usr/bin/env python3
"""
Usage:
    ./debug.py <cmd> [--which=<n>]
"""
from docopt import docopt

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
        img, box, cls, score_masks = ds[which]
        img = img.cuda()
        preds = model(img[None])[0]

