#!/usr/bin/env python3
"""
Usage:
    ./train.py [--pct=<pct>] [--bs=<bs>] [--workers=<workers>]
"""

from tqdm import tqdm
from docopt import docopt
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter

from config import INPUT_SZ, CELL_SZS, CELL_CENTERS
from dataset import CocoDataset
from model import EfficientDet

import sys
sys.path.insert(0, str(Path.home() / 'coco' / 'cocoapi' / 'PythonAPI'))
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def main(args):
    writer = SummaryWriter()

    assert torch.cuda.is_available()

    ds = CocoDataset(is_train=True, img_size=INPUT_SZ, 
        img_transform=img_transform)

    if args['--pct']:
        split_sz = int(int(args['--pct']) / 100 * len(ds))
        ds, _ = torch.utils.data.random_split(ds, [split_sz, len(ds)-split_sz])
    print('len(ds):', len(ds))

    bs = 8
    if args['--bs']:
        bs = int(args['--bs'])

    workers = 0
    if args['--workers']:
        workers = int(args['--workers'])

    dl = torch.utils.data.DataLoader(ds,
        batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)

    model = EfficientDet().cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(100):
        print('epoch', epoch)
        model.train()
        epoch_train_loss = 0.
        epoch_coord_loss = 0.
        epoch_obj_loss = 0.
        epoch_cls_loss = 0.

        start_t = time.time()
        sec_spent_waiting = 0.
        t0 = time.time()

        for imgs, boxes, classes, score_masks in tqdm(dl):
            t1 = time.time()
            sec_spent_waiting += (t1-t0)
            waiting_frac = sec_spent_waiting / (t1-start_t)

            imgs = imgs.cuda()
            preds = model(imgs)
            
            total_coord_loss = 0.
            total_obj_loss = 0.
            total_cls_loss = 0.

            for i, cell_sz in enumerate(CELL_SZS):
                boxes[i] = boxes[i].cuda()
                classes[i] = classes[i].cuda()

                cell_x_centers = CELL_CENTERS[i][1]
                cell_y_centers = CELL_CENTERS[i][0]
                pred_xs = torch.sigmoid(preds[i][0][:,1])*cell_sz + cell_x_centers
                pred_ys = torch.sigmoid(preds[i][0][:,2])*cell_sz + cell_y_centers
                pred_ws = cell_sz * torch.exp(preds[i][0][:,3])
                pred_hs = cell_sz * torch.exp(preds[i][0][:,4])
                coord_loss  = F.mse_loss(pred_xs, boxes[i][:,1], reduction='none')
                coord_loss += F.mse_loss(pred_ys, boxes[i][:,2], reduction='none')
                coord_loss += F.mse_loss(
                    torch.sqrt(pred_ws), torch.sqrt(boxes[i][:,3]))
                coord_loss += F.mse_loss(
                    torch.sqrt(pred_hs), torch.sqrt(boxes[i][:,4]))
                coord_loss[score_masks[i] == 0] = 0
                coord_loss *= 5
                total_coord_loss += torch.sum(coord_loss) / bs

                obj_loss = F.binary_cross_entropy_with_logits(
                    preds[i][0][:,0], boxes[i][:,0], reduction='none')
                total_obj_loss += torch.sum(obj_loss) / bs

                cls_loss = F.cross_entropy(
                    preds[i][1], classes[i], reduction='none')
                cls_loss[score_masks[i] == 0] = 0
                total_cls_loss += torch.sum(cls_loss) / bs

            total_loss = total_coord_loss + total_obj_loss + total_cls_loss
            total_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            epoch_train_loss += total_loss.item()
            epoch_coord_loss += total_coord_loss.item()
            epoch_obj_loss += total_obj_loss.item()
            epoch_cls_loss += total_cls_loss.item()

            t0 = time.time()

        print('epoch train loss:', epoch_train_loss)
        writer.add_scalar('train_loss', epoch_train_loss, epoch)
        writer.add_scalar('coord_loss', epoch_train_loss, epoch)
        writer.add_scalar('obj_loss', epoch_obj_loss, epoch)
        writer.add_scalar('cls_loss', epoch_cls_loss, epoch)

        torch.save(model.state_dict(), f'epoch{epoch}.pt')

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    main(args)
