#!/usr/bin/env python3
"""
Usage:
    ./train.py [options]

Options:
    --pct=<pct>     Percent of dataset to use
    --n=<n>         Number of samples from dataset to use
    --bs=<n>        Batch size
    --workers=<n>   Number of data loader workers
    --epochs=<n>    Number of epochs
    --resume=<path> Resume from checkpoint
    --profile       Use torch.autograd.profiler
    --lr=<lr>       Learning rate [default: 1e-3]
"""

from tqdm import tqdm
from docopt import docopt
import time
from pathlib import Path
import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter

from config import INPUT_SZ, CELL_SZS, CELL_CENTERS
from dataset import CocoDataset
from model import EfficientDet
from detection_utils import transform_boxes

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 1.5

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def calc_loss(cell_sz, preds, gt_boxes, gt_classes, score_masks,
              apply_mean=True):
    # x, y, w, h
    coord_loss = F.mse_loss(
        torch.sigmoid(preds[0][:,1]), gt_boxes[:,1], reduction='none')
    coord_loss += F.mse_loss(
        torch.sigmoid(preds[0][:,2]), gt_boxes[:,2], reduction='none')
    coord_loss += F.mse_loss(
        torch.exp(preds[0][:,3]), gt_boxes[:,3])
    coord_loss += F.mse_loss(
        torch.exp(preds[0][:,4]), gt_boxes[:,4])
    coord_loss[score_masks != 2] = 0
    coord_loss *= 5

    obj_bce = F.binary_cross_entropy_with_logits(
        preds[0][:,0], gt_boxes[:,0], reduction='none')
    pt = torch.exp(-obj_bce)
    obj_loss = FOCAL_ALPHA * (1-pt)**FOCAL_GAMMA * obj_bce

    cls_loss = F.cross_entropy(preds[1], gt_classes, reduction='none')
    cls_loss[score_masks != 2] = 0

    losses = (coord_loss, obj_loss, cls_loss)
    return tuple(l.mean() for l in losses) if apply_mean else losses

def main(args):
    torch.manual_seed(0)
    writer = SummaryWriter()

    assert torch.cuda.is_available()

    ds = CocoDataset(is_train=True, img_size=INPUT_SZ, 
        img_transform=img_transform)
    vds = CocoDataset(is_train=False, img_size=INPUT_SZ, 
        img_transform=img_transform)

    print('len(ds):', len(ds), 'len(vds):', len(vds))

    if args['--pct']:
        pct = int(args['--pct']) / 100
        ds_idxs = range(int(pct * len(ds)))
        ds = torch.utils.data.Subset(ds, ds_idxs)
        vds_idxs = range(max(1024, int(pct * len(vds))))
        # vds = torch.utils.data.Subset(vds, vds_idxs)
    if args['--n']:
        n_samples = int(args['--n'])
        ds = torch.utils.data.Subset(ds, range(n_samples))
        vds = torch.utils.data.Subset(vds, range(n_samples))
    print('using len(ds):', len(ds), 'len(vds):', len(vds))

    bs = 8
    if args['--bs']:
        bs = int(args['--bs'])

    workers = 0
    if args['--workers']:
        workers = int(args['--workers'])

    epochs = 100
    if args['--epochs']:
        epochs = int(args['--epochs'])

    lr = float(args['--lr'])

    profiling = args['--profile']

    dl = torch.utils.data.DataLoader(ds,
        batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    vdl = torch.utils.data.DataLoader(vds,
        batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)

    model = EfficientDet(drop_rate=0, drop_connect_rate=0).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    nowstr = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    epochs_dir = Path('.') / 'epochs' / nowstr
    epochs_dir.mkdir()

    start_epoch = 0
    if args['--resume']:
        ckpt = torch.load(args['--resume'])
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        print('resuming from', start_epoch)

    for epoch in range(start_epoch, epochs):
        start_t = time.time()
        sec_spent_waiting = 0.
        t0 = time.time()

        model.train()

        epoch_loss_t = 0.
        epoch_coord_loss_t = 0.
        epoch_obj_loss_t = 0.
        epoch_cls_loss_t = 0.

        if profiling:
            prof = torch.autograd.profiler.profile(use_cuda=True)
            prof.__enter__()

        pbar = tqdm(dl)
        for imgs, boxes, classes, score_masks in pbar:
            t1 = time.time()
            sec_spent_waiting += (t1-t0)
            waiting_frac = sec_spent_waiting / (t1-start_t)

            imgs = imgs.cuda()
            preds = model(imgs)
            
            total_coord_loss = 0.
            total_obj_loss = 0.
            total_cls_loss = 0.

            for i, cell_sz in enumerate(CELL_SZS):
                coord_loss, obj_loss, cls_loss = calc_loss(cell_sz, preds[i], 
                    boxes[i].cuda(), classes[i].cuda(), score_masks[i])
                total_coord_loss += coord_loss
                total_obj_loss += obj_loss
                total_cls_loss += cls_loss

            total_loss = total_coord_loss + total_obj_loss + total_cls_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            epoch_loss_t += total_loss.item() / len(ds)
            epoch_coord_loss_t += total_coord_loss.item() / len(ds)
            epoch_obj_loss_t += total_obj_loss.item() / len(ds)
            epoch_cls_loss_t += total_cls_loss.item() / len(ds)

            t0 = time.time()
            pbar.set_description(
                f'{epoch} train loss:{epoch_loss_t:.3e} wf:{waiting_frac:.2f}')

        if profiling:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(f'prof{epoch}.chrometrace')

        writer.add_scalar('train_loss', epoch_loss_t, epoch)
        writer.add_scalar('coord_loss', epoch_coord_loss_t, epoch)
        writer.add_scalar('obj_loss', epoch_obj_loss_t, epoch)
        writer.add_scalar('cls_loss', epoch_cls_loss_t, epoch)
        writer.flush()

        model.eval()

        epoch_loss_v = 0.
        epoch_coord_loss_v = 0.
        epoch_obj_loss_v = 0.
        epoch_cls_loss_v = 0.

        with torch.no_grad():
            pbar = tqdm(vdl)
            for imgs, boxes, classes, score_masks in pbar:
                imgs = imgs.cuda()
                preds = model(imgs)

                total_coord_loss = 0.
                total_obj_loss = 0.
                total_cls_loss = 0.

                for i, cell_sz in enumerate(CELL_SZS):
                    coord_loss, obj_loss, cls_loss = calc_loss(cell_sz, preds[i], 
                        boxes[i].cuda(), classes[i].cuda(), score_masks[i])
                    total_coord_loss += coord_loss
                    total_obj_loss += obj_loss
                    total_cls_loss += cls_loss

                total_loss = total_coord_loss + total_obj_loss + total_cls_loss

                epoch_loss_v += total_loss.item() / len(vds)
                epoch_coord_loss_v += total_coord_loss.item() / len(vds)
                epoch_obj_loss_v += total_obj_loss.item() / len(vds)
                epoch_cls_loss_v += total_cls_loss.item() / len(vds)
                pbar.set_description(
                    f'{epoch} val loss:{epoch_loss_v:.3e}')

        writer.add_scalar('val_loss', epoch_loss_v, epoch)
        writer.add_scalar('val_coord_loss', epoch_coord_loss_v, epoch)
        writer.add_scalar('val_obj_loss', epoch_obj_loss_v, epoch)
        writer.add_scalar('val_cls_loss', epoch_cls_loss_v, epoch)
        writer.flush()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(), 
                'opt': opt.state_dict(),
            }, str(epochs_dir/f'{epoch}.pt'))

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    main(args)
