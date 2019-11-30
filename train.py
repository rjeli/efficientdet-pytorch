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
    --save=<n>      Save every n epochs
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
from model import EfficientDet, StupidNet, NoBifpnNet
from detection_utils import transform_boxes

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 1.5

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

aug_transform = transforms.Compose([
    transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.05, hue=.05),
    img_transform,
])

def calc_loss(preds, gt_boxes, gt_classes, score_masks,
              apply_mean=True):
    mask = torch.zeros_like(score_masks, dtype=torch.float32).cuda()
    mask[score_masks == 2] = 1.

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
    # print(mask.shape, coord_loss.shape)
    # coord_loss *= mask
    # print(coord_loss.shape)

    obj_bce = F.binary_cross_entropy_with_logits(
        preds[0][:,0], gt_boxes[:,0], reduction='none')
    pt = torch.exp(-obj_bce)
    obj_loss = FOCAL_ALPHA * (1-pt)**FOCAL_GAMMA * obj_bce
    obj_loss = obj_bce

    cls_loss = F.cross_entropy(preds[1], gt_classes, reduction='none')
    # cls_loss[score_masks != 2] = 0
    cls_loss *= mask

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
        # ds = torch.utils.data.Subset(ds, [0])
        vds = torch.utils.data.Subset(vds, ds_idxs[:len(vds)])
    if args['--n']:
        n_samples = int(args['--n'])
        ds_idxs = list(range(n_samples))
        if n_samples < 64:
            print('small num samples, stretching for better perf')
            ds_idxs = ds_idxs * (64//n_samples)
        ds = torch.utils.data.Subset(ds, ds_idxs)
        # ds = torch.utils.data.Subset(ds, [0] * n_samples)
        vds_idxs = range(min(len(vds), max(64, n_samples)))
        vds = torch.utils.data.Subset(vds, vds_idxs)
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

    save = 10
    if args['--save']:
        save = int(args['--save'])

    lr = float(args['--lr'])

    profiling = args['--profile']

    dl = torch.utils.data.DataLoader(ds,
        batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    vdl = torch.utils.data.DataLoader(vds,
        batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)

    model = EfficientDet().cuda()
    # model = StupidNet().cuda()
    # model = NoBifpnNet().cuda()

    # opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)

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

            coord_losses = []
            obj_losses = []
            cls_losses = []

            for i, cell_sz in enumerate(CELL_SZS):
                coord_loss, obj_loss, cls_loss = calc_loss(preds[i], 
                    boxes[i].cuda(), classes[i].cuda(), score_masks[i])
                total_coord_loss += coord_loss
                total_obj_loss += obj_loss
                total_cls_loss += cls_loss

                coord_losses.append(coord_loss)
                obj_losses.append(obj_loss)
                cls_losses.append(cls_loss)

                # total_coord_loss.backward()
                # total_obj_loss.backward()
                # total_cls_loss.backward()

            coord_losses = torch.stack(coord_losses).sum()
            obj_losses = torch.stack(obj_losses).sum()
            cls_losses = torch.stack(cls_losses).sum()

            # total_loss = total_coord_loss + total_obj_loss + total_cls_loss
            # total_loss = total_cls_loss
            total_loss = coord_losses + obj_losses + cls_losses
            # total_loss = obj_losses

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
                if any([torch.isnan(x).any() for x in boxes]):
                    print('boxes nan')
                if any([torch.isnan(x).any() for x in classes]):
                    print('classes nan')
                if any([torch.isnan(x).any() for x in score_masks]):
                    print('sm nan')
                preds = model(imgs)
                if any([torch.isnan(x[0]).any() for x in preds]):
                    print('pred0 nan')
                if any([torch.isnan(x[1]).any() for x in preds]):
                    print('pred1 nan')

                total_coord_loss = 0.
                total_obj_loss = 0.
                total_cls_loss = 0.

                for i, cell_sz in enumerate(CELL_SZS):
                    # pboxes = preds[i][0]
                    # clip w, h to prevent instability
                    # pboxes[:,3:5] = torch.clamp(pboxes[:,3:5], max=3)
                    coord_loss, obj_loss, cls_loss = calc_loss(preds[i], 
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

        if epoch % save == 0:
            save_path = epochs_dir / f'{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(), 
                'opt': opt.state_dict(),
            }, str(save_path))
            print('saved to', save_path)

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    main(args)
