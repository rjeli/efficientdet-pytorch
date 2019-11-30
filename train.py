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
    --max-lr=<lr>   Max learning rate [default: 1e-3]
    --save=<n>      Save every n epochs
    --find-lr       Run lr finder
    --flips         Augment with flips
    --parallel
    --data-dir=<d>  Data directory [default: /data]
"""

from tqdm import tqdm
from docopt import docopt
import time
from pathlib import Path
import datetime
import math
import itertools
import re
import numpy as np
import uuid
import pytz
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter

from config import INPUT_SZ, CELL_SZS, CELL_CENTERS, COCO_PATH
from dataset import CocoDataset
from model import EfficientDet, StupidNet, NoBifpnNet
from detection_utils import transform_boxes
from cython_utils import add_coco_preds

sys.path.append(str(Path.home() / 'coco' / 'cocoapi' / 'PythonAPI'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
    coord_loss *= 10 * mask

    obj_bce = F.binary_cross_entropy_with_logits(
        preds[0][:,0], gt_boxes[:,0], reduction='none')
    pt = torch.exp(-obj_bce)
    obj_loss = FOCAL_ALPHA * (1-pt)**FOCAL_GAMMA * obj_bce

    cls_loss = F.cross_entropy(preds[1], gt_classes, reduction='none')
    cls_loss *= mask

    losses = (coord_loss, obj_loss, cls_loss)
    return tuple(l.mean() for l in losses) if apply_mean else losses

def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # don't decay batch norm weights
        if re.search('bn\d?\.(weight|bias)$', name):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.},
    ]

def short_sci(x):
    exp = int(np.floor(np.log10(x)))
    base = int(x/(10**exp))
    assert base*10**exp == x
    return f'{base}e{exp}'

def main(args):
    torch.manual_seed(0)
    data_dir = args['--data-dir']

    assert torch.cuda.is_available()

    ds = CocoDataset(is_train=True, img_size=INPUT_SZ, 
        img_transform=img_transform, flips=args['--flips'])
    vds = CocoDataset(is_train=False, img_size=INPUT_SZ, 
        img_transform=img_transform, return_info=True)

    print('len(ds):', len(ds), 'len(vds):', len(vds))

    pct, n_samples = None, None
    if args['--pct']:
        pct = int(args['--pct']) / 100
        ds_idxs = range(int(pct * len(ds)))
        ds = torch.utils.data.Subset(ds, ds_idxs)
        vds = torch.utils.data.Subset(vds, ds_idxs[:len(vds)])
    if args['--n']:
        n_samples = int(args['--n'])
        ds_idxs = list(range(n_samples))
        if n_samples < 64:
            print('small num samples, stretching for better perf')
            ds_idxs = ds_idxs * (64//n_samples)
        ds = torch.utils.data.Subset(ds, ds_idxs)
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

    max_lr = float(args['--max-lr'])

    profiling = args['--profile']

    dl = torch.utils.data.DataLoader(ds,
        batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    vdl = torch.utils.data.DataLoader(vds,
        batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)

    model = EfficientDet().cuda()
    if args['--parallel']:
        model = nn.DataParallel(model)

    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
    opt = torch.optim.AdamW(param_groups(model, weight_decay=1e-4))

    if args['--find-lr']:
        assert len(ds)//bs >= 100
        start_lr = 1e-7
        end_lr = 1e1
        lr_gamma = (end_lr/start_lr)**(1/99)
        print('gamma:', lr_gamma)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_gamma)

        lrs, losses = [], []
        pbar = tqdm(itertools.islice(dl, 100), total=100)
        for imgs, boxes, classes, score_masks in pbar:
            imgs = imgs.cuda()
            preds = model(imgs)
            coord_losses = []
            obj_losses = []
            cls_losses = []
            for i, cell_sz in enumerate(CELL_SZS):
                coord_loss, obj_loss, cls_loss = calc_loss(preds[i], 
                    boxes[i].cuda(), classes[i].cuda(), score_masks[i])
                coord_losses.append(coord_loss)
                obj_losses.append(obj_loss)
                cls_losses.append(cls_loss)
            total_loss = 0.
            total_loss += torch.stack(coord_losses).sum()
            total_loss += torch.stack(obj_losses).sum()
            total_loss += torch.stack(cls_losses).sum()
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            lrs.append(scheduler.get_lr()[0])
            losses.append(total_loss.item())
            print(lrs)
            print(losses)
            scheduler.step()

        # lrs = np.array(lrs)
        # losses = np.array(losses)

        """
        from scipy.interpolate import UnivariateSpline
        xx = np.linspace(np.log10(start_lr), np.log10(end_lr), num=len(losses))
        xx = xx[losses<10]
        losses2 = losses[losses<10]
        spl = UnivariateSpline(xx, losses2, s=len(xx)/3)
        """

        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(lrs, losses, 'r-')
        # plt.plot(10**xx, spl(xx), 'r-')
        # plt.ylim(0, 10)
        plt.xscale('log')
        fig.savefig('lrfinder.png')
        return

    start_epoch = 0
    if args['--resume']:
        ckpt = torch.load(args['--resume'])
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        print('resuming from', start_epoch)

    lr_sched = torch.optim.lr_scheduler.OneCycleLR(opt,
        max_lr=max_lr, div_factor=10, # start at lr/10
        steps_per_epoch=len(dl), epochs=epochs,
        last_epoch=start_epoch*len(dl)-1)

    name = ''
    now = datetime.datetime.now(pytz.timezone('US/Pacific'))
    name += now.strftime('%a').lower()
    name += now.strftime('%I%M%p').lower()
    name += '_'
    if pct is not None:
        name += f'{int(pct*100)}pct'
    elif n_samples is not None:
        name += f'{n_samples}samples'
    else:
        name += 'full'
    name += f'_{epochs}epochs'
    name += f'_bs{bs}'
    name += f'_maxlr{short_sci(max_lr)}'
    if args['--flips']:
        name += '_flips'
    if start_epoch != 0:
        name += f'_resume{start_epoch}'
    name += '_' + str(uuid.uuid4())[:4]
    print('name:', name)

    writer = SummaryWriter(Path('runs') / name)

    epochs_dir = Path(data_dir) / 'epochs' / name
    epochs_dir.mkdir()

    cocoGt = COCO(str(COCO_PATH/'annotations'/'instances_val2017.json'))

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

            coord_losses = []
            obj_losses = []
            cls_losses = []

            for i, cell_sz in enumerate(CELL_SZS):
                coord_loss, obj_loss, cls_loss = calc_loss(preds[i], 
                    boxes[i].cuda(), classes[i].cuda(), score_masks[i])
                coord_losses.append(coord_loss)
                obj_losses.append(obj_loss)
                cls_losses.append(cls_loss)

            coord_losses = torch.stack(coord_losses).sum()
            obj_losses = torch.stack(obj_losses).sum()
            cls_losses = torch.stack(cls_losses).sum()

            total_loss = coord_losses + obj_losses + cls_losses

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            lr_sched.step()

            epoch_loss_t += total_loss.item() / len(dl)
            epoch_coord_loss_t += coord_losses.item() / len(dl)
            epoch_obj_loss_t += obj_losses.item() / len(dl)
            epoch_cls_loss_t += cls_losses.item() / len(dl)

            t0 = time.time()
            pbar.set_description(
                f'{epoch} train loss:{epoch_loss_t:.3e} wf:{waiting_frac:.2f}')

        if profiling:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(f'prof{epoch}.chrometrace')

        writer.add_scalar('train_loss/total', epoch_loss_t, epoch)
        writer.add_scalar('train_loss/coord', epoch_coord_loss_t, epoch)
        writer.add_scalar('train_loss/obj', epoch_obj_loss_t, epoch)
        writer.add_scalar('train_loss/cls', epoch_cls_loss_t, epoch)
        writer.add_scalar('lr', lr_sched.get_lr()[0], epoch)
        writer.flush()

        model.eval()

        epoch_loss_v = 0.
        epoch_coord_loss_v = 0.
        epoch_obj_loss_v = 0.
        epoch_cls_loss_v = 0.

        coco_results = []

        with torch.no_grad():
            pbar = tqdm(vdl)
            for imgs, boxes, classes, score_masks, infos in pbar:
                imgs = imgs.cuda()
                preds = model(imgs)

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

                epoch_loss_v += total_loss.item() / len(vdl)
                epoch_coord_loss_v += total_coord_loss.item() / len(vdl)
                epoch_obj_loss_v += total_obj_loss.item() / len(vdl)
                epoch_cls_loss_v += total_cls_loss.item() / len(vdl)
                pbar.set_description(
                    f'{epoch} val loss:{epoch_loss_v:.3e}')

                preds = [(objs.cpu().numpy(), cls.cpu().numpy()) 
                         for objs, cls in preds]
                add_coco_preds(preds, infos, coco_results)

        writer.add_scalar('val_loss/total', epoch_loss_v, epoch)
        writer.add_scalar('val_loss/coord', epoch_coord_loss_v, epoch)
        writer.add_scalar('val_loss/obj', epoch_obj_loss_v, epoch)
        writer.add_scalar('val_loss/cls', epoch_cls_loss_v, epoch)

        if len(coco_results) > 0:
            cocoDt = cocoGt.loadRes(coco_results)
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = [ann['image_id'] for ann in coco_results]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            writer.add_scalar('coco/map', cocoEval.stats[0], epoch)
            writer.add_scalar('coco/map_50', cocoEval.stats[1], epoch)
            writer.add_scalar('coco/map_small', cocoEval.stats[3], epoch)
            writer.add_scalar('coco/map_medium', cocoEval.stats[4], epoch)
            writer.add_scalar('coco/map_large', cocoEval.stats[5], epoch)
        else:
            writer.add_scalar('coco/map', 0, epoch)
            writer.add_scalar('coco/map_50', 0, epoch)
            writer.add_scalar('coco/map_small', 0, epoch)
            writer.add_scalar('coco/map_medium', 0, epoch)
            writer.add_scalar('coco/map_large', 0, epoch)
        writer.flush()

        if epoch % save == 0 or epoch == epochs-1:
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
