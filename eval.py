#!/usr/bin/env python3
"""
Usage:
    ./eval.py <model> [--coco] [--show=<n>] [--thres=<p>]
"""

from docopt import docopt
from tqdm import tqdm

import torch
import numpy as np

from model import EfficientDet
from dataset import CocoDataset
from train import img_transform
from config import INPUT_SZ, CELL_SZS, CAT_IDX_TO_NAME, CELL_CENTERS
from cython_utils import add_coco_preds

from pathlib import Path
import sys
sys.path.append(str(Path.home() / 'coco' / 'cocoapi' / 'PythonAPI'))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# @profile
def main(args):
    model = EfficientDet().cuda()
    model.load_state_dict(torch.load(args['<model>']))
    model.eval()

    batch_sz = 16

    ds = CocoDataset(is_train=True, img_size=INPUT_SZ, 
        img_transform=img_transform, 
        return_extra=True)

    # split_sz = int(0.01*len(ds))
    # ds, _ = torch.utils.data.random_split(ds, [split_sz, len(ds)-split_sz])

    if args['--coco']:
        dl = torch.utils.data.DataLoader(ds, 
            batch_size=batch_sz, shuffle=False, num_workers=4)

        results = []

        for imgs, boxes, classes, score_masks, infos in tqdm(dl):
            imgs = imgs.cuda()
            with torch.no_grad():
                preds = model(imgs)
                preds = [(objs.cpu().numpy(), cls.cpu().numpy()) 
                         for objs, cls in preds]

            add_coco_preds(preds, infos, results)

        cocoGt = COCO(str(Path.home()/'coco'/'annotations'/'instances_val2017.json'))
        cocoDt = cocoGt.loadRes(results)

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = [ann['image_id'] for ann in results]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    print(args['--show'])
    if args['--show']:
        idx = int(args['--show'])
        thres = float(args['--thres'])
        print('showing', idx, 'at thres', thres)
        img, boxes, classes, score_masks, (info, orig_img) = ds[idx]
        orig_w, orig_h = orig_img.size
        w_ratio = orig_w / INPUT_SZ
        h_ratio = orig_h / INPUT_SZ

        img = img.cuda()
        with torch.no_grad():
            import time
            torch.cuda.synchronize()
            t0 = time.time()
            preds = model(img[None])
            torch.cuda.synchronize()
            t1 = time.time()
            print(f'inference in {(t1-t0)*1000} ms')
            preds = [(objs.cpu().numpy()[0], cls.cpu().numpy()[0]) 
                     for objs, cls in preds]

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(orig_img)

        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
        font = ImageFont.truetype(font_path, 12)

        def sigmoid(x):
            return 1 / (1+np.exp(-x))

        for scale_i, cell_sz in enumerate(CELL_SZS):
            # gt
            sbox = boxes[scale_i]
            scls = classes[scale_i]

            cell_y_centers, cell_x_centers = CELL_CENTERS[scale_i]

            pbox, pcls = preds[scale_i]
            pred_xs = sigmoid(pbox[1]) * cell_sz + cell_x_centers
            pred_ys = sigmoid(pbox[2]) * cell_sz + cell_y_centers
            pred_ws = cell_sz * np.exp(pbox[3])
            pred_hs = cell_sz * np.exp(pbox[4])
            pred_objs = sigmoid(pbox[0])
            pred_cls = np.argmax(pcls, axis=1)

            for row in range(INPUT_SZ//cell_sz):
                for col in range(INPUT_SZ//cell_sz):

                    # draw gt
                    if sbox[0,row,col] == 1.0:
                        cx, cy, w, h = sbox[1:,row,col]
                        cx *= w_ratio
                        cy *= h_ratio
                        w *= w_ratio
                        h *= h_ratio
                        x0 = int(cx-w/2)
                        y0 = int(cy-h/2) 
                        x1 = int(cx+w/2)
                        y1 = int(cy+h/2)
                        draw.rectangle([x0,y0,x1,y1], fill=None, outline=(0,255,0))
                        cls_id = int(scls[row,col])
                        cls_name = CAT_IDX_TO_NAME[cls_id]
                        draw.text((x0,y0-12), cls_name, fill=(0,255,0), font=font)

                    # draw pred
                    if pred_objs[row,col] > thres:
                        cx = pred_xs[row,col] * w_ratio
                        cy = pred_ys[row,col] * h_ratio
                        w = pred_ws[row,col] * w_ratio
                        h = pred_hs[row,col] * h_ratio
                        x0 = int(cx-w/2)
                        y0 = int(cy-h/2) 
                        x1 = int(cx+w/2)
                        y1 = int(cy+h/2)
                        draw.rectangle([x0,y0,x1,y1], fill=None, outline=(255,255,0))
                        cls_id = pred_cls[row,col]
                        cls_name = CAT_IDX_TO_NAME[cls_id]
                        draw.text((x0,y0-12), cls_name, fill=(255,255,0), font=font)

        orig_img.save('evalshow.png')

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
