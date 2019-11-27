#!/usr/bin/env python3
"""
Usage:
    ./dataset.py [--show] [--ipy]
"""

import torch
import numpy as np
import json
import pickle
import random
import itertools
from pathlib import Path
from collections import defaultdict, OrderedDict
from PIL import Image, ImageDraw, ImageFont

from config import COCO_PATH, INPUT_SZ, BOX_PRED_SZ, CLASS_PRED_SZ, CELL_SZS, \
    CATEGORY_INFO, CAT_ID_TO_IDX, CAT_IDX_TO_NAME

import cython_utils

def process_annotations(ann_json):
    id_to_anns = defaultdict(lambda: [])
    for ann in ann_json['annotations']:
        id_to_anns[ann['image_id']].append({ 
            'category': ann['category_id'],
            'bbox': ann['bbox'],
        })
    for img in ann_json['images']:
        yield {
            'fn': img['file_name'],
            'id': img['id'],
            'w': img['width'],
            'h': img['height'],
            'anns': id_to_anns[img['id']],
        }

def argmax2d(x):
    a = np.argmax(x)
    return (a//x.shape[1], a%x.shape[1])

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, img_size, img_transform=None, return_extra=False):
        name = 'train2017' if is_train else 'val2017'
        self.img_size = img_size
        self.img_transform = img_transform
        self.return_extra = return_extra
        self.imgs_path = COCO_PATH / name
        ann_path = COCO_PATH / 'annotations' / ('instances_'+name+'.json')
        ann_cache_path = ann_path.with_suffix('.pickle')
        if ann_cache_path.exists():
            print(f'using {name} ann cache')
            with ann_cache_path.open('rb') as f:
                self.anns = pickle.load(f)
        else:
            print(f'loading {name} ann json')
            with ann_path.open('rb') as f:
                ann_json = json.load(f)
            print(f'processing ann json')
            self.anns = list(process_annotations(ann_json))
            print(f'saving {name} ann cache')
            with ann_cache_path.open('wb') as f:
                pickle.dump(self.anns, f)

    def __len__(self):
        return len(self.anns)

    # @profile
    def __getitem__(self, idx):
        info = self.anns[idx]
        img = Image.open(self.imgs_path / info['fn']).convert('RGB')
        orig_w, orig_h = img.size
        w_ratio = self.img_size / orig_w
        h_ratio = self.img_size / orig_h
        if self.return_extra:
            orig_img = img.copy()
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.img_transform is not None:
            img = self.img_transform(img)
        # score 0 = negative
        #       1 = ignore (IOU > .5 but not best)
        #       2 = positive
        score_masks = [np.zeros((INPUT_SZ//s, INPUT_SZ//s), dtype=np.int32)
                       for s in CELL_SZS]
        boxes = [torch.zeros(BOX_PRED_SZ, INPUT_SZ//s, INPUT_SZ//s)
                 for s in CELL_SZS]
        classes = [torch.zeros(INPUT_SZ//s, INPUT_SZ//s, dtype=torch.long)
                   for s in CELL_SZS]
        for ann in info['anns']:
            cat_idx = CAT_ID_TO_IDX[ann['category']]
            x, y, w, h = ann['bbox']
            x *= w_ratio
            y *= h_ratio
            w *= w_ratio
            h *= h_ratio
            x0, x1 = x, x+w
            y0, y1 = y, y+h
            bbox = [x0, y0, x1, y1]
            ious = [np.zeros((INPUT_SZ//s, INPUT_SZ//s)) for s in CELL_SZS]
            for cell_sz, scale_iou, score_mask in zip(CELL_SZS, ious, score_masks):
                n_cells = INPUT_SZ // cell_sz
                min_cell_x = max(0, int(x0) // cell_sz)
                max_cell_x = min(n_cells, int(x1) // cell_sz + 1)
                min_cell_y = max(0, int(y0) // cell_sz)
                max_cell_y = min(n_cells, int(y1) // cell_sz + 1)
                cython_utils.calc_all_ious(
                    bbox, cell_sz,
                    min_cell_x, max_cell_x,
                    min_cell_y, max_cell_y,
                    scale_iou, score_mask)
            maxes = [scale_iou.max() for scale_iou in ious]
            best_scale = max(range(len(maxes)), key=lambda i: maxes[i])
            best_row, best_col = argmax2d(ious[best_scale])
            def set_box(i, val):
                boxes[best_scale][i, best_row, best_col] = val
            set_box(0, 1)
            set_box(1, x+w/2)
            set_box(2, y+h/2)
            set_box(3, w)
            set_box(4, h)
            classes[best_scale][best_row, best_col] = cat_idx
            score_masks[best_scale][best_row, best_col] = 2
        rets = [img, boxes, classes, score_masks]
        if self.return_extra:
            rets.append((info, orig_img))
        return tuple(rets)

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    ds = CocoDataset(is_train=True, img_size=INPUT_SZ)
    print('len:', len(ds))


    if args['--show']:
        print('showing')

        idx = random.randint(0, len(ds)-1)
        img, boxes, classes, score_masks = ds[idx]

        print('img:')
        print(img)

        for i, b in enumerate(boxes):
            print(f'boxes {i}: {b.shape}')
        for i, c in enumerate(classes):
            print(f'classes {i}: {c.shape}')
        for i, m in enumerate(score_masks):
            print(f'score_masks {i}: {m.shape}')

        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
        font = ImageFont.truetype(font_path, 12)

        scale_imgs = []
        for i, cell_sz in enumerate(CELL_SZS):
            print('ignores:', (score_masks[i] == 1).sum())
            scale_img = img.copy()
            draw = ImageDraw.Draw(scale_img)
            draw.text((0, 0), f'{cell_sz}x{cell_sz}', fill=(255,0,0), font=font)
            idxs = itertools.product(
                range(INPUT_SZ//cell_sz), range(INPUT_SZ//cell_sz))
            for row, col in idxs:
                x0 = col * cell_sz
                x1 = (col+1) * cell_sz - 1
                y0 = row * cell_sz
                y1 = (row+1) * cell_sz - 1
                score = score_masks[i][row, col]
                if score == 1:
                    draw.rectangle([x0, y0, x1, y1],
                        fill=None, outline=(255,255,0))
                elif score == 2:
                    draw.rectangle([x0, y0, x1, y1], 
                        fill=None, outline=(255,255,255))
                    cls_idx = classes[i][row, col]
                    cls_name = CAT_IDX_TO_NAME[int(cls_idx)]
                    x, y, w, h = boxes[i][1:, row, col]
                    draw.rectangle([int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)],
                        fill=None, outline=(0,255,0))
                    draw.text((int(x-w/2), int(y-h/2)), cls_name, 
                        fill=(0,255,0), font=font)
                    draw.line([int(x0+cell_sz/2), int(y0+cell_sz/2), x, y],
                        fill=(255,255,255))
            scale_imgs.append(scale_img)

        combined = Image.new('RGB', (INPUT_SZ, INPUT_SZ*len(scale_imgs)))
        for i, scale_img in enumerate(scale_imgs):
            combined.paste(scale_img, (0, i*INPUT_SZ))
        combined.save('target-vis.png')

        """
        for cat_idx, (x, y, w, h) in targets:
            x0 = int(x)
            y0 = int(y)
            x1 = int(x+w)
            y1 = int(y+h)
            draw.rectangle([x0, y0, x1, y1], fill=None, outline=(255,0,0))
            cat_name = CAT_IDX_TO_NAME[cat_idx]
            print(cat_name)
            draw.text((x0, y0-12), cat_name, fill=(255,0,0), font=font)
        img.save('dataset_example.png')
        """

    if args['--ipy']:
        from IPython import embed; embed()