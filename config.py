import json
import pickle
from pathlib import Path
import numpy as np

# so everyone can import cython_utils
import pyximport
pyximport.install(
    setup_args={
        'include_dirs': np.get_include(),
        # 'extra_compile_args': ['-Wno-#warnings'],
    },
    language_level=3)

COCO_PATH = Path.home() / 'coco'
INPUT_SZ = 512

def get_category_info():
    cats_path = COCO_PATH / 'categories.pickle'
    if cats_path.exists():
        with cats_path.open('rb') as f:
            return pickle.load(f)
    else:
        print('loading categories')
        ann_json_path = COCO_PATH / 'annotations' / 'instances_val2017.json'
        with ann_json_path.open('rb') as f:
            ann_json = json.load(f)
        cats = {c['id']: c for c in ann_json['categories']}
        with cats_path.open('wb') as f:
            pickle.dump(cats, f)
        return cats

CATEGORY_INFO = get_category_info()
CAT_ID_TO_IDX = {cid: i for i, cid in enumerate(CATEGORY_INFO.keys())}
CAT_IDX_TO_ID = {i: cid for i, cid in enumerate(CATEGORY_INFO.keys())}
CAT_IDX_TO_NAME = {i: c['name'] for i, c in enumerate(CATEGORY_INFO.values())}

N_CLASSES = len(CATEGORY_INFO)

# (objectness) + (tx, ty, tw, th)
# box x = sig(tx)+cx, y = sig(ty)+cy, w = cw*e^tw, h = ch*e^th
BOX_PRED_SZ = 1 + 4
CLASS_PRED_SZ = N_CLASSES

CELL_SZS = (8, 16, 32, 64, 128)

CELL_STARTS = [np.linspace(0, INPUT_SZ-s, num=INPUT_SZ//s) for s in CELL_SZS]
CELL_CENTERS = [np.meshgrid(cs, cs) for cs in CELL_STARTS]
