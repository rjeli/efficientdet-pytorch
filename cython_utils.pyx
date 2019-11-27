import numpy as np
cimport numpy as np

from config import CELL_SZS, INPUT_SZ, CELL_CENTERS

def calc_iou(b0, b1):
    # intersection coords
    ix0 = max(b0[0], b1[0])
    iy0 = max(b0[1], b1[1])
    ix1 = min(b0[2], b1[2])
    iy1 = min(b0[3], b1[3])
    # area of intersection
    ia = max(0, ix1-ix0) * max(0, iy1-iy0)
    # bbox areas
    b0a = (b0[2]-b0[0]) * (b0[3]-b0[1])
    b1a = (b1[2]-b1[0]) * (b1[3]-b1[1])
    return ia / (b0a + b1a - ia)

def calc_all_ious(bbox, cell_sz,
                  min_cell_x, max_cell_x,
                  min_cell_y, max_cell_y,
                  scale_iou, score_mask):
    for row in range(min_cell_y, max_cell_y):
        for col in range(min_cell_x, max_cell_x):
            cx0 = col * cell_sz
            cx1 = (col+1) * cell_sz - 1
            cy0 = row * cell_sz
            cy1 = (row+1) * cell_sz - 1
            iou = calc_iou(bbox, [cx0, cy0, cx1, cy1])
            scale_iou[row, col] = iou
    above_thres = (scale_iou > 0.5)
    score_mask[scale_iou > 0.5] = 1

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def add_coco_preds(preds, infos, results):
    cdef int scale_i, batch_i, cell_sz
    cdef np.ndarray objs, cls

    for scale_i, (cell_sz, (objs, cls)) in enumerate(zip(CELL_SZS, preds)):
        cell_y_centers, cell_x_centers = CELL_CENTERS[scale_i]
        pred_xs = sigmoid(objs[:,1]) * cell_sz + cell_x_centers
        pred_ys = sigmoid(objs[:,2]) * cell_sz + cell_y_centers
        pred_ws = cell_sz * np.exp(objs[:,3])
        pred_hs = cell_sz * np.exp(objs[:,4])
        pred_objs = sigmoid(objs[:,0])
        pred_cls = np.argmax(cls, axis=1)

        for batch_i in range(len(infos['id'])):
            image_id = int(infos['id'][batch_i])
            w_ratio = float(infos['w'][batch_i]) / INPUT_SZ
            h_ratio = float(infos['h'][batch_i]) / INPUT_SZ

            pred_xs2 = pred_xs * w_ratio
            pred_ys2 = pred_ys * h_ratio
            pred_ws2 = pred_ws * w_ratio
            pred_hs2 = pred_hs * h_ratio

            pred_xs2 += pred_ws2 / 2
            pred_ys2 += pred_hs2 / 2

            for row in range(INPUT_SZ//cell_sz):
                for col in range(INPUT_SZ//cell_sz):
                    score = pred_objs[batch_i, row, col]
                    cls_id = pred_cls[batch_i, row, col]
                    x = pred_xs2[batch_i, row, col]
                    y = pred_ys2[batch_i, row, col]
                    w = pred_ws2[batch_i, row, col]
                    h = pred_hs2[batch_i, row, col]
                    results.append({
                        'image_id': image_id,
                        'category_id': cls_id,
                        'bbox': [x, y, w, h],
                        'score': score,
                    })
