import torch

from config import CELL_SZS, CELL_CENTERS

CELL_CENTERS_T = [
    (torch.from_numpy(ys).float().cuda(), torch.from_numpy(xs).float().cuda()) 
    for ys, xs in CELL_CENTERS]

def transform_boxes(preds, scale_i):
    cell_sz = CELL_SZS[scale_i]
    cell_y_centers, cell_x_centers = CELL_CENTERS_T[scale_i]

    boxes = preds[0]

    pred_xs = torch.sigmoid(boxes[:,1])*cell_sz + cell_x_centers
    pred_ys = torch.sigmoid(boxes[:,2])*cell_sz + cell_y_centers
    pred_ws = cell_sz * torch.exp(boxes[:,3])
    pred_hs = cell_sz * torch.exp(boxes[:,4])

    return pred_xs, pred_ys, pred_ws, pred_hs
