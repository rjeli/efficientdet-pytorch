#!/usr/bin/env python3
"""
Usage:
    ./model.py [--print] [--print-sizes] [--export-onnx] [--make-dot] [--time=<n>]
"""

import os
import itertools

import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch.model import EfficientNet, MBConvBlock
from efficientnet_pytorch.utils import BlockArgs, BlockDecoder, \
    round_filters, round_repeats

from config import N_CLASSES, INPUT_SZ, BOX_PRED_SZ, CLASS_PRED_SZ

DEBUG_SIZES = os.environ.get('DBG_SZS') is not None
def dbgsz(s, x):
    if DEBUG_SIZES:
        print(f'{s}: {x.shape}')

EXTRA_BLOCKS_ARGS = [
    'r4_k5_s22_e6_i192_o224_se0.25',
    'r4_k5_s22_e6_i224_o256_se0.25',
]
BIFPN_INPUT_LAYER_IDXS = set([4, 10, 14, 18, 22])
BIFPN_N_LAYERS = 2
BIFPN_N_CHS = 64
BOX_CLASS_N_LAYERS = 3

def conv2d_relu_bn(in_ch, out_ch, k):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k//2),
        nn.ReLU(),
        nn.BatchNorm2d(out_ch),
    )

def conv2d_bn(in_ch, out_ch, k):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k//2),
        nn.BatchNorm2d(out_ch),
    )

class BiFPN(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs

        self.intermediate_fusion_weights = nn.Parameter(
            torch.ones(self.n_inputs-2, 2, dtype=torch.float32))
        self.output_fusion_weights = nn.Parameter(
            torch.ones(self.n_inputs, 3, dtype=torch.float32))

        self.intermediate_convs = nn.ModuleList([])
        for _ in range(self.n_inputs-2):
            self.intermediate_convs.append(conv2d_relu_bn(
                in_ch=BIFPN_N_CHS, out_ch=BIFPN_N_CHS, k=3))
        self.output_convs = nn.ModuleList([])
        for _ in range(self.n_inputs):
            self.output_convs.append(conv2d_relu_bn(
                in_ch=BIFPN_N_CHS, out_ch=BIFPN_N_CHS, k=3))

    @staticmethod
    def fast_fuse(xs, ws):
        "Output has shape of first x"
        target_sz = xs[0].shape[-1]
        output = torch.zeros_like(xs[0])
        for i, x in enumerate(xs):
            if x.shape[-1] != target_sz:
                assert x.shape[-1] in [target_sz//2, target_sz*2]
                x = F.interpolate(x, size=target_sz, mode='nearest')
            output += ws[i] * x
        output /= ws.sum() + 1e-4
        return output

    def forward(self, *xs):
        assert len(xs) == self.n_inputs
        ifws = F.relu(self.intermediate_fusion_weights)
        intermediates = []
        for i in range(self.n_inputs-2):
            small = xs[-1] if i == 0 else intermediates[-1]
            same = xs[self.n_inputs-i-2]
            fused = self.fast_fuse([same, small], ifws[i])
            intermediates.append(self.intermediate_convs[i](fused))
        ofws = F.relu(self.output_fusion_weights)
        outputs = []
        for i in range(self.n_inputs):
            if i == 0:
                fuse_inps = [xs[0], intermediates[-1]]
            elif i > 0 and i < self.n_inputs - 1:
                fuse_inps = [xs[i], intermediates[-i], outputs[-1]]
            elif i == self.n_inputs - 1:
                fuse_inps = [xs[i], outputs[-1]]
            fused = self.fast_fuse(fuse_inps, ofws[i])
            outputs.append(self.output_convs[i](fused))
        return outputs

class EfficientDet(nn.Module):
    def __init__(self):
        super().__init__()
        en = EfficientNet.from_pretrained('efficientnet-b0')
        self.dcr = en._global_params.drop_connect_rate
        self.stem = nn.Sequential(en._conv_stem, en._bn0, en._swish)
        self.en_blocks = nn.ModuleList(en._blocks[:-1])
        for bs in EXTRA_BLOCKS_ARGS:
            b_args = BlockDecoder._decode_block_string(bs)
            b_args = b_args._replace(
                input_filters=round_filters(b_args.input_filters, 
                    en._global_params),
                output_filters=round_filters(b_args.output_filters,
                    en._global_params),
                num_repeat=round_repeats(b_args.num_repeat,
                    en._global_params),
            )
            self.en_blocks.append(
                MBConvBlock(b_args, en._global_params))
            if b_args.num_repeat > 1:
                b_args = b_args._replace(
                    input_filters=b_args.output_filters, stride=1)
            for _ in range(b_args.num_repeat-1):
                self.en_blocks.append(
                    MBConvBlock(b_args, en._global_params))
        self.bifpn_input_convs = nn.ModuleList([])
        for i in BIFPN_INPUT_LAYER_IDXS:
            block = self.en_blocks[i]
            self.bifpn_input_convs.append(conv2d_relu_bn(
                in_ch=block._block_args.output_filters, out_ch=BIFPN_N_CHS, k=3))
        self.bifpns = nn.ModuleList([])
        for _ in range(BIFPN_N_LAYERS):
            self.bifpns.append(BiFPN(n_inputs=len(BIFPN_INPUT_LAYER_IDXS)))
        box_layers = nn.ModuleList([])
        class_layers = nn.ModuleList([])
        for _ in range(BOX_CLASS_N_LAYERS):
            box_layers.append(
                conv2d_relu_bn(in_ch=BIFPN_N_CHS, out_ch=BIFPN_N_CHS, k=3))
            class_layers.append(
                conv2d_relu_bn(in_ch=BIFPN_N_CHS, out_ch=BIFPN_N_CHS, k=3))
        # output layers
        box_layers.append(nn.Conv2d(BIFPN_N_CHS, BOX_PRED_SZ, 3, padding=1))
        class_layers.append(nn.Conv2d(BIFPN_N_CHS, CLASS_PRED_SZ, 3, padding=1))
        self.box_layers = nn.Sequential(*box_layers)
        self.class_layers = nn.Sequential(*class_layers)

    def forward(self, x):
        x = self.stem(x)
        dbgsz('stem', x)
        bifpn_xs = []
        for i, b in enumerate(self.en_blocks):
            dcr = self.dcr * float(i) / len(self.en_blocks)
            x = b(x, drop_connect_rate=dcr)
            dbgsz(f'block {i}', x)
            if i in BIFPN_INPUT_LAYER_IDXS:
                bifpn_xs.append(x)
        bifpn_xs = [self.bifpn_input_convs[i](inp) 
                    for i, inp in enumerate(bifpn_xs)]
        for i, bifpn_x in enumerate(bifpn_xs):
            dbgsz(f'bifpn_input {i}', bifpn_x)
        for i, bifpn in enumerate(self.bifpns):
            bifpn_xs = bifpn(*bifpn_xs)
            for j, bifpn_x in enumerate(bifpn_xs):
                dbgsz(f'bifpn {i}-{j}', bifpn_x)
        outputs = [(self.box_layers(x), self.class_layers(x)) for x in bifpn_xs]
        for i, (box_pred, class_pred) in enumerate(outputs):
            dbgsz(f'box_pred {i}', box_pred)
            dbgsz(f'class_pred {i}', class_pred)
        return outputs

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    model = EfficientDet().cuda()

    if args['--print']:
        print(model)

    if args['--print-sizes']:
        x = torch.randn(1, 3, INPUT_SZ, INPUT_SZ, device='cuda')
        print(f'input: {x.shape}')
        en = model.en
        x = en._swish(en._bn0(en._conv_stem(x)))
        print(f'stem: {x.shape}')
        def ratio(x):
            return f'(in/{int(INPUT_SZ/x.shape[-1])})'
        for i, b in enumerate(model.en._blocks):
            x = b(x)
            print(f'block {i}: {x.shape} {ratio(x)}')
        x = en._swish(en._bn1(en._conv_head(x)))
        print(f'head: {x.shape} {ratio(x)}')

    if args['--export-onnx']:
        x = torch.randn(1, 3, INPUT_SZ, INPUT_SZ, device='cuda')
        torch.onnx.export(model, x, 'model.onnx', verbose=True)

    if args['--make-dot']:
        x = torch.randn(1, 3, INPUT_SZ, INPUT_SZ, device='cuda')
        y = model(x)
        from torchviz import make_dot
        G = make_dot(y[0].sum(), params=dict(model.named_parameters()))
        G.format = 'pdf'
        G.render('model-graph.pdf')

    if args['--time']:
        import time
        x = torch.randn(1, 3, INPUT_SZ, INPUT_SZ, device='cuda')
        for _ in range(int(args['--time'])):
            t0 = time.time()
            y = model(x)
            t1 = time.time()
            print(f'inference in {(t1-t0)*1000} ms')
