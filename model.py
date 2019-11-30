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

import geffnet
from geffnet.efficientnet_builder import decode_arch_def, EfficientNetBuilder, \
    InvertedResidual, DepthwiseSeparableConv
from geffnet.activations import get_act_layer

from config import N_CLASSES, INPUT_SZ, BOX_PRED_SZ, CLASS_PRED_SZ, CELL_SZS

DEBUG_SIZES = os.environ.get('DBG_SZS') is not None
def dbgsz(s, x):
    if DEBUG_SIZES:
        print(f'{s}: {x.shape}')
    return x

def conv_block(in_ch, out_ch, k, act=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, k, padding=k//2),
    ]
    if act:
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class BiFPN(nn.Module):
    def __init__(self, n_inputs, n_chs):
        super().__init__()
        self.n_inputs = n_inputs

        self.intermediate_fusion_weights = nn.Parameter(
            torch.ones(self.n_inputs-2, 2, dtype=torch.float32))
        self.output_fusion_weights = nn.Parameter(
            torch.ones(self.n_inputs, 3, dtype=torch.float32))

        self.intermediate_convs = nn.ModuleList([])
        for _ in range(self.n_inputs-2):
            self.intermediate_convs.append(conv_block(
                in_ch=n_chs, out_ch=n_chs, k=3))
        self.output_convs = nn.ModuleList([])
        for _ in range(self.n_inputs):
            self.output_convs.append(conv_block(
                in_ch=n_chs, out_ch=n_chs, k=3))

    @staticmethod
    def fast_fuse(xs, ws):
        "Output has shape of first x"
        target_sz = xs[0].shape[-1]
        output = torch.zeros_like(xs[0])
        for i, x in enumerate(xs):
            if x.shape[-1] != target_sz:
                if x.shape[-1] == target_sz//2:
                    x = F.interpolate(x, scale_factor=2, mode='nearest')
                elif x.shape[-1] == target_sz*2:
                    x = F.avg_pool2d(x, 2)
                else:
                    assert False
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

class ExtendedEN(nn.Module):
    def __init__(self):
        super().__init__()

        en = geffnet.efficientnet_b0(
            pretrained=True, as_sequential=True,
            drop_rate=0, drop_connect_rate=0)
        self.en_blocks = nn.ModuleList(iter(en[:9]))

        arch = [
            ['ir_r4_k5_s2_e6_c224_se0.25'],
            ['ir_r4_k5_s2_e6_c256_se0.25'],
        ]
        block_args = decode_arch_def(arch)

        en_builder = EfficientNetBuilder(
            act_layer=get_act_layer('swish'),
            drop_connect_rate=0)
        en_builder.in_chs = 192
        en_builder.block_count = sum(len(x) for x in block_args)
        en_builder.block_idx = 9
        for ba in block_args:
            self.en_blocks.append(en_builder._make_stack(ba))

        self.output_idxs = [5, 7, 8, 9, 10]
        self.output_blocks = [self.en_blocks[i] for i in self.output_idxs]

    def forward(self, x):
        outputs = []
        for i, b in enumerate(self.en_blocks):
            x = b(x)
            if i in self.output_idxs:
                outputs.append(x)
        return outputs

class EfficientDet(nn.Module):
    def __init__(self):
        super().__init__()

        # params (set nicely later)
        self.bifpn_n_layers = 2
        self.bifpn_n_chs = 64
        self.heads_n_layers = 3

        self.en = ExtendedEN()

        self.bifpn_input_convs = nn.ModuleList([])
        for b in self.en.output_blocks:
            in_ch = b[-1].bn3.num_features
            self.bifpn_input_convs.append(conv_block(
                in_ch=in_ch, out_ch=self.bifpn_n_chs, k=3))

        self.bifpns = nn.ModuleList([])
        for _ in range(self.bifpn_n_layers):
            self.bifpns.append(BiFPN(
                n_inputs=len(self.en.output_blocks),
                n_chs=self.bifpn_n_chs))

        box_layers = []
        class_layers = []
        for _ in range(self.heads_n_layers):
            box_layers.append(
                conv_block(in_ch=self.bifpn_n_chs, out_ch=self.bifpn_n_chs, k=3))
            class_layers.append(
                conv_block(in_ch=self.bifpn_n_chs, out_ch=self.bifpn_n_chs, k=3))
        # output layers
        box_layers.append(conv_block(
            in_ch=self.bifpn_n_chs, out_ch=BOX_PRED_SZ, k=3, act=False))
        class_layers.append(conv_block(
            in_ch=self.bifpn_n_chs, out_ch=CLASS_PRED_SZ, k=3, act=False))
        self.box_layers = nn.Sequential(*box_layers)
        self.class_layers = nn.Sequential(*class_layers)

    def forward(self, x):
        dbgsz('input', x)
        bifpn_xs = []
        for en_output, bic in zip(self.en(x), self.bifpn_input_convs):
            bifpn_xs.append(bic(en_output))
        for i, bifpn in enumerate(self.bifpns):
            bifpn_xs = bifpn(*bifpn_xs)
            for j, bifpn_x in enumerate(bifpn_xs):
                dbgsz(f'bifpn {i}-{j}', bifpn_x)
        outputs = [(self.box_layers(x), self.class_layers(x)) for x in bifpn_xs]
        for i, (box_pred, class_pred) in enumerate(outputs):
            dbgsz(f'box_pred {i}', box_pred)
            dbgsz(f'class_pred {i}', class_pred)
            # clamp w, h to prevent exp overflow
            if not self.train:
                box_pred[:,3:5] = torch.clamp(box_pred[:,3:5], max=3)
        return outputs

class NoBifpnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = ExtendedEN()
        self.boxnets, self.clsnets = nn.ModuleList([]), nn.ModuleList([])
        n_ch = 256
        for b in self.en.output_blocks:
            in_ch = b[-1].bn3.num_features
            self.boxnets.append(nn.Sequential(
                conv_block(in_ch=in_ch, out_ch=n_ch, k=3),
                conv_block(in_ch=n_ch, out_ch=n_ch, k=3),
                conv_block(in_ch=n_ch, out_ch=BOX_PRED_SZ, k=3, act=False)))
            self.clsnets.append(nn.Sequential(
                conv_block(in_ch=in_ch, out_ch=n_ch, k=3),
                conv_block(in_ch=n_ch, out_ch=n_ch, k=3),
                conv_block(in_ch=n_ch, out_ch=CLASS_PRED_SZ, k=3, act=False)))
    def forward(self, x):
        outputs = []
        for en_out, boxnet, clsnet in zip(self.en(x), self.boxnets, self.clsnets):
            outputs.append((boxnet(en_out), clsnet(en_out)))
        return outputs

class StupidNet(nn.Module):
    def __init__(self):
        super().__init__()
        chs = 64
        self.downsamples = nn.ModuleList([nn.Sequential(
                conv_block(3, chs, k=3),
                nn.MaxPool2d(2),
                conv_block(chs, chs, k=3),
                nn.MaxPool2d(2),
                conv_block(chs, chs, k=3),
                nn.MaxPool2d(2),
        )])
        for _ in range(4):
            self.downsamples.append(nn.Sequential(
                conv_block(chs, chs, k=3),
                nn.MaxPool2d(2),
            ))
        self.boxnets, self.clsnets = nn.ModuleList([]), nn.ModuleList([])
        for i in range(5):
            self.boxnets.append(nn.Sequential(
                conv_block(chs, chs, k=3),
                conv_block(chs, chs, k=3),
                conv_block(chs, BOX_PRED_SZ, k=3, act=False)))
            self.clsnets.append(nn.Sequential(
                conv_block(chs, chs, k=3),
                conv_block(chs, chs, k=3),
                conv_block(chs, CLASS_PRED_SZ, k=3, act=False)))

    def forward(self, x):
        outputs = []
        for ds, bn, cn in zip(self.downsamples, self.boxnets, self.clsnets):
            x = ds(x)
            outputs.append((bn(x), cn(x)))
        return outputs

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    model = EfficientDet().cuda()

    if args['--print']:
        print(model)
        ps = list(model.parameters())
        print('np:', sum(p.numel() for p in ps if p.requires_grad))
        import thop
        x = torch.randn(1, 3, INPUT_SZ, INPUT_SZ, device='cuda')
        flops, params = thop.profile(model, inputs=(x,))
        print(flops, params)

        n = 0
        for b in model.bifpns:
            n += sum(p.numel() for p in b.parameters())
        n += sum(p.numel() for p in model.box_layers.parameters())
        n += sum(p.numel() for p in model.class_layers.parameters())
        print('n:', n)

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
        y = model(x)[0]
        from torchviz import make_dot
        G = make_dot(y[0].sum(), params=dict(model.named_parameters()))
        G.format = 'pdf'
        G.render('objs_graph')
        G = make_dot(y[1].sum(), params=dict(model.named_parameters()))
        G.format = 'pdf'
        G.render('cls_graph')

    if args['--time']:
        import time
        x = torch.randn(1, 3, INPUT_SZ, INPUT_SZ, device='cuda')
        for _ in range(int(args['--time'])):
            t0 = time.time()
            y = model(x)
            t1 = time.time()
            print(f'inference in {(t1-t0)*1000} ms')
