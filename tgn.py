#!/usr/bin/env python3

import torch
import torch.nn as nn

import geffnet
from geffnet.efficientnet_builder import decode_arch_def, EfficientNetBuilder, \
    InvertedResidual, DepthwiseSeparableConv
from geffnet.activations import get_act_layer

import itertools

def pm(m):
    res = 512
    print(f'  ({res})')
    for i, l in enumerate(m):
        if isinstance(l, nn.Conv2d):
            res //= l.stride[0]
        if isinstance(l, nn.Sequential):
            for j, ll in enumerate(l):
                print(f'{i} ' if j == 0 else '  ', end='')
                if isinstance(ll, DepthwiseSeparableConv):
                    inch = ll.conv_dw.in_channels
                    outch = ll.bn2.num_features
                    k = ll.conv_dw.kernel_size
                    s = ll.conv_dw.stride
                    print(f'DepthwiseSeparableConv({inch}, {outch}, k={k}, s={s})')
                    res //= s[0]
                elif isinstance(ll, InvertedResidual):
                    inch = ll.conv_pw.in_channels
                    outch = ll.bn3.num_features
                    s = ll.conv_dw.stride
                    k = ll.conv_dw.kernel_size
                    print(f'InvertedResidual({inch}, {outch}, k={k}, s={s})')
                    res //= s[0]
                else:
                    print(ll)
                    assert False
        else:
            print(i, l)
        print(f'  ({res})')

if __name__ == '__main__':
    dr = .25
    dcr = .2

    en = geffnet.efficientnet_b0(
        pretrained=True, drop_rate=dr, drop_connect_rate=dcr,
        as_sequential=True)

    ls = en[:9]

    arch = [
        ['ir_r4_k5_s2_e6_c224_se0.25'],
        ['ir_r4_k5_s2_e6_c256_se0.25'],
    ]
    block_args = decode_arch_def(arch)

    en_builder = EfficientNetBuilder(
        act_layer=get_act_layer('swish'),
        drop_connect_rate=dcr)
    en_builder.in_chs = 192
    en_builder.block_count = sum(len(x) for x in block_args)
    en_builder.block_idx = 9
    blocks = [en_builder._make_stack(ba) for ba in block_args]

    full_model = nn.Sequential(*list(itertools.chain(ls, blocks)))
    # full_model = full_model.cuda()
    # full_model.train()

    x = torch.randn(1, 3, 512, 512)
    print(x.shape)
    for i, l in enumerate(full_model):
        print(i, 'applying', type(l))
        x = l(x)
        print(x.shape)

    pm(full_model)

    # from IPython import embed; embed()
