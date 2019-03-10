from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import json
import skimage.io
import matplotlib.pyplot as plt

sys.path.append('/usr/local/lib/python2.7/dist-packages/')
sys.path.append(os.path.join('..', 'lib'))

from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

import torch
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

def init_net(cnn):
    # 80 categories for COCO, #0 is background
    if cnn == 'vgg16':
        net = vgg16()
        net.create_architecture(81, tag='default', anchor_scales=[4, 8, 16, 32])
        model = torch.load('vgg16_faster_rcnn_iter_1190000.pth')
    elif cnn == 'resnet101':
        net = resnetv1(101)
        net.create_architecture(81, tag='default', anchor_scales=[4, 8, 16, 32])
        model = torch.load('res101_faster_rcnn_iter_1190000.pth')
    else:
        raise Exception('unsupported network: %s' % cnn)

    # print(net)
    # keys = []
    # for k in model:
    #     keys.append(k)
    # for k in sorted(keys):
    #     print('%s: %s' % (k, model[k].shape))

    net.load_state_dict(model)
    net.to('cuda')
    net.eval()
    return net

def bbox(im, scores, boxes, cat_desc, thres, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im[:, :, [2, 1, 0]], aspect='equal')
    for idx in range(0, scores.shape[0]):
        confident = scores[idx].max()
        if confident > thres:
            cat_id = scores[idx].argmax()
            if 0 != cat_id:
                cat_name = '/'.join(cat_desc[cat_id])
                x1, y1, x2, y2 = boxes[idx][cat_id]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor='red', linewidth=3))
                ax.text(x1 + 2, y1 + 15, '%s %.3f' % (cat_name, confident),
                    fontsize=14, color='white')
    plt.tight_layout()
    plt.title(title)
    plt.show()

def detect():
    imgs = ['000456', '000542', '001150', '001763', '004545']
    imgs = map(lambda no: os.path.join('../data/demo', no) + '.jpg', imgs)
    n_class = 81
    with open('coco2014_objects.json', 'r') as fp:
        coco_cat = json.load(fp)

    for arch in ['vgg16', 'resnet101']:
        net = init_net(arch)

        for img_file in imgs:
            im = skimage.io.imread(img_file)[:, :, [2, 1, 0]]
            scores, boxes = im_detect(net, im)
            boxes = boxes.reshape(-1, n_class, 4)
            thres = 0.9
            bbox(im, scores, boxes, coco_cat, thres,
                '%s [threshold=%.2f]' % (arch, thres))

if __name__ == '__main__':
    detect()
