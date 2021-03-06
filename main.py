from dataIO import MultiScaleFlipAug
import cv2
import argparse
import os

import torch
from util import Config
from Model.builder import build_engine
from Model.checkpoint import load_weights
from Model.parallel.data_parallel import MMDataParallel


def parse_args():
    parser = argparse.ArgumentParser(description='val on a image')
    parser.add_argument('--config', default='tinaface.py',
                        help='train config file path')
    parser.add_argument('--checkpoint', default='epoch_184_weights.pth',
                        help='checkpoint file')
    parser.add_argument('--img_path',
                        default='/dfsdata2/xuwj16_data/vedadet/data/WIDERFace/WIDER_val/50--Celebration_Or_Party/50_Celebration_Or_Party_houseparty_50_432.jpg',
                        help='to test on the image')
    parser.add_argument('--show_thresh', default=0.5,
                        help='the confidence to show a box')

    args = parser.parse_args()
    return args


def toC(im_path):
    results = {}
    raw_img = cv2.imread(im_path)
    results['filename'] = im_path
    results['ori_filename'] = im_path
    results['img'] = raw_img
    results['img_shape'] = raw_img.shape
    results['ori_shape'] = raw_img.shape
    results['img_fields'] = ['img']

    img_scale = (1100, 1650)
    flip = False
    transforms = [
        dict(typename='Resize', keep_ratio=True),
        dict(typename='RandomFlip', flip_ratio=0.0),
        dict(typename='Normalize', **dict(
            mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)),
        dict(typename='Pad', size_divisor=32, pad_val=0),
        dict(typename='ImageToTensor', keys=['img']),
        dict(typename='Collect', keys=['img'])
    ]

    toc = MultiScaleFlipAug(transforms, img_scale, flip=flip)
    container = toc(results)
    container['img'][0] = container['img'][0].unsqueeze(0)
    temp = container['img_metas'][0]
    temp._data = [[temp._data]]
    container['img_metas'][0] = temp
    return container, raw_img


if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.img_path), '%s do not exist' % args.img_path
    cfg = Config.fromfile(args.config)
    container, image = toC(args.img_path)
    print('building and loading engine ...')

    with torch.no_grad():
        engine = build_engine(cfg.val_engine)
        load_weights(engine.model, args.checkpoint, map_location='cpu')
        device = torch.cuda.current_device()
        engine = MMDataParallel(
            engine.to(device), device_ids=[torch.cuda.current_device()])
        engine.eval()
        print('eval on %s' % args.img_path)
        dets = engine(container)[0][0]
    for i in range(dets.shape[0]):
        x1, y1, x2, y2, score = dets[i]
        if score >= args.show_thresh:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255),
                          1)  # 画图参数分别是(图像，（x起始坐标,y起始坐标），（x终点坐标,y终点坐标），（R，G，B）框框的颜色，width框框的粗细)
    cv2.imwrite('show1.png', image)
    # cv2.imshow('test', image)  # 显示该图像
    # cv2.waitKey(0)  # 每张图片的显示时间，必须要有，不然就会出现图片现实太快，像没显示一样
    print('Done eval on %s' % args.img_path)
