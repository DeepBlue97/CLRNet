import os
import time

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner
from clrnet.datasets import build_dataloader

from grabscreen import grab_screen


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    runner = Runner(cfg)

    if args.validate:
        runner.validate()
    elif args.test:
        runner.test()
    elif args.infer_one:
        runner.infer_one(img=args.img)
    elif args.infer:
        while True:
            img = grab_screen(region=(0, 0+200, 1600, 576+200))
            img = cv2.resize(img, (1640, 590))
            runner.infer_one(img=img)

            # time.sleep(1)
            # cv2.imshow('view', cv2.resize(img, (200, 76)))
            # cv2.waitKey(1000)
            # runner.infer_one(img=args.img)
    else:
        runner.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--img',
                        type=str,
                        default=None,
                        help='img path')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument(
        '--infer',
        action='store_true',
        help='infer one img')
    parser.add_argument(
        '--infer_one',
        action='store_true',
        help='infer one img')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
