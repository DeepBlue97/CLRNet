import os

import torch

from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from clrnet.models.registry import build_net
from clrnet.utils.config import Config

from main import parse_args

class Runner(object):
    def __init__(self, cfg):
        # torch.manual_seed(cfg.seed)
        # np.random.seed(cfg.seed)
        # random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net.eval()
    
    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def infer(self, x):

        # x = self.to_cuda(x)
        
        pred = self.net(x)

        print(pred)

        return pred


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
runner = Runner(cfg)

import cv2

img = cv2.imread('/media/peter/ocean/data/dataset/lane/CULane/driver_23_30frame/05151640_0419.MP4/00030.jpg')

img = cv2.resize(img, (800, 320))
# cv2.imshow('img', img)
cv2.imwrite('test.jpg', img)

img = torch.Tensor(img)
img = img.permute([2, 0, 1]).unsqueeze(0)

# img = img.cuda()

pred = runner.infer(img)
pred_gpu = pred.detach().clone().cuda()
pred_lane = runner.net.heads.get_lanes(pred_gpu)
# from aio_lane_nms import Lane_nms

# pred_lane = Lane_nms(pred)

# pred_lane = runner.net.hea

# pred_line = runner.net.predictions_to_pred(pred)  # runner.net.heads.get_lanes(pred)
# runner.net.heads.get_lanes(pred.cuda()) # runner.net.heads.get_lanes(pred_gpu) 
print()

"""
python main_infer.py \
    work_dirs/clr/r18_culane/20230113_113829_lr_6e-04_b_24/config.py \
        --load_from work_dirs/clr/r18_culane/20230113_113829_lr_6e-04_b_24/ckpt/10.pth \
            --gpus 0


"""