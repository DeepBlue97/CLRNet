import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os
import os.path as osp

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from clrnet.utils.visualization import imshow_lanes
from mmcv.parallel import MMDataParallel


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def infer_one(self, img):
        # if not self.test_loader:
        #     self.test_loader = build_dataloader(self.cfg.dataset.test,
        #                                         self.cfg,
        #                                         is_train=False)
        # for f in os.listdir('/media/peter/ocean/data/dataset/lane/CULane/driver_23_30frame/05151640_0419.MP4/'):
        #     if f.endswith('.jpg'):

        # img = cv2.imread(img) #'/media/peter/ocean/data/dataset/lane/CULane/driver_23_30frame/05151640_0419.MP4/00030.jpg')
        # img = cv2.imread(os.path.join('/media/peter/ocean/data/dataset/lane/CULane/driver_23_30frame/05151640_0419.MP4/', f))
        if isinstance(img, str):
            img = cv2.imread(img)
        # img = img[self.cfg.cut_height:, :, :]
        data = img[self.cfg.cut_height:, :, :]
        data = cv2.resize(data, (800, 320))
        data = data.astype(np.float32) / 255.0
        data = torch.Tensor(data)#.long()
        data = data.permute([2, 0, 1]).unsqueeze(0)
        data = data.detach().clone()

        self.net.eval()
        # predictions = []
        # for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
        # data = self.to_cuda(data)
        data = data.cuda()
        with torch.no_grad():
            output = self.net(data)
            output = self.net.module.heads.get_lanes(output)
            # print(output)
        
        # img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes in output:
            # img_name = img_meta['img_name']
            # img = cv2.imread(osp.join(self.data_root, img_name))
            # out_file = osp.join(self.cfg.work_dir, 'visualization',
            #                     img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(
                img, 
                lanes, 
                show=True,
                # out_file=out_file
            )
            
            # return output


            # predictions.extend(output)
        # if self.cfg.view:
        #     self.test_loader.dataset.view(output, data['meta'])

        # metric = self.test_loader.dataset.evaluate(predictions,
        #                                            self.cfg.work_dir)
        # if metric is not None:
        #     self.recorder.logger.info('metric: ' + str(metric))

    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                print(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
