import os

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

    def infer(self, x):
        
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
runner = Runner()
