import torch.nn as nn

from clrnet.models.backbones.resnet import resnet18
from clrnet.models.necks.fpn import FPN

from aio_clrnet_head import CLRHead

class CLRNet(nn.Module):

    def __init__(self) -> None:

        # backbone
        self.backbone = resnet18()

        # neck
        self.neck = FPN(in_channels=[128, 256, 512], out_channels=64, num_outs=3, attention=False)

        # head
        self.clr_head = CLRHead()
        self.clr_head.training = False


    def forward(self, x):
        pass


    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        return super().load_state_dict(state_dict, strict)

