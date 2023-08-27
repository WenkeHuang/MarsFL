from utils.conf import get_device, checkpoint_path, net_path
from utils.utils import create_if_not_exists
from argparse import Namespace
import torch.nn as nn
import numpy as np
import torch
import os


class SeverMethod(nn.Module):
    """
    Federated learning Methods.
    """
    NAME = None

    def __init__(self, args, cfg) -> None:
        super(SeverMethod, self).__init__()

        self.args = args
        self.cfg = cfg
        self.device = get_device(device_id=self.args.device_id)

    def sever_update(self, **kwargs):
        pass

