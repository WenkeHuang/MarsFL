import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.loralib.layers import *


class Conv_Net_Wrap(LoRALayer):
    LORA_PARAM_MODEL = LoRALayer(0, 1, 0., True)

    @staticmethod
    def set_conv_net_lora_param(
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = True,
    ):
        '''设置lora 参数'''
        temp = Conv_Net_Wrap.LORA_PARAM_MODEL
        temp.r = r
        temp.lora_alpha = lora_alpha
        temp.lora_dropout = lora_dropout
        temp.merge_weights = merge_weights

    @staticmethod
    def conv(input_channel, output_channel, kernel_size, bias=False):
        lora_param = Conv_Net_Wrap.LORA_PARAM_MODEL
        return Conv2d(input_channel, output_channel, kernel_size=kernel_size, bias=bias, r=lora_param.r,
                      lora_alpha=lora_param.lora_alpha, lora_dropout=lora_param.lora_dropout)


class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = Conv_Net_Wrap.conv(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = Conv_Net_Wrap.conv(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class SimpleCNN_lora(nn.Module):

    def __init__(self, cfg):
        super(SimpleCNN_lora, self).__init__()
        Conv_Net_Wrap.set_conv_net_lora_param(cfg.DATASET.rank, cfg.DATASET.lora_alpha, cfg.DATASET.lora_dropout)
        self.name = 'SimpleCNN'
        self.feats = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=cfg.DATASET.n_classes)
        num_ftrs = 84

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, 256)

        # last layer
        self.cls = nn.Linear(256, cfg.DATASET.n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feats(x)
        h = h.squeeze()
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        return h

    def classifier(self, h: torch.Tensor) -> torch.Tensor:
        y = self.cls(h)
        return y

    def forward(self, x):
        h = self.feats(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.cls(x)
        return y
