from Backbones.SimpleCNN_lora import SimpleCNN_lora
from Backbones.ResNet import resnet10, resnet12, resnet20, resnet18, resnet34
from Backbones.ResNet_Lora import resnet10_lora, resnet18_lora
from Backbones.ResNet_pretrain import resnet18_pretrained
# from Backbones.SimpleCNN import SimpleCNN
from argparse import Namespace
from Backbones.SimpleCNN import SimpleCNN, SimpleCNN_sr
# from Backbones.SimpleCNN_lora_sr import SimpleCNN_lora_sr

Backbone_NAMES = {
    'simple_cnn': SimpleCNN,
    # 'simple_cnn_lora': SimpleCNN_lora,
    # 'simple_cnn_lora_sr': SimpleCNN_lora_sr,
    # 'simple_cnn_sr': SimpleCNN_sr,
    # 'resnet10': resnet10,
    # 'resnet10_lora': resnet10_lora,
    # 'resnet18_lora': resnet18_lora,
    'resnet18': resnet18,
    'resnet18_pretrained':resnet18_pretrained
}


def get_private_backbones(cfg):
    if type(cfg.DATASET.backbone) == str:
        priv_models = []
        assert cfg.DATASET.backbone in Backbone_NAMES.keys()
        for _ in range(cfg.DATASET.parti_num):
            if 'FedSR' not in cfg:
                priv_model=Backbone_NAMES[cfg.DATASET.backbone](cfg)
            else:
                priv_model = Backbone_NAMES[cfg.DATASET.backbone+'_sr'](cfg)
                priv_model.num_samples=cfg.FedSR.num_samples
            priv_models.append(priv_model)
        return priv_models
