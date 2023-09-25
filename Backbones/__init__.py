from Backbones.ResNet import resnet10, resnet12, resnet20, resnet18, resnet34, resnet50
from Backbones.ResNet_pretrain import resnet18_pretrained
from Backbones.SimpleCNN import SimpleCNN, SimpleCNN_sr

Backbone_NAMES = {
    'simple_cnn': SimpleCNN,
    'resnet10': resnet10,
    'resnet18': resnet18,
    'resnet18_pretrained': resnet18_pretrained,
    'resnet34':resnet34,
    'resnet50': resnet50
}


def get_private_backbones(cfg):
    if type(cfg.DATASET.backbone) == str:
        priv_models = []
        assert cfg.DATASET.backbone in Backbone_NAMES.keys()
        for _ in range(cfg.DATASET.parti_num):
            if 'FedSR' not in cfg:
                priv_model = Backbone_NAMES[cfg.DATASET.backbone](cfg)
            else:
                priv_model = Backbone_NAMES[cfg.DATASET.backbone + '_sr'](cfg)
                priv_model.num_samples = cfg.FedSR.num_samples
            priv_models.append(priv_model)
        return priv_models
