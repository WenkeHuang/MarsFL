from yacs.config import CfgNode as CN
from utils.utils import log_msg


def show_cfg(cfg, optimizer):
    dump_cfg = CN()
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    dump_cfg[optimizer] = cfg[optimizer]
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))
    return dump_cfg


CFG = CN()

CFG.DATASET = CN()
CFG.DATASET.dataset = "Digits"  #
CFG.DATASET.communication_epoch = 2
CFG.DATASET.n_classes = 10

CFG.DATASET.parti_num = 5
CFG.DATASET.online_ratio = 1.0  # online 参与者的ratio
CFG.DATASET.domain_ratio = 1.0  # domain 中采样的ratio
CFG.DATASET.train_eval_domain_ratio = 0.01  # 聚合时每个域采样的ratio
CFG.DATASET.backbone = "resnet18"
CFG.DATASET.pretrained = False
CFG.DATASET.aug = "weak"
CFG.DATASET.beta = 0.5

CFG.OPTIMIZER = CN()
CFG.OPTIMIZER.type = 'SGD'
CFG.OPTIMIZER.momentum = 0.9
CFG.OPTIMIZER.weight_decay = 1e-5
CFG.OPTIMIZER.local_epoch = 10
CFG.OPTIMIZER.local_train_batch = 64
CFG.OPTIMIZER.local_test_batch = 64
CFG.OPTIMIZER.val_batch = 64
CFG.OPTIMIZER.local_train_lr = 1e-3

'''Federated Method'''

# FedAVG
CFG.FedAVG = CN()
CFG.FedAVG.local_method = 'BaseLocal'
CFG.FedAVG.global_method = 'BaseGlobal'

# FedProx
CFG.FedProx = CN()
CFG.FedProx.local_method = 'FedProxLocal'
CFG.FedProx.global_method = 'BaseGlobal'
CFG.FedProx.mu = 0.01
