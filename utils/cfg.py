from yacs.config import CfgNode as CN
from utils.utils import log_msg


# 简化cfg 只留有关的
def simplify_cfg(args, cfg):
    dump_cfg = CN()
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    dump_cfg[args.method] = cfg[args.method]
    dump_cfg[args.task] = cfg[args.task]

    # simplify Sever cfg
    if cfg[args.method].global_method in list(cfg['Sever'].keys()):
        dump_cfg['Sever'] = CN()
        dump_cfg['Sever'][cfg[args.method].global_method] = CN()
        dump_cfg['Sever'][cfg[args.method].global_method] = cfg['Sever'][cfg[args.method].global_method]

    # simplify Local cfg
    if cfg[args.method].local_method in list(cfg['Local'].keys()):
        dump_cfg['Local'] = CN()
        dump_cfg['Local'][cfg[args.method].local_method] = CN()
        dump_cfg['Local'][cfg[args.method].local_method] = cfg['Local'][cfg[args.method].local_method]

    if args.attack_type != 'None':
        dump_cfg['attack'] = CN()
        dump_cfg['attack'].bad_client_rate = cfg['attack'].bad_client_rate
        dump_cfg['attack'].noise_data_rate = cfg['attack'].bad_client_rate
        dump_cfg['attack'][args.attack_type] = cfg['attack'][args.attack_type]

    return dump_cfg


def show_cfg(cfg, method):
    dump_cfg = CN()
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    dump_cfg[method] = cfg[method]
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))
    return dump_cfg


CFG = CN()
'''Federated dataset'''
CFG.DATASET = CN()
CFG.DATASET.dataset = "fl_cifar10"  #
CFG.DATASET.communication_epoch = 2
CFG.DATASET.n_classes = 10

CFG.DATASET.parti_num = 4
CFG.DATASET.online_ratio = 1.0  # online 参与者的ratio
CFG.DATASET.domain_ratio = 1.0  # domain 中采样的ratio
CFG.DATASET.train_eval_domain_ratio = 0.01  # 聚合时每个域采样的ratio
CFG.DATASET.backbone = "resnet18"
CFG.DATASET.pretrained = False
CFG.DATASET.aug = "weak"
CFG.DATASET.beta = 0.5

'''attack'''
CFG.attack = CN()
CFG.attack.bad_client_rate = 0.4
CFG.attack.noise_data_rate = 0.5

CFG.attack.byzantine = CN()
CFG.attack.byzantine.evils = 'min_sum'  # PairFlip SymFlip RandomNoise lie_attack min_max min_sum
CFG.attack.byzantine.dataset_type = 'single_domain'

# attack para for min_max and min_sum
CFG.attack.byzantine.dev_type = 'std'
CFG.attack.byzantine.lamda = 10.0
CFG.attack.byzantine.threshold_diff = 1e-5

CFG.attack.backdoor = CN()
CFG.attack.backdoor.evils = 'base_backdoor'  # base_backdoor semantic_backdoor
CFG.attack.backdoor.backdoor_label = 2
CFG.attack.backdoor.trigger_position = [
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 5], [0, 0, 6],
    [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
CFG.attack.backdoor.trigger_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]

CFG.attack.backdoor.semantic_backdoor_label = 3

'''task'''
# label_skew
CFG.label_skew = CN()

# domain_skew
CFG.domain_skew = CN()

# OOD
CFG.OOD = CN()
# Digits: MNIST, USPS, SVHN, SYN
# PACS: 'photo', 'art_painting', 'cartoon', 'sketch'
# OfficeCaltech 'caltech', 'amazon','webcam','dslr'
# OfficeHome 'Art', 'Clipart', 'Product', 'Real_World'
# DomainNet 'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
CFG.OOD.out_domain = 'MNIST'

'''Federated OPTIMIZER'''
CFG.OPTIMIZER = CN()
CFG.OPTIMIZER.type = 'SGD'
CFG.OPTIMIZER.momentum = 0.9
CFG.OPTIMIZER.weight_decay = 1e-5
CFG.OPTIMIZER.local_epoch = 2
CFG.OPTIMIZER.local_train_batch = 64
CFG.OPTIMIZER.local_test_batch = 64
CFG.OPTIMIZER.val_batch = 64
CFG.OPTIMIZER.local_train_lr = 1e-3

'''Sever'''
CFG.Sever = CN()

CFG.Sever.FedOptSever = CN()
CFG.Sever.FedOptSever.global_lr = 0.5

CFG.Sever.FLTrustSever = CN()
CFG.Sever.FLTrustSever.public_dataset_name = 'pub_svhn'
CFG.Sever.FLTrustSever.pub_len = 5000
CFG.Sever.FLTrustSever.pub_aug = 'weak'
CFG.Sever.FLTrustSever.public_batch_size = 64
CFG.Sever.FLTrustSever.public_epoch = 20

CFG.Sever.SageFlowSever = CN()
CFG.Sever.SageFlowSever.public_dataset_name = 'pub_svhn'
CFG.Sever.SageFlowSever.pub_len = 5000
CFG.Sever.SageFlowSever.pub_aug = 'weak'
CFG.Sever.SageFlowSever.public_batch_size = 64
CFG.Sever.SageFlowSever.public_epoch = 20

CFG.Sever.KD3ASever = CN()
CFG.Sever.KD3ASever.confidence_gate_begin = 0.9
CFG.Sever.KD3ASever.confidence_gate_end = 0.95

CFG.Sever.FedProxGASever = CN()
CFG.Sever.FedProxGASever.base_step_size = 0.2

CFG.Sever.CRFLSever = CN()
CFG.Sever.CRFLSever.param_clip_thres = 15
CFG.Sever.CRFLSever.epoch_index_weight = 0.1
CFG.Sever.CRFLSever.epoch_index_bias = 2
CFG.Sever.CRFLSever.sigma = 0.01

'''Local'''
CFG.Local = CN()

CFG.Local.FedProxLocal = CN()
CFG.Local.FedProxLocal.mu = 0.01

CFG.Local.MOONLocal = CN()
CFG.Local.MOONLocal.mu = 1
CFG.Local.MOONLocal.temperature_moon = 0.5

CFG.Local.FedProtoLocal = CN()
CFG.Local.FedProtoLocal.mu = 2

CFG.Local.FedDynLocal = CN()
CFG.Local.FedDynLocal.reg_lamb = 1e-2

CFG.Local.FPLLocal = CN()
CFG.Local.FPLLocal.infoNCET = 0.2

CFG.Local.FedNTDLocal = CN()
CFG.Local.FedNTDLocal.tau = 3.0
CFG.Local.FedNTDLocal.beta = 1.0

CFG.Local.ScaffoldLocal = CN()
CFG.Local.ScaffoldLocal.max_grad_norm = 100

CFG.Local.FedLCLocal = CN()
CFG.Local.FedLCLocal.tau = 0.5

CFG.Local.FedNovaLocal = CN()
CFG.Local.FedNovaLocal.rho = 0.9

CFG.Local.FedRSLocal = CN()
CFG.Local.FedRSLocal.alpha = 0.5

CFG.Local.qffeAVGLocal = CN()
CFG.Local.qffeAVGLocal.q = 0.05

CFG.Local.CRFLLocal = CN()
CFG.Local.CRFLLocal.scale_factor = 100

'''Federated Method'''
# qffeAVG
CFG.qffeAVG = CN()
CFG.qffeAVG.local_method = 'qffeAVGLocal'
CFG.qffeAVG.global_method = 'qffeAVGSever'

# FedAVG
CFG.FedAVG = CN()
CFG.FedAVG.local_method = 'BaseLocal'
CFG.FedAVG.global_method = 'BaseSever'

# FedProx
CFG.FedProx = CN()
CFG.FedProx.local_method = 'FedProxLocal'
CFG.FedProx.global_method = 'BaseSever'

# FedProxGA
CFG.FedProxGA = CN()
CFG.FedProxGA.local_method = 'FedProxLocal'
CFG.FedProxGA.global_method = 'FedProxGASever'

# FedProxDefense
CFG.FedProxDefense = CN()
CFG.FedProxDefense.local_method = 'FedProxLocal'
CFG.FedProxDefense.global_method = 'DncSever'

# FedProc
CFG.FedProc = CN()
CFG.FedProc.local_method = 'FedProcLocal'
CFG.FedProc.global_method = 'FedProcSever'

# FedProto
CFG.FedProto = CN()
CFG.FedProto.local_method = 'FedProtoLocal'
CFG.FedProto.global_method = 'FedProtoSever'

# FPL
CFG.FPL = CN()
CFG.FPL.local_method = 'FPLLocal'
CFG.FPL.global_method = 'FPLSever'

# FedOpt
CFG.FedOpt = CN()
CFG.FedOpt.local_method = 'BaseLocal'
CFG.FedOpt.global_method = 'FedOptSever'

# Moon
CFG.MOON = CN()
CFG.MOON.local_method = 'MOONLocal'
CFG.MOON.global_method = 'BaseSever'

# FedDyn
CFG.FedDyn = CN()
CFG.FedDyn.local_method = 'FedDynLocal'
CFG.FedDyn.global_method = 'BaseSever'

# Scaffold
CFG.Scaffold = CN()
CFG.Scaffold.local_method = 'ScaffoldLocal'
CFG.Scaffold.global_method = 'ScaffoldSever'

# FedLC
CFG.FedLC = CN()
CFG.FedLC.local_method = 'FedLCLocal'
CFG.FedLC.global_method = 'BaseSever'

# FedRS
CFG.FedRS = CN()
CFG.FedRS.local_method = 'FedRSLocal'
CFG.FedRS.global_method = 'BaseSever'

# FedNTD
CFG.FedNTD = CN()
CFG.FedNTD.local_method = 'FedNTDLocal'
CFG.FedNTD.global_method = 'BaseSever'

# FedNova
CFG.FedNova = CN()
CFG.FedNova.local_method = 'FedNovaLocal'
CFG.FedNova.global_method = 'FedNovaSever'

# KD3A
CFG.KD3A = CN()
CFG.KD3A.local_method = 'BaseLocal'
CFG.KD3A.global_method = 'KD3ASever'

# FADA
CFG.FADA = CN()
CFG.FADA.local_method = 'BaseLocal'
CFG.FADA.global_method = 'FADASever'

# COPADA
CFG.COPADA = CN()
CFG.COPADA.local_method = 'COPALocal'
CFG.COPADA.global_method = 'COPAGSever'

# COPADG
CFG.COPADG = CN()
CFG.COPADG.local_method = 'COPALocal'
CFG.COPADG.global_method = 'COPADGSever'

# CRFL
CFG.CRFL = CN()
CFG.CRFL.local_method = 'CRFLLocal'
CFG.CRFL.global_method = 'CRFLSever'
