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
CFG.DATASET.use_two_crop = "ASY"


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
# COPA
CFG.COPA = CN()
# CFG.COPA.pred_threshold = 0.15

# FedADG
CFG.FedADG = CN()
CFG.FedADG.pretrain_epoch = 3
CFG.FedADG.train_epoch = 7
CFG.FedADG.rp_size = 1024
CFG.FedADG.DG_train_lr = 1e-3
CFG.FedADG.DG_momentum = 0.9
CFG.FedADG.DG_weight_decay = 1e-5
CFG.FedADG.D_train_lr = 1e-3
CFG.FedADG.D_momentum = 0.9
CFG.FedADG.D_weight_decay = 1e-5
CFG.FedADG.alpha = 0.15

# FedAVG
CFG.FedAVG = CN()

# FedAVGCOSAddGlobal
CFG.FedAVGCOSAddGlobal = CN()
CFG.FedAVGCOSAddGlobal.alpha = 1.0
CFG.FedAVGCOSAddGlobal.beta = 1.0
CFG.FedAVGCOSAddGlobal.w = 1.0
CFG.FedAVGCOSAddGlobal.temperature = 1.0
CFG.FedAVGCOSAddGlobal.weight_epoch = 10
CFG.FedAVGCOSAddGlobal.weight_lr = 1e-3
CFG.FedAVGCOSAddGlobal.weight_opt_type = 'Adam'

# FedAVGCOSAddNH
CFG.FedAVGCOSAddNH = CN()
CFG.FedAVGCOSAddNH.alpha = 1.0
CFG.FedAVGCOSAddNH.beta = 1.0
CFG.FedAVGCOSAddNH.w = 1.0
CFG.FedAVGCOSAddNH.temperature = 1.0
CFG.FedAVGCOSAddNH.weight_epoch = 10
CFG.FedAVGCOSAddNH.weight_lr = 1e-3
CFG.FedAVGCOSAddNH.weight_opt_type = 'Adam'

# FedAVGCOSNH
CFG.FedAVGCOSNH = CN()
CFG.FedAVGCOSNH.alpha = 1.0
CFG.FedAVGCOSNH.beta = 1.0
CFG.FedAVGCOSNH.w = 1.0
CFG.FedAVGCOSNH.weight_epoch = 10
CFG.FedAVGCOSNH.weight_lr = 1e-3
CFG.FedAVGCOSNH.weight_opt_type = 'Adam'

# FedProx
CFG.FedProx = CN()
CFG.FedProx.mu = 0.01

# FedProxCOSNHNew
CFG.FedProxCOSNHNew = CN()
CFG.FedProxCOSNHNew.weight_epoch = 10
CFG.FedProxCOSNHNew.weight_lr = 1e-3
CFG.FedProxCOSNHNew.weight_opt_type = 'Adam'
CFG.FedProxCOSNHNew.mu = 0.01
CFG.FedProxCOSNHNew.alpha = 1.0
CFG.FedProxCOSNHNew.beta = 1.0
CFG.FedProxCOSNHNew.w = 1.0
CFG.FedProxCOSNHNew.temperature = 1.0

CFG.FedProxCOSAddNHNew = CN()
CFG.FedProxCOSAddNHNew.weight_epoch = 10
CFG.FedProxCOSAddNHNew.weight_lr = 1e-3
CFG.FedProxCOSAddNHNew.weight_opt_type = 'Adam'
CFG.FedProxCOSAddNHNew.mu = 0.01
CFG.FedProxCOSAddNHNew.alpha = 1.0
CFG.FedProxCOSAddNHNew.beta = 1.0
CFG.FedProxCOSAddNHNew.w = 1.0
CFG.FedProxCOSAddNHNew.temperature = 1.0


CFG.FedProxCOSAddGlobal = CN()
CFG.FedProxCOSAddGlobal.weight_epoch = 10
CFG.FedProxCOSAddGlobal.weight_lr = 1e-3
CFG.FedProxCOSAddGlobal.weight_opt_type = 'Adam'
CFG.FedProxCOSAddGlobal.mu = 0.01
CFG.FedProxCOSAddGlobal.alpha = 1.0
CFG.FedProxCOSAddGlobal.beta = 1.0
CFG.FedProxCOSAddGlobal.w = 1.0
CFG.FedProxCOSAddGlobal.temperature = 1.0

CFG.FedProxCOSAddSoft = CN()
CFG.FedProxCOSAddSoft.weight_epoch = 10
CFG.FedProxCOSAddSoft.weight_lr = 1e-3
CFG.FedProxCOSAddSoft.weight_opt_type = 'Adam'
CFG.FedProxCOSAddSoft.mu = 0.01
CFG.FedProxCOSAddSoft.alpha = 1.0
CFG.FedProxCOSAddSoft.beta = 1.0
CFG.FedProxCOSAddSoft.smooth = 1.0

CFG.FedProxCOSAddNHMad = CN()
CFG.FedProxCOSAddNHMad.weight_epoch = 10
CFG.FedProxCOSAddNHMad.weight_lr = 1e-3
CFG.FedProxCOSAddNHMad.weight_opt_type = 'Adam'
CFG.FedProxCOSAddNHMad.mu = 0.01
CFG.FedProxCOSAddNHMad.alpha = 1.0
CFG.FedProxCOSAddNHMad.beta = 1.0
CFG.FedProxCOSAddNHMad.w = 1.0
CFG.FedProxCOSAddNHMad.temperature=1.0

# FedProxGA
CFG.FedProxGA = CN()
CFG.FedProxGA.mu = 0.01
CFG.FedProxGA.base_step_size = 0.2

# FedDyn
CFG.FedDyn = CN()
CFG.FedDyn.reg_lamb = 1e-3

# FedOpt
CFG.FedOpt = CN()
CFG.FedOpt.global_lr = 0.5

# MOONCOSAddNH
CFG.MOONCOSAddNH = CN()
CFG.MOONCOSAddNH.mu = 5.0
CFG.MOONCOSAddNH.temperature_moon = 0.5
CFG.MOONCOSAddNH.alpha = 1.0
CFG.MOONCOSAddNH.beta = 1.0
CFG.MOONCOSAddNH.w = 1.0
CFG.MOONCOSAddNH.temperature = 1.0
CFG.MOONCOSAddNH.weight_epoch = 10
CFG.MOONCOSAddNH.weight_lr = 1e-3
CFG.MOONCOSAddNH.weight_opt_type = 'Adam'

# MOONCOSNH
CFG.MOONCOSNH = CN()
CFG.MOONCOSNH.mu = 5.0
CFG.MOONCOSNH.temperature_moon = 0.5
CFG.MOONCOSNH.alpha = 1.0
CFG.MOONCOSNH.beta = 1.0
CFG.MOONCOSNH.w = 1.0
CFG.MOONCOSNH.temperature = 1.0
CFG.MOONCOSNH.weight_epoch = 10
CFG.MOONCOSNH.weight_lr = 1e-3
CFG.MOONCOSNH.weight_opt_type = 'Adam'

# MOONCOSAddGlobal
CFG.MOONCOSAddGlobal = CN()
CFG.MOONCOSAddGlobal.mu = 5.0
CFG.MOONCOSAddGlobal.temperature_moon = 0.5
CFG.MOONCOSAddGlobal.alpha = 1.0
CFG.MOONCOSAddGlobal.beta = 1.0
CFG.MOONCOSAddGlobal.w = 1.0
CFG.MOONCOSAddGlobal.temperature = 1.0
CFG.MOONCOSAddGlobal.weight_epoch = 10
CFG.MOONCOSAddGlobal.weight_lr = 1e-3
CFG.MOONCOSAddGlobal.weight_opt_type = 'Adam'

# MOON
CFG.MOON = CN()
CFG.MOON.mu = 5.0
CFG.MOON.temperature_moon = 0.5

# FedProc
CFG.FedProc = CN()

# FedProto
CFG.FedProto = CN()
CFG.FedProto.mu = 1.0

# FPL
CFG.FPL = CN()
CFG.FPL.infoNCET = 0.02

# FedSR
CFG.FedSR = CN()
CFG.FedSR.L2R_coeff = 1e-2
CFG.FedSR.CMI_coeff = 5e-4
CFG.FedSR.num_samples = 20

# FedDG
CFG.FedDG = CN()
CFG.FedDG.meta_step_size = 1e-3
CFG.FedDG.clip_value = 1.0

# FedRC
CFG.FedRC = CN()
CFG.FedRC.weight_epoch = 10
CFG.FedRC.weight_lr = 1e-3
CFG.FedRC.weight_opt_type = 'Adam'
CFG.FedRC.mu = 0.01
CFG.FedRC.alpha = 1.0
CFG.FedRC.beta = 1.0
CFG.FedRC.w = 1.0
CFG.FedRC.temperature = 1.0

# FedProxRC
CFG.FedProxRC = CN()
CFG.FedProxRC.weight_epoch = 10
CFG.FedProxRC.weight_lr = 1e-2
CFG.FedProxRC.weight_opt_type = 'Adam'
CFG.FedProxRC.mu = 0.01
CFG.FedProxRC.alpha = 1.0
CFG.FedProxRC.beta = 1.0
CFG.FedProxRC.w = 1.0
CFG.FedProxRC.temperature = 1.0

# FedProxRCO
CFG.FedProxRCO = CN()
CFG.FedProxRCO.weight_epoch = 10
CFG.FedProxRCO.weight_lr = 1e-2
CFG.FedProxRCO.weight_opt_type = 'Adam'
CFG.FedProxRCO.mu = 0.01
CFG.FedProxRCO.alpha = 1.0
CFG.FedProxRCO.beta = 1.0
CFG.FedProxRCO.w = 1.0
CFG.FedProxRCO.temperature = 1.0

# FedR 默认 TTA
CFG.FedR = CN()
CFG.FedR.weight_epoch = 10
CFG.FedR.weight_lr = 1e-3
CFG.FedR.weight_opt_type = 'Adam'
CFG.FedR.mu = 0.01
CFG.FedR.alpha = 1.0
CFG.FedR.beta = 1.0

# FedC
CFG.FedC = CN()
CFG.FedC.weight_epoch = 10
CFG.FedC.weight_lr = 1e-3
CFG.FedC.weight_opt_type = 'Adam'
CFG.FedC.mu = 0.01
CFG.FedC.alpha = 1.0
CFG.FedC.beta = 1.0
CFG.FedC.gamma = 1.0
CFG.FedC.temperature = 1.0

# FedCos
CFG.FedCos = CN()
CFG.FedCos.weight_epoch = 10
CFG.FedCos.weight_lr = 1e-3
CFG.FedCos.weight_opt_type = 'Adam'
CFG.FedCos.mu = 0.01
CFG.FedCos.alpha = 1.0
CFG.FedCos.beta = 1.0
CFG.FedCos.temperature = 1.0

# # FedProxCOS
# CFG.FedProxCOS = CN()
# CFG.FedProxCOS.weight_epoch = 10
# CFG.FedProxCOS.weight_lr = 1e-3
# CFG.FedProxCOS.weight_opt_type = 'Adam'
# CFG.FedProxCOS.mu = 0.01
# CFG.FedProxCOS.alpha = 1.0
# CFG.FedProxCOS.beta = 0
# CFG.FedProxCOS.w = 1.0
# CFG.FedProxCOS.temperature = 1.0

# # FedProxCOSNH
# CFG.FedProxCOSNH = CN()
# CFG.FedProxCOSNH.weight_epoch = 10
# CFG.FedProxCOSNH.weight_lr = 1e-3
# CFG.FedProxCOSNH.weight_opt_type = 'Adam'
# CFG.FedProxCOSNH.mu = 0.01
# CFG.FedProxCOSNH.alpha = 1.0
# CFG.FedProxCOSNH.beta = 1.0
# CFG.FedProxCOSNH.w = 1.0
# CFG.FedProxCOSNH.temperature = 1.0

# # FedProxCOSAdd
# CFG.FedProxCOSAdd = CN()
# CFG.FedProxCOSAdd.weight_epoch = 10
# CFG.FedProxCOSAdd.weight_lr = 1e-3
# CFG.FedProxCOSAdd.weight_opt_type = 'Adam'
# CFG.FedProxCOSAdd.mu = 0.01
# CFG.FedProxCOSAdd.alpha = 1.0
# CFG.FedProxCOSAdd.beta = 1.0
# CFG.FedProxCOSAdd.w = 1.0
# CFG.FedProxCOSAdd.temperature = 1.0
#
# # FedProxCOSAddNH
# CFG.FedProxCOSAddNH = CN()
# CFG.FedProxCOSAddNH.weight_epoch = 10
# CFG.FedProxCOSAddNH.weight_lr = 1e-3
# CFG.FedProxCOSAddNH.weight_opt_type = 'Adam'
# CFG.FedProxCOSAddNH.mu = 0.01
# CFG.FedProxCOSAddNH.alpha = 1.0
# CFG.FedProxCOSAddNH.beta = 1.0
# CFG.FedProxCOSAddNH.w = 1.0
# CFG.FedProxCOSAddNH.temperature = 1.0

# FedProxCOSAddNHNew


# FedCe
CFG.FedCe = CN()
CFG.FedCe.weight_epoch = 10
CFG.FedCe.weight_lr = 1e-3
CFG.FedCe.weight_opt_type = 'Adam'
CFG.FedCe.mu = 0.01
CFG.FedCe.alpha = 1.0
CFG.FedCe.beta = 1.0
CFG.FedCe.temperature = 1.0
