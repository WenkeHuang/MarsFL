# Generalization Federated Learning

##  Crosss Calibration
Guancheng Wan

|  Method   | Venue  | Code | Paper | Check|
|  ----  | ----  | ----  | ----  |----  |
| FedAvg | AISTATS‘17 | [Code](https://github.com/katsura-jp/fedavg.pytorch)|[Paper](https://arxiv.org/abs/1602.05629)| Yes|
| FedProx| MLSys'20 | [Code](https://github.com/ki-ljl/FedProx-PyTorch) |[Paper](https://arxiv.org/abs/1812.06127)|Yes|
|Scaffold|ICML 20| [Code]() | [Paper](https://arxiv.org/abs/1910.06378)| No |
|FedNova| NeurIPS'20| [Code](https://github.com/JYWa/FedNova/tree/master/distoptim)| No |
| FedBE| ICLR'21 | [Code](https://github.com/hongyouc/FedBE) |[Paper](https://arxiv.org/abs/2009.01974)| No|
| FedRS| SIGKDD'21 | [Code](https://github.com/lxcnju/FedRepo/tree/main/algorithms) |[Paper](https://dlnext.acm.org/doi/10.1145/3447548.3467254)| 之前仓库|
| FedOPT| ICLR'21 | [Code](https://github.com/lxcnju/FedRepo/tree/main/algorithms) |[Paper](https://arxiv.org/abs/2003.002957)|Yes|
|MOON | CVPR'21 |[Code](https://github.com/QinbinLi/MOON)|[Paper]()| Yes|
|FedProc| arXiv'21 | [Code](https://github.com/973891422/Moon_FedProc)| [Paper](https://github.com/QinbinLi/MOON)|Yes|
| FedDyn| ICLR'21 | [Code1](https://github.com/alpemreacar/FedDyn) [Code2](https://github.com/lxcnju/FedRepo/tree/main/algorithms) |[Paper](https://arxiv.org/abs/2003.002957)| Yes|
| FedDC| CVPR'22 | [Code](https://github.com/gaoliang13/FedDC) |[Paper] [Zhihu](https://zhuanlan.zhihu.com/p/505889549)|
| FedNTD| NeurIPS'22 | [Code](https://github.com/Lee-Gihun/FedNTD) |[Paper](https://arxiv.org/abs/2106.03097)|
| FedLC| ICML'22 | [Code] |[Paper](https://proceedings.mlr.press/v162/zhang22p.html)| 
| FPL | CVPR'23 | [Code](https://github.com/WenkeHuang/RethinkFL) |[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)|


## Unknown Generalization

### Federated Domain Adaptation >=2
Guancheng Wan

|  Method   | Venue  | Code | Paper
|  ----  | ----  |----  |----  |
| FADA | ICLR‘20 | [Code](https://drive.google.com/file/d/1OekTpqB6qLfjlE2XUjQPm3F110KDMFc0/view)| [Paper](https://arxiv.org/abs/1911.02054)|
| COPA | ICCV'21 | [Code]| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.pdf)|
| KD3A | ICML‘21 | [Code](https://github.com/FengHZ/KD3A)| [Paper](https://arxiv.org/abs/1911.02054)|
| MCC-DA  | TCSVT'22 | [Code]| [Paper](https://ieeexplore.ieee.org/document/9940295)|

### Federated Domain Generalization >=2
Zekun Shi

|  Method   | Venue  | Code | Paper
|  ----  | ----  |----  |----  |
| COPA | ICCV'21 | [Code]| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.pdf)|
| CSAC | TKDE'23 | [Code](https://github.com/junkunyuan/CSAC)|[Paper](https://arxiv.org/pdf/2110.06736.pdf)|
| FedGA | CVPR 23 | [Code](https://github.com/MediaBrain-SJTU/FedDG-GA)| [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf)|


# Robustness Federated Learning
Based on FedProx

## Byzantine Attack
|  Method   | Venue  | Code | Paper
|  ----  | ----  |----  |----  |
| SymF | NeurIPS'15 | | [Paper](https://arxiv.org/abs/1505.07634)|
| PairF | NeurIPS'18 | |[Paper](https://arxiv.org/abs/1804.06872)|
| RandomNoise | - | - |-|
| LIE | USNEIX'20 | [Code](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)| [Paper](https://arxiv.org/abs/1902.06156) |
| Fang | USNEIX'20 | [Code](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)| [Paper](https://www.usenix.org/system/files/sec20-fang.pdf)|
| MiMa | NDSS'21  | [Code](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning) |[ Paper](https://www.ndss-symposium.org/ndss-program/ndss-2021/)|
| MiSu | NDSS'21 |[Code](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)  |[Paper](https://www.ndss-symposium.org/ndss-program/ndss-2021/)|

## Backdoor Attack

https://github.com/THUYimingLi/backdoor-learning-resources#federated-learning

|  Method   | Venue  | Code | Paper
|  ----  | ----  |----  |----  |
| Backdoor | arXiv'17 | [Code](https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning/blob/master/clients_attackers.py#L34) | [Paper](https://arxiv.org/abs/1712.05526) |
| Semantic Backdoor | AISTATS'20 | [Code](https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning/blob/master/clients_attackers.py#L34) | [Paper](https://arxiv.org/pdf/1807.00459.pdf)
| F3BA | AAAI'23 | [Code](https://github.com/jinghuichenFocused-Flip-Federated-Backdoor-Attack)| [Paper](https://arxiv.org/abs/2301.08170)|
| FLIP | ICLR'23 | [Code](https://github.com/KaiyuanZh/FLIP) | [Paper](https://github.com/KaiyuanZh/FLIP)


## Byzantine Defense

## Backdoor Defense
Zekun Shi
|  Method   | Venue  | Code | Paper
|  ----  | ----  |----  |----  |
| RLR | AAAI'21 | [Code](https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate) | [Paper](https://arxiv.org/pdf/2007.03767.pdf)|
| CRFL | ICML'21 | [Code](https://github.com/AI-secure/CRFL) | [Paper](https://arxiv.org/pdf/2106.08283.pdf)|


# Fairness Federated Learning

## Collaboration Fairness >=2
Zekun Shi

|  Method   | Venue  | Code | Paper
|  ----  | ----  |----  |----  |
| CFFL | FL'20 | [Code](https://github.com/XinyiYS/CollaborativeFairFederatedLearning)| [Paper](https://arxiv.org/abs/2008.12161)|
| CGSV | NeurIPS'21 | [Code](https://github.com/XinyiYS/Gradient-Driven-Rewards-to-Guarantee-Fairness-in-Collaborative-Machine-Learning)| [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/8682cc30db9c025ecd3fee433f8ab54c-Paper.pdf)|
| RRFL | ICML'21 | [Code](https://github.com/XinyiYS/Robust-and-Fair-Federated-Learning)| [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/8682cc30db9c025ecd3fee433f8ab54c-Paper.pdf)|



## Performance Fairness
Zekun Shi

| Method | Venue      | Code | Paper
|--------|------------|----  |----  |
| **AFL** | ICML'19 | [Code](https://github.com/Chelsiehi/Agnostic-Federated-Learning/tree/master)| [Paper](https://arxiv.org/abs/1902.00146)|
| **qFFL**   | ICLR‘20    | [Code](https://github.com/illidanlab/FADE)| [Paper](https://arxiv.org/abs/1905.10497)|
| FADE   | SIGKDD'21  |[Code](https://github.com/illidanlab/FADE)| [Paper]()|
| FCFL   | NeurIPS'21 | [Code](https://github.com/cuis15/FCFL)| [Paper](https://arxiv.org/pdf/2108.08435.pdf)|
| FedFV  | IJCAI'21   | [Code](https://github.com/WwZzz/easyFL)| [Paper](https://arxiv.org/abs/2104.14937)|
| **Ditto**  | ICML'21   | [Code](https://github.com/litian96/ditto)| [Paper](https://arxiv.org/pdf/2012.04221.pdf)|