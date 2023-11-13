# A Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark
> Wenke Huang, Mang Ye, Zekun Shi, Guancheng Wan, He Li, Bo Du, Qiang Yang
> [Link](https://github.com/WenkeHuang/MarsFL)

By [MARS](https://marswhu.github.io/index.html) Group at the [Wuhan University](https://www.whu.edu.cn/), led by [Prof. Mang Ye](https://marswhu.github.io/index.html).

## Abstract
Federated learning has emerged as a promising paradigm for privacy-preserving collaboration among different parties. Recently, with the popularity of federated learning, an influx of approaches have delivered towards different realistic challenges. In this survey, we provide a systematic overview of the important and recent developments of research on federated learning. Firstly, we introduce the study history and terminology definition of this area. Then, we comprehensively review three basic lines of research: generalization, robustness, and fairness, by introducing their respective background concepts, task settings, and main challenges. We also offer a detailed overview of representative literature on both methods and datasets. We further benchmark the reviewed methods on several well-known datasets. Finally, we point out several open issues in this field and suggest opportunities for further research. We also provide a public website to continuously track developments in this fast advancing field.


## Our Works
  - [Federated Learning Survey](#federated-learning-survey)
  - [Heterogeneity Federated Learning](#heterogeneity-federated-learning)
- [Robustness Federated Learning](#robustness-federated-learning)

### Federated Learning Survey

- [A Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark]()
 
- [Heterogeneous Federated Learning: State-of-the-art and Research Challenges](https://arxiv.org/abs/2307.10616) *ACM Computing Surveys 2023* [[Code](https://github.com/marswhu/HFL_Survey?utm_source=catalyzex.com)]

### Heterogeneity Federated Learning

- **FCCL+** — [Generalizable Heterogeneous Federated Cross-Correlation and Instance Similarity Learning](https://arxiv.org/pdf/2309.16286.pdf) *TPAMI 2023* [[Code](https://github.com/WenkeHuang/FCCL)]
  
- **FPL** — [Rethinking Federated Learning with Domain Shift: A Prototype View](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf) *CVPR 2023* [[Code](https://github.com/WenkeHuang/RethinkFL)]

- **FGSSL** — [Federated Graph Semantic and Structural Learning](https://marswhu.github.io/publications/files/FGSSL.pdf) *IJCAI 2023* [[Code](https://github.com/WenkeHuang/FGSSL)]

- **FCCL** — [Learn from Others and Be Yourself in Heterogeneous Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf) *CVPR 2022* [[Code](https://github.com/WenkeHuang/FCCL)]

### Robustness Federated Learning

- **DynamicPFL** — [Dynamic Personalized Federated Learning with Adaptive Differential Privacy](https://openreview.net/pdf?id=RteNLuc8D9) *NeurIPS 2023* [[Code](https://github.com/xiyuanyang45/DynamicPFL)]

- **AugHFL** — [Robust Heterogeneous Federated Learning under Data Corruption](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_Robust_Heterogeneous_Federated_Learning_under_Data_Corruption_ICCV_2023_paper.pdf) *ICCV 2023* [[Code](https://github.com/FangXiuwen/AugHFL)]
  
- **RHFL** — [Robust Federated Learning With Noisy and Heterogeneous Clients](https://openaccess.thecvf.com/content/CVPR2022/papers/Fang_Robust_Federated_Learning_With_Noisy_and_Heterogeneous_Clients_CVPR_2022_paper.pdf) *CVPR 2022* [[Code](https://github.com/fangxiuwen/robust_fl)]

- **FSMAFL** — [Few-Shot Model Agnostic Federated Learning](https://dl.acm.org/doi/10.1145/3503161.3548764) *ACMMM 2022* [[Code](https://github.com/FangXiuwen/FSMAFL)]

## Citation

Please kindly cite these papers in your publications if it helps your research:
```bibtex
@article{HFL_CSUR23,
  title={Heterogeneous Federated Learning: State-of-the-art and Research Challenges},
  author={Ye, Mang and Fang, Xiuwen and Du, Bo and Yuen, Pong C and Tao, Dacheng},
  journal={CSUR},
  year={2023}
}
@article{FLSurveyandBenchmarkforGenRobFair_arXiv23,
  title={A Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark},
  author={Wenke Huang and Mang Ye and Zekun Shi and Guancheng Wan and He Li and Bo Du and Qiang Yang},
  journal={arXiv},
  year={2023}
}
@article{FCCLPlus_TPAMI23,
    title={Generalizable Heterogeneous Federated Cross-Correlation and Instance Similarity Learning}, 
    author={Wenke Huang and Mang Ye and Zekun Shi and Bo Du},
    year={2023},
    journal={TPAMI}
}
@inproceedings{DynamicPFL_NeurIPS23,
    title={Dynamic Personalized Federated Learning with Adaptive Differential Privacy},
    author={Yang, Xiyuan and Huang, Wenke and Ye, Mang},
    booktitle={NeurIPS},
    year={2023},
}
@inproceedings{FPL_CVPR23,
    title={Rethinking Federated Learning with Domain Shift: A Prototype View},
    author={Huang, Wenke and Ye, Mang and Shi, Zekun and Li, He and Du, Bo},
    booktitle={CVPR},
    year={2023}
}

@inproceedings{FGSSL_IJCAI23,
    title={Federated Graph Semantic and Structural Learning},
    author={Huang, Wenke and Wan, Guancheng and Ye, Mang and Du, Bo},
    booktitle={IJCAI},
    year={2023}
}
@inproceedings{AugHFL_ICCV23,
  title={Robust heterogeneous federated learning under data corruption},
  author={Fang, Xiuwen and Ye, Mang and Yang, Xiyuan},
  booktitle={ICCV},
  pages={5020--5030},
  year={2023}
}
@inproceedings{FCCL_CVPR22,
    title={Learn from others and be yourself in heterogeneous federated learning},
    author={Huang, Wenke and Ye, Mang and Du, Bo},
    booktitle={CVPR},
    year={2022}
}
@inproceedings{FSMAFL_ACMMM22,
  title={Few-Shot Model Agnostic Federated Learning},
  author={Huang, Wenke and Ye, Mang and Du, Bo and Gao, Xiang},
  booktitle={ACMMM},
  pages={7309--7316},
  year={2022}
}
@inproceedings{RHFL_CVPR22,
    title={Robust Federated Learning with Noisy and Heterogeneous Clients},
    author={Fang, Xiuwen and Ye, Mang},
    booktitle={CVPR},
    year={2022}
}
```
## Contact

This repository is currently maintained by [Wenke Huang](mailto:wenkehuang@whu.edu.cn).

