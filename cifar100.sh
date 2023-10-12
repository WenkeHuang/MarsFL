'''label_skew'''
# -------------------------------------------------------------------------------------cifar1000-------------------------------------------------------------------------------------
#python main.py --task label_skew --dataset fl_cifar100 --method FedAVG --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedAVG --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedAVG --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedAVG --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_cifar100 --method FedProx --device_id 4 --csv_log --csv_name beta_1.0_mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01 DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProx --device_id 5 --csv_log --csv_name beta_0.3_mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01 DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProx --device_id 6 --csv_log --csv_name beta_0.5_mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01 DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProx --device_id 7 --csv_log --csv_name beta_0.1_mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01 DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_cifar100 --method FedProc --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProc --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProc --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProc --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

#python main.py --task label_skew --dataset fl_cifar100 --method FedProto --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProto --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProto --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedProto --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_cifar100 --method FedOpt --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedOpt --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedOpt --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedOpt --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

#python main.py --task label_skew --dataset fl_cifar100 --method FedDyn --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedDyn --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedDyn --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedDyn --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
##
#python main.py --task label_skew --dataset fl_cifar100 --method Scaffold --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method Scaffold --device_id 4 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method Scaffold --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method Scaffold --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
###
#python main.py --task label_skew --dataset fl_cifar100 --method MOON --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method MOON --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method MOON --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method MOON --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

#python main.py --task label_skew --dataset fl_cifar100 --method FPL --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FPL --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FPL --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FPL --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

#python main.py --task label_skew --dataset fl_cifar100 --method FedNova --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedNova --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedNova --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedNova --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_cifar100 --method FedNTD --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedNTD --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedNTD --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedNTD --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_cifar100 --method FedLC --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedLC --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedLC --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedLC --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_cifar100 --method FedRS --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedRS --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedRS --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_cifar100 --method FedRS --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &




