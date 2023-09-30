'''byzantine'''
'''
fl_cifar100 0.5 0.2 PairFlip
'''
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 0 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_MultiKrum --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method MultiKrumSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_Bulyan --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method BulyanSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_TrimmedMean --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method TrimmedMeanSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 3 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_FoolsGold --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method FoolsGoldSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_Dnc --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method DncSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 5 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_Rfa --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method RfaSever &


'''
fl_cifar100 0.5 0.4 PairFlip
'''
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_MultiKrum --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method MultiKrumSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 7 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_Bulyan --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method BulyanSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 0 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_TrimmedMean --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method TrimmedMeanSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_FoolsGold --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method FoolsGoldSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_Dnc --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method DncSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 3 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_Rfa --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method RfaSever &

'''
fl_cifar100 0.3 0.2 PairFlip
'''
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_MultiKrum --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method MultiKrumSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 5 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_Bulyan --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method BulyanSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_TrimmedMean --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method TrimmedMeanSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 7 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_FoolsGold --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method FoolsGoldSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 0 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_Dnc --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method DncSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_Rfa --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method RfaSever &


'''
fl_cifar100-10 0.3 0.4 PairFlip
'''
python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 2 \
       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_MultiKrum --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method MultiKrumSever &

python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 3 \
       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_Bulyan --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method BulyanSever &

python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 4 \
       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_TrimmedMean --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method TrimmedMeanSever &

python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 5 \
       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_FoolsGold --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method FoolsGoldSever &

python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 6 \
       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_Dnc --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method DncSever &

python main.py --task label_skew --attack_type byzantine --dataset fl_cifar100 --method FedProx --device_id 7 \
       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_Rfa --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method RfaSever &