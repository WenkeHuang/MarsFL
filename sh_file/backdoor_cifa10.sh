
'''
Cifar-10 0.5 0.2 base_backdoor
'''
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 7 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_MultiKrum --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method MultiKrumSever &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_Bulyan --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method BulyanSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 6 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_TrimmedMean --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method TrimmedMeanSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 7 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_FoolsGold --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method FoolsGoldSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 4 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_Dnc --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method DncSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_Rfa --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method RfaSever &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method CRFL --device_id 0 \
#       --csv_log --csv_name beta_0.5_bcr_0.2 --save_checkpoint \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method RLR --device_id 4 \
#       --csv_log --csv_name beta_0.5_bcr_0.2 --save_checkpoint \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 &

'''
Cifar-10 0.3 0.2 base_backdoor
'''
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 7 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_MultiKrum --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method MultiKrumSever &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_Bulyan --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method BulyanSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_TrimmedMean --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method TrimmedMeanSever &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_FoolsGold --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method FoolsGoldSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 6 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_Dnc --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method DncSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 7 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_Rfa --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 FedAVG.global_method RfaSever &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method CRFL --device_id 2 \
#       --csv_log --csv_name beta_0.3_bcr_0.2 --save_checkpoint \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 &
#
python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method RLR --device_id 4 \
       --csv_log --csv_name beta_0.3_bcr_0.2 --save_checkpoint \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils base_backdoor attack.bad_client_rate 0.2 &

#'''
#Cifar-10 0.5 0.2 semantic_backdoor
#'''
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 4 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_MultiKrum --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method MultiKrumSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_Bulyan --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method BulyanSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 6 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_TrimmedMean --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method TrimmedMeanSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 7 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_FoolsGold --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method FoolsGoldSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 4 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_Dnc --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method DncSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.5_bcr_0.2_Rfa --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method RfaSever &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method CRFL --device_id 3 \
#       --csv_log --csv_name beta_0.5_bcr_0.2 --save_checkpoint \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method RLR --device_id 3 \
#       --csv_log --csv_name beta_0.5_bcr_0.2 --save_checkpoint \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 &

#'''
#Cifar-10 0.3 0.2 semantic_backdoor
#'''
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 6 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_MultiKrum --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method MultiKrumSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 7 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_Bulyan --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method BulyanSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 4 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_TrimmedMean --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method TrimmedMeanSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 5 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_FoolsGold --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method FoolsGoldSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 6 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_Dnc --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method DncSever &
#
#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method FedAVG --device_id 7 \
#       --csv_log --csv_name beta_0.3_bcr_0.2_Rfa --save_checkpoint  \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 FedAVG.global_method RfaSever &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method CRFL --device_id 4 \
#       --csv_log --csv_name beta_0.3_bcr_0.2 --save_checkpoint \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 &

#python main.py --task label_skew --attack_type backdoor --dataset fl_cifar10 --method RLR --device_id 4 \
#       --csv_log --csv_name beta_0.3_bcr_0.2 --save_checkpoint \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.backdoor.evils semantic_backdoor attack.bad_client_rate 0.2 &

