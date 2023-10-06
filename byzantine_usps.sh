'''byzantine'''

'''
fl_usps 0.5 0.2 PairFlip
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &


'''
fl_usps 0.5 0.4 PairFlip
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &

'''
fl_usps 0.3 0.2 PairFlip
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &


'''
fl_usps-10 0.3 0.4 PairFlip
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils PairFlip attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &

'''
fl_usps 0.5 0.2 SymFlip
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils SymFlip attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils SymFlip attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &


'''
fl_usps 0.5 0.4 SymFlip
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils SymFlip attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils SymFlip attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &

'''
fl_usps 0.3 0.2 SymFlip
'''
python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 0 \
       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_FoolsGoldSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils SymFlip attack.bad_client_rate 0.2 FedProx.global_method FoolsGoldSever &


#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils SymFlip attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils SymFlip attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &
#

'''
fl_usps-10 0.3 0.4 SymFlip
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils SymFlip attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils SymFlip attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &
#


'''
fl_usps 0.5 0.2 RandomNoise
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &


'''
fl_usps 0.5 0.4 RandomNoise
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &

'''
fl_usps 0.3 0.2 RandomNoise
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &


'''
fl_usps-10 0.3 0.4 RandomNoise
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils RandomNoise attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &



'''
fl_usps 0.5 0.2 min_sum
'''
python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 0 \
       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_BulyanSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils min_sum attack.bad_client_rate 0.2 FedProx.global_method BulyanSever &


#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils min_sum attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils min_sum attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &


'''
fl_usps 0.5 0.4 min_sum
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils min_sum attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.5_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.5 attack.byzantine.evils min_sum attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &

'''
fl_usps 0.3 0.2 min_sum
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 1 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils min_sum attack.bad_client_rate 0.2 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 2 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.2_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils min_sum attack.bad_client_rate 0.2 FedProx.global_method SageFlowSever &


'''
fl_usps-10 0.3 0.4 min_sum
'''

#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 4 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_FLTrustSever --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils min_sum attack.bad_client_rate 0.4 FedProx.global_method FLTrustSever &
#
#python main.py --task label_skew --attack_type byzantine --dataset fl_usps --method FedProx --device_id 6 \
#       --csv_log --csv_name beta_0.3_mu_0.01_bcr_0.4_SageFlow --save_checkpoint Local.FedProxLocal.mu 0.01 \
#       DATASET.parti_num 10 DATASET.beta 0.3 attack.byzantine.evils min_sum attack.bad_client_rate 0.4 FedProx.global_method SageFlowSever &

