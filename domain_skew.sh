'''Digits'''
#python main.py --task domain_skew --dataset Digits \
#  --method FedAVG --device_id 0 --csv_log --csv_name 4_0.001 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 &
#
#python main.py --task domain_skew --dataset Digits \
#  --method FedProx --device_id 1 --csv_log --csv_name 4_0.001_mu_0.01 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset Digits \
#  --method FedAVG --device_id 2 --csv_log --csv_name 4_0.005 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 &
#
#python main.py --task domain_skew --dataset Digits \
#  --method FedProx --device_id 3 --csv_log --csv_name 4_0.005_mu_0.01 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&
#wait
#python main.py --task domain_skew --dataset Digits \
#  --method FedAVG --device_id 0 --csv_log --csv_name 12_0.001 --save_checkpoint DATASET.parti_num 12 OPTIMIZER.local_train_lr 0.001 &
#
#python main.py --task domain_skew --dataset Digits \
#  --method FedProx --device_id 1 --csv_log --csv_name 12_0.001_mu_0.01 --save_checkpoint DATASET.parti_num 12 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset Digits \
#  --method FedAVG --device_id 2 --csv_log --csv_name 12_0.005 --save_checkpoint DATASET.parti_num 12 OPTIMIZER.local_train_lr 0.005 &
#
#python main.py --task domain_skew --dataset Digits \
#  --method FedProx --device_id 3 --csv_log --csv_name 12_0.005_mu_0.01 --save_checkpoint DATASET.parti_num 12 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&


#python main.py --task domain_skew --dataset Digits --method Scaffold --device_id 2 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method MOON --device_id 3 --csv_name mu_1 --csv_log  --save_checkpoint Local.MOONLocal.mu 1&
#python main.py --task domain_skew --dat  aset Digits --method FedProc --device_id 3 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method MOON --device_id 1 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FedDyn --device_id 2 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FedOpt --device_id 6 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FedProto --device_id 4 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FPL --device_id 2 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FedNTD --device_id 6 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FedNova --device_id 7 --csv_log  --save_checkpoint &

'''OfficeCaltech'''
#python main.py --task domain_skew --dataset OfficeCaltech \
#  --method FedAVG --device_id 0 --csv_log --csv_name 4_0.001 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 &
#
#python main.py --task domain_skew --dataset OfficeCaltech \
#  --method FedProx --device_id 1 --csv_log --csv_name 4_0.001_mu_0.01 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset OfficeCaltech \
#  --method FedProx --device_id 2 --csv_log --csv_name 4_0.001_mu_0.001 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.001&
#
#python main.py --task domain_skew --dataset OfficeCaltech \
#  --method FedAVG --device_id 3 --csv_log --csv_name 4_0.005 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 &
#
#python main.py --task domain_skew --dataset OfficeCaltech \
#  --method FedProx --device_id 4 --csv_log --csv_name 4_0.005_mu_0.01 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset OfficeCaltech \
#  --method FedProx --device_id 5 --csv_log --csv_name 4_0.005_mu_0.001 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.001&

'''OfficeCaltech'''

python main.py --task domain_skew --dataset OfficeCaltech \
  --method FedProx --device_id 0 --csv_log --csv_name 4_0.001_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.5 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&

python main.py --task domain_skew --dataset OfficeCaltech \
  --method FedProx --device_id 1 --csv_log --csv_name 4_0.005_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.5 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&

python main.py --task domain_skew --dataset OfficeCaltech \
  --method FedProx --device_id 2 --csv_log --csv_name 4_0.001_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&

python main.py --task domain_skew --dataset OfficeCaltech \
  --method FedProx --device_id 3 --csv_log --csv_name 4_0.005_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&

'''pacs'''
python main.py --task domain_skew --dataset PACS \
  --method FedProx --device_id 4 --csv_log --csv_name 4_0.001_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.5 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&

python main.py --task domain_skew --dataset PACS \
  --method FedProx --device_id 5 --csv_log --csv_name 4_0.005_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.5 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&

python main.py --task domain_skew --dataset PACS \
  --method FedProx --device_id 6 --csv_log --csv_name 4_0.001_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&

python main.py --task domain_skew --dataset PACS \
  --method FedProx --device_id 7 --csv_log --csv_name 4_0.005_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&
