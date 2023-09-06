

'''label_skew'''
python main.py --task label_skew --dataset fl_cifar10 --method FedAVG --device_id 0 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_cifar10 --method FedAVG --device_id 1 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
python main.py --task label_skew --dataset fl_cifar10 --method FedAVG --device_id 2 --csv_log --csv_name beta_0.01 --save_checkpoint DATASET.beta 0.01 &

python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 3 --csv_log --csv_name beta_0.5_mu_0.01 --save_checkpoint FedProx.mu 0.01 DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 4 --csv_log --csv_name beta_0.1_mu_0.01 --save_checkpoint FedProx.mu 0.01 DATASET.beta 0.1 &
python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 5 --csv_log --csv_name beta_0.01_mu_0.01 --save_checkpoint FedProx.mu 0.01 DATASET.beta 0.01 &

python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 6 --csv_log --csv_name beta_0.5_mu_0.1 --save_checkpoint FedProx.mu 0.1 DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 7 --csv_log --csv_name beta_0.1_mu_0.1 --save_checkpoint FedProx.mu 0.1 DATASET.beta 0.1 &
python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 0 --csv_log --csv_name beta_0.01_mu_0.1 --save_checkpoint FedProx.mu 0.1 DATASET.beta 0.01 &

python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 1 --csv_log --csv_name beta_0.5_mu_0.05 --save_checkpoint FedProx.mu 0.05 DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 2 --csv_log --csv_name beta_0.1_mu_0.05 --save_checkpoint FedProx.mu 0.05 DATASET.beta 0.1 &
python main.py --task label_skew --dataset fl_cifar10 --method FedProx --device_id 3 --csv_log --csv_name beta_0.01_mu_0.05 --save_checkpoint FedProx.mu 0.05 DATASET.beta 0.01 &


