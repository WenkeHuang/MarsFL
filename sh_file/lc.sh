'''label_skew'''
# -------------------------------------------------------------------------------------Fashion-------------------------------------------------------------------------------------
python main.py --task label_skew --dataset fl_fashionmnist --method FedLC --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedLC --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedLC --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedLC --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

python main.py --task label_skew --dataset fl_fashionmnist --method FedRS --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedRS --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedRS --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedRS --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

python main.py --task label_skew --dataset fl_cifar10 --method FedLC --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
python main.py --task label_skew --dataset fl_cifar10 --method FedLC --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
python main.py --task label_skew --dataset fl_cifar10 --method FedLC --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_cifar10 --method FedLC --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

python main.py --task label_skew --dataset fl_cifar10 --method FedRS --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
python main.py --task label_skew --dataset fl_cifar10 --method FedRS --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
python main.py --task label_skew --dataset fl_cifar10 --method FedRS --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_cifar10 --method FedRS --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &





