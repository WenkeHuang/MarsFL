python main.py --task label_skew --dataset fl_cifar10 --method Scaffold --device_id 5 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_mnist --method Scaffold --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_fashionmnist --method Scaffold --device_id 7 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_cifar100 --method Scaffold --device_id 4 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &

