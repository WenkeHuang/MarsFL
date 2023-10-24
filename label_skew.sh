# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
python main.py --task label_skew --dataset fl_fashionmnist --method FedDf --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedDf --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedDf --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_fashionmnist --method FedDf --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

python main.py --task label_skew --dataset fl_fashionmnist --method FcclPlus --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
python main.py --task label_skew --dataset fl_fashionmnist --method FcclPlus --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
python main.py --task label_skew --dataset fl_fashionmnist --method FcclPlus --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_fashionmnist --method FcclPlus --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

python main.py --task label_skew --dataset fl_fashionmnist --method RHFL --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
python main.py --task label_skew --dataset fl_fashionmnist --method RHFL --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
python main.py --task label_skew --dataset fl_fashionmnist --method RHFL --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
python main.py --task label_skew --dataset fl_fashionmnist --method RHFL --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#python main.py --task label_skew --dataset fl_mnist --method FedDf --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_mnist --method FedDf --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_mnist --method FedDf --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_mnist --method FedDf --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_mnist --method FcclPlus --device_id 4 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_mnist --method FcclPlus --device_id 5 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_mnist --method FcclPlus --device_id 6 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_mnist --method FcclPlus --device_id 7 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &
#
#python main.py --task label_skew --dataset fl_mnist --method RHFL --device_id 0 --csv_log --csv_name beta_1.0 --save_checkpoint DATASET.beta 1.0 &
#python main.py --task label_skew --dataset fl_mnist --method RHFL --device_id 1 --csv_log --csv_name beta_0.3 --save_checkpoint DATASET.beta 0.3 &
#python main.py --task label_skew --dataset fl_mnist --method RHFL --device_id 2 --csv_log --csv_name beta_0.5 --save_checkpoint DATASET.beta 0.5 &
#python main.py --task label_skew --dataset fl_mnist --method RHFL --device_id 3 --csv_log --csv_name beta_0.1 --save_checkpoint DATASET.beta 0.1 &

