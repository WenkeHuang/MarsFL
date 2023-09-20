'''domain_skew'''
# Digits
#python main.py --task domain_skew --dataset Digits --method FedAVG --device_id 5 --csv_log --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FedProx --device_id 6 --csv_log --csv_name mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01&
#python main.py --task domain_skew --dataset Digits --method FedProc --device_id 7 --csv_log  --save_checkpoint &

python main.py --task domain_skew --dataset Digits --averaging Equal --method FedAVG --device_id 5 --csv_log --save_checkpoint &
python main.py --task domain_skew --dataset Digits --averaging Equal --method FedProx --device_id 6 --csv_log --csv_name mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01&
python main.py --task domain_skew --dataset Digits --averaging Equal --method FedProc --device_id 7 --csv_log  --save_checkpoint &

