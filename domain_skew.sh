'''Digits'''
#python main.py --task domain_skew --dataset Digits --method FedAVG --device_id 0 --csv_log --save_checkpoint &
#python main.py --task domain_skew --dataset Digits --method FedProx --device_id 1 --csv_log --csv_name mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01&
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
python main.py --task domain_skew --dataset OfficeCaltech --method FedAVG --device_id 4 --csv_log --save_checkpoint &
python main.py --task domain_skew --dataset OfficeCaltech --method FedProx --device_id 6 --csv_log --csv_name mu_0.01 --save_checkpoint Local.FedProxLocal.mu 0.01&
python main.py --task domain_skew --dataset OfficeCaltech --method Scaffold --device_id 7 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method MOON --device_id 7 --csv_name mu_1 --csv_log  --save_checkpoint Local.MOONLocal.mu 1&

#python main.py --task domain_skew --dat  aset OfficeCaltech --method FedProc --device_id  --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method MOON --device_id 5 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method FedDyn --device_id 6 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method FedOpt --device_id 7 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method FedProto --device_id 0 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method FPL --device_id 1 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method FedNTD --device_id 2 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method FedNova --device_id 3 --csv_log  --save_checkpoint &