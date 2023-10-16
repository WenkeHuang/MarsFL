'''OfficeCaltech'''
# caltech
#python main.py --task OOD --dataset OfficeCaltech --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedDC --device_id 3 --csv_log --save_checkpoint OOD.out_domain caltech &

# amazon
python main.py --task OOD --dataset OfficeCaltech --method FedNova --device_id 0 --csv_log --save_checkpoint OOD.out_domain amazon &
python main.py --task OOD --dataset OfficeCaltech --method FedProc --device_id 1 --csv_log --save_checkpoint OOD.out_domain amazon &
python main.py --task OOD --dataset OfficeCaltech --method FedOpt --device_id 3 --csv_log --save_checkpoint OOD.out_domain amazon &
python main.py --task OOD --dataset OfficeCaltech --method FedProto --device_id 4 --csv_log --save_checkpoint OOD.out_domain amazon &
python main.py --task OOD --dataset OfficeCaltech --method FedDC --device_id 5 --csv_log --save_checkpoint OOD.out_domain amazon &