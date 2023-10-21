'''OfficeCaltech'''
# caltech
#python main.py --task OOD --dataset OfficeCaltech --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedDC --device_id 3 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method COPADA --device_id 3 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain caltech &
#python main.py --task OOD --dataset OfficeCaltech --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain caltech &
#wait

# amazon
#python main.py --task OOD --dataset OfficeCaltech --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method FedProc --device_id 3 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method FedOpt --device_id 5 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method FedProto --device_id 6 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method FedDC --device_id 7 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method COPADA --device_id 3 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset OfficeCaltech --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain amazon &
#wait

# webcam
#python main.py --task OOD --dataset OfficeCaltech --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method FedProc --device_id 3 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method FedOpt --device_id 5 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method FedProto --device_id 6 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method FedDC --device_id 7 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method COPADA --device_id 3 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset OfficeCaltech --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain webcam &
#wait
# dslr
#python main.py --task OOD --dataset OfficeCaltech --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset OfficeCaltech --method FedProc --device_id 3 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset OfficeCaltech --method FedOpt --device_id 5 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset OfficeCaltech --method FedProto --device_id 6 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset OfficeCaltech --method FedDC --device_id 7 --csv_log --save_checkpoint OOD.out_domain dslr &

#python main.py --task OOD --dataset OfficeCaltech --method COPADA --device_id 3 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset OfficeCaltech --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset OfficeCaltech --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset OfficeCaltech --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain dslr &

'''Digits'''
# MNIST
#python main.py --task OOD --dataset Digits --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain MNIST &
#python main.py --task OOD --dataset Digits --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain MNIST &
#python main.py --task OOD --dataset Digits --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain MNIST &
#python main.py --task OOD --dataset Digits --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain MNIST &

python main.py --task OOD --dataset Digits --method COPADA --device_id 1 --csv_log --save_checkpoint OOD.out_domain MNIST &
python main.py --task OOD --dataset Digits --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain MNIST &
python main.py --task OOD --dataset Digits --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain MNIST &
python main.py --task OOD --dataset Digits --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain MNIST &


# USPS
#python main.py --task OOD --dataset Digits --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain USPS &
#python main.py --task OOD --dataset Digits --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain USPS &
#python main.py --task OOD --dataset Digits --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain USPS &
#python main.py --task OOD --dataset Digits --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain USPS &

python main.py --task OOD --dataset Digits --method COPADA --device_id 1 --csv_log --save_checkpoint OOD.out_domain USPS &
python main.py --task OOD --dataset Digits --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain USPS &
python main.py --task OOD --dataset Digits --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain USPS &
python main.py --task OOD --dataset Digits --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain USPS &


# SVHN
#python main.py --task OOD --dataset Digits --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain SVHN &
#python main.py --task OOD --dataset Digits --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain SVHN &
#python main.py --task OOD --dataset Digits --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain SVHN &
#python main.py --task OOD --dataset Digits --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain SVHN &

python main.py --task OOD --dataset Digits --method COPADA --device_id 1 --csv_log --save_checkpoint OOD.out_domain SVHN &
python main.py --task OOD --dataset Digits --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain SVHN &
python main.py --task OOD --dataset Digits --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain SVHN &
python main.py --task OOD --dataset Digits --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain SVHN &

# SYN
#python main.py --task OOD --dataset Digits --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain SYN &
#python main.py --task OOD --dataset Digits --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain SYN &
#python main.py --task OOD --dataset Digits --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain SYN &
#python main.py --task OOD --dataset Digits --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain SYN &

python main.py --task OOD --dataset Digits --method COPADA --device_id 1 --csv_log --save_checkpoint OOD.out_domain SYN &
python main.py --task OOD --dataset Digits --method KD3A --device_id 5 --csv_log --save_checkpoint OOD.out_domain SYN &
python main.py --task OOD --dataset Digits --method COPADG --device_id 6 --csv_log --save_checkpoint OOD.out_domain SYN &
python main.py --task OOD --dataset Digits --method FedProxGA --device_id 7 --csv_log --save_checkpoint OOD.out_domain SYN &


'''PACS'''
## photo
#python main.py --task OOD --dataset PACS --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain photo &
#python main.py --task OOD --dataset PACS --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain photo &
#python main.py --task OOD --dataset PACS --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain photo &
#python main.py --task OOD --dataset PACS --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain photo &
#
#wait
#
## art_painting
#python main.py --task OOD --dataset PACS --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain art_painting &
#python main.py --task OOD --dataset PACS --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain art_painting &
#python main.py --task OOD --dataset PACS --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain art_painting &
#python main.py --task OOD --dataset PACS --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain art_painting &
#
#wait
#
## cartoon
#python main.py --task OOD --dataset PACS --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain cartoon &
#python main.py --task OOD --dataset PACS --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain cartoon &
#python main.py --task OOD --dataset PACS --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain cartoon &
#python main.py --task OOD --dataset PACS --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain cartoon &
#
#wait
#
## sketch
#python main.py --task OOD --dataset PACS --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain sketch &
#python main.py --task OOD --dataset PACS --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain sketch &
#python main.py --task OOD --dataset PACS --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain sketch &
#python main.py --task OOD --dataset PACS --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain sketch &

'''Office31'''
# amazon dslr webcam
#python main.py --task OOD --dataset Office31 --method FedNova --device_id 0 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset Office31 --method FedProc --device_id 1 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset Office31 --method FedOpt --device_id 2 --csv_log --save_checkpoint OOD.out_domain amazon &
#python main.py --task OOD --dataset Office31 --method FedProto --device_id 3 --csv_log --save_checkpoint OOD.out_domain amazon &

# dslr
#python main.py --task OOD --dataset Office31 --method FedNova --device_id 4 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset Office31 --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset Office31 --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain dslr &
#python main.py --task OOD --dataset Office31 --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain dslr &
#wait
# webcam
#python main.py --task OOD --dataset Office31 --method FedNova --device_id 3 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset Office31 --method FedProc --device_id 5 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset Office31 --method FedOpt --device_id 6 --csv_log --save_checkpoint OOD.out_domain webcam &
#python main.py --task OOD --dataset Office31 --method FedProto --device_id 7 --csv_log --save_checkpoint OOD.out_domain webcam &

