'''OfficeCaltech'''
#python main.py --task OOD --dataset OfficeCaltech --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain caltech
#python main.py --task OOD --dataset OfficeCaltech --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain amazon
#python main.py --task OOD --dataset OfficeCaltech --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain webcam
#python main.py --task OOD --dataset OfficeCaltech --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain dslr

#python main.py --task OOD --dataset Digits --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain MNIST
#python main.py --task OOD --dataset Digits --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain USPS
#python main.py --task OOD --dataset Digits --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain SVHN
#python main.py --task OOD --dataset Digits --method Scaffold --device_id 4 --csv_log --save_checkpoint OOD.out_domain SYN

python main.py --task OOD --dataset PACS --method Scaffold --device_id 1 --csv_log --save_checkpoint OOD.out_domain photo &
python main.py --task OOD --dataset PACS --method Scaffold --device_id 1 --csv_log --save_checkpoint OOD.out_domain art_painting &
python main.py --task OOD --dataset PACS --method Scaffold --device_id 2 --csv_log --save_checkpoint OOD.out_domain cartoon &
python main.py --task OOD --dataset PACS --method Scaffold --device_id 2 --csv_log --save_checkpoint OOD.out_domain sketch &

python main.py --task OOD --dataset Office31 --method Scaffold --device_id 5 --csv_log --save_checkpoint OOD.out_domain amazon &
python main.py --task OOD --dataset Office31 --method Scaffold --device_id 6 --csv_log --save_checkpoint OOD.out_domain webcam &
python main.py --task OOD --dataset Office31 --method Scaffold --device_id 7 --csv_log --save_checkpoint OOD.out_domain dslr &
