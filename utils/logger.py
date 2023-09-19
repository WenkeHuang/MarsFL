import copy
import os
import csv
from utils.conf import log_path
from utils.utils import create_if_not_exists

import yaml
from yacs.config import CfgNode as CN

except_args = ['csv_log', 'csv_name', 'device_id', 'seed', 'tensorboard', 'conf_jobnum', 'conf_timestamp', 'conf_host', 'opts']


class CsvWriter:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.model_path = self.model_folder_path()
        self.para_path = self.write_para()
        print(self.para_path)

    def model_folder_path(self):
        if self.args.task == 'OOD':
            model_path = os.path.join(log_path(), self.args.task,self.args.attack_type, self.args.dataset, self.cfg.OOD.out_domain, self.args.averaging, self.args.method)
        else:
            model_path = os.path.join(log_path(), self.args.task,self.args.attack_type, self.args.dataset, self.args.averaging, self.args.method)
        create_if_not_exists(model_path)
        return model_path

    def write_weight(self, weight_dict, epoch_index, client_domain_list):
        weight_path = os.path.join(self.para_path, 'weight.csv')
        if epoch_index != 0:
            write_type = 'a'
        else:
            write_type = 'w'

        with open(weight_path, write_type) as result_file:
            result_file.write(str(epoch_index) + ':' + '\n')
            for i in range(len(client_domain_list)):
                result_file.write(client_domain_list[i] + ',')
            result_file.write('\n')
            for k in weight_dict:
                result_file.write(k + ':' + str(list(weight_dict[k])) + '\n')

    def write_acc(self, acc, name, mode='ALL'):
        if mode == 'ALL':
            acc_path = os.path.join(self.para_path, name + '_all_acc.csv')
            self.write_all_acc(acc_path, acc)
        elif mode == 'MEAN':
            mean_acc_path = os.path.join(self.para_path, name + '_mean_acc.csv')
            self.write_mean_acc(mean_acc_path, acc)

    def cfg_to_dict(self, cfg):
        d = {}
        for k, v in cfg.items():
            if isinstance(v, CN):
                d[k] = self.cfg_to_dict(v)
            else:
                d[k] = v
        return d

    def dict_to_cfg(self, d):
        cfg = CN()
        for k, v in d.items():
            if isinstance(v, dict):
                cfg[k] = self.dict_to_cfg(v)
            else:
                cfg[k] = v
        return cfg

    def write_para(self) -> None:
        from yacs.config import CfgNode as CN

        args = copy.deepcopy((self.args))  # 当前的args
        args = vars(args)
        cfg = copy.deepcopy(self.cfg)  # 当前的cfg

        for cc in except_args:
            if cc in args: del args[cc]  # 删掉当前args的无关元素
        for key, value in args.items():
            args[key] = str(value)
        paragroup_dirs = os.listdir(self.model_path)  # 获取所有存储的参数组
        n_para = len(paragroup_dirs)  # 获取长度
        final_check = False  # 假定不存在！
        # 判断是否参数一致
        for para in paragroup_dirs:
            exist_para_args = True  # 默认不存在对应的参数组 args!
            exist_para_cfg = True
            dict_from_csv = {}
            key_value_list = []
            para_path = os.path.join(self.model_path, para)
            args_path = para_path + '/args.csv'
            with open(args_path, mode='r') as inp:
                reader = csv.reader(inp)
                for rows in reader:
                    key_value_list.append(rows)
            for index, _ in enumerate(key_value_list[0]):
                dict_from_csv[key_value_list[0][index]] = key_value_list[1][index]
            if args != dict_from_csv:  # 如果对应不上
                exist_para_args = False  # 不存在对应的args！
            cfg_path = para_path + '/cfg.yaml'
            query_cfg = copy.deepcopy(cfg)
            query_cfg.merge_from_file(cfg_path)
            for name, value1 in cfg.items():
                if isinstance(value1, CN):
                    if name not in query_cfg or self.cfg_to_dict(query_cfg[name]) != self.cfg_to_dict(value1):
                        exist_para_cfg = False  # 存在不一样的
            if exist_para_args == True and exist_para_cfg == True:
                final_check = True
                break
        if final_check == False:
            '''
            定义存储的名字
            '''
            if self.args.csv_name == None:
                path = os.path.join(self.model_path, 'para' + str(n_para + 1))
                k = 1
                while os.path.exists(path):
                    path = os.path.join(self.model_path, 'para' + str(n_para + k))
                    k = k + 1
            else:
                path = os.path.join(self.model_path, self.args.csv_name)

            create_if_not_exists(path)
            columns = list(args.keys())
            write_headers = True
            args_path = path + '/args.csv'
            cfg_path = path + '/cfg.yaml'
            with open(args_path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)
            with open(cfg_path, 'w') as f:
                f.write(yaml.dump(self.cfg_to_dict(cfg)))
        else:
            path = para_path
        return path

    def write_mean_acc(self, mean_path, acc_list):
        if os.path.exists(mean_path):
            with open(mean_path, 'a') as result_file:
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
        else:
            with open(mean_path, 'w') as result_file:
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

    def write_all_acc(self, all_path, all_acc_list):
        if os.path.exists(all_path):
            with open(all_path, 'a') as result_file:
                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
        else:
            with open(all_path, 'w') as result_file:
                result_file.write('domain,')
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
