from argparse import Namespace
from abc import abstractmethod


class FederatedAggregation:
    """
    Federated Aggregation
    """
    NAME = None

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @abstractmethod
    def weight_calculate(self, **kwargs):
        pass

    def agg_parts(self, **kwargs):
        freq = kwargs['freq']
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        global_net = kwargs['global_net']
        global_w = {}
        except_part = kwargs['except_part']  # 不进行聚合的部分
        global_only = kwargs['global_only']  # 是否只分发给global而不分发local

        # 是否使用额外的结构
        use_additional_net = False
        additional_net_list = None
        additional_freq = None
        if 'use_additional_net' in kwargs:
            use_additional_net = kwargs['use_additional_net']
            additional_net_list = kwargs['additional_net_list']
            additional_freq = kwargs['additional_freq']

        first = True
        for index, net_id in enumerate(online_clients_list):
            net = nets_list[net_id]  # 获取 online client 中对应的网络的索引
            net_para = net.state_dict()

            # 排除所有不用的的部分
            used_net_para = {}
            for k, v in net_para.items():
                is_in = False
                for part_str_index in range(len(except_part)):
                    if except_part[part_str_index] in k:
                        is_in = True
                        break
                # 只有不在的排除范围内的 才选中
                if not is_in:
                    used_net_para[k] = v
            # 只加载需要的参数
            if first:
                first = False
                for key in used_net_para:
                    global_w[key] = used_net_para[key] * freq[index]
            else:
                for key in used_net_para:
                    global_w[key] += used_net_para[key] * freq[index]

        if use_additional_net:
            for index, _ in enumerate(additional_net_list):
                net = additional_net_list[index]  # 获取 online client 中对应的网络的索引
                net_para = net.state_dict()

                # 排除所有不用的的部分
                used_net_para = {}
                for k, v in net_para.items():
                    is_in = False
                    for part_str_index in range(len(except_part)):
                        if except_part[part_str_index] in k:
                            is_in = True
                            break
                    # 只有不在的排除范围内的 才选中
                    if not is_in:
                        used_net_para[k] = v
                # 只加载需要的参数

                for key in used_net_para:
                    global_w[key] += used_net_para[key] * additional_freq[index]

        # 分发local
        if not global_only:
            for net in nets_list:
                net.load_state_dict(global_w, strict=False)

        # 更新global
        global_net.load_state_dict(global_w, strict=False)
        return

    # def aggregation_weight(self, **kwargs):
    #     freq = kwargs['freq']
    #     online_clients_list = kwargs['online_clients_list']
    #     nets_list = kwargs['nets_list']
    #     global_w = kwargs['global_net'].state_dict()
    # 
    #     first = True
    #     for index, net_id in enumerate(online_clients_list):
    #         net = nets_list[net_id]
    #         net_para = net.state_dict()
    # 
    #         if first:
    #             first = False
    #             for key in net_para:
    #                 global_w[key] = net_para[key] * freq[index]
    #         else:
    #             for key in net_para:
    #                 global_w[key] += net_para[key] * freq[index]
    #     return global_w
