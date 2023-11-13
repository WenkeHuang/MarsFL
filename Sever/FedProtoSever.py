from Sever.utils.sever_methods import SeverMethod


class FedProtoSever(SeverMethod):
    NAME = 'FedProtoSever'

    def __init__(self, args, cfg):
        super(FedProtoSever, self).__init__(args, cfg)

    def proto_aggregation(self, online_clients, local_protos_dict):
        agg_protos_label = dict()
        for idx in online_clients:
            local_protos = local_protos_dict[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        local_protos = kwargs['local_protos']

        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)
        global_protos = self.proto_aggregation(online_clients_list, local_protos)
        return freq, global_protos
