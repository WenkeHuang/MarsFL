from Aggregations.utils.meta_aggregation import FederatedAggregation
import numpy as np

class Weight(FederatedAggregation):
    NAME = 'Weight'

    def __init__(self,args) -> None:
        super().__init__(args)

    def weight_calculate(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']

        online_clients_dl = [priloader_list[online_clients_index] for online_clients_index in online_clients_list]
        online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
        online_clients_all = np.sum(online_clients_len)
        freq = online_clients_len / online_clients_all
        return freq

