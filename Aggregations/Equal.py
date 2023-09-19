from Aggregations.utils.meta_aggregation import FederatedAggregation
import numpy as np

class Equal(FederatedAggregation):
    NAME = 'Equal'

    def __init__(self,args) -> None:
        super().__init__(args)

    def weight_calculate(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        freq =  [1/len(online_clients_list) for _ in range(len(online_clients_list))]
        return freq

