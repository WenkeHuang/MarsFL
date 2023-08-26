import os
import inspect
import importlib
from Aggregations.utils.meta_aggregation import FederatedAggregation
from argparse import Namespace
def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Aggregations')
            if not model.find('__') > -1 and 'py' in model]

Aggregation_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('Aggregations.' + model)
    dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'FederatedAggregation' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        Aggregation_NAMES[c.NAME] = c

def get_fed_aggregation(args: Namespace) -> FederatedAggregation:
    assert args.averaging in Aggregation_NAMES.keys()
    return Aggregation_NAMES[args.averaging](args)
