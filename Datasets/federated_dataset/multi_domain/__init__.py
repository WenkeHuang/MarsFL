import os
import inspect
import importlib
from Datasets.federated_dataset.multi_domain.utils.multi_domain_dataset import MultiDomainDataset


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Datasets/federated_dataset/multi_domain')
            if not model.find('__') > -1 and 'py' in model]


Priv_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('Datasets.federated_dataset.multi_domain.' + model)
    dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'MultiDomainDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        Priv_NAMES[c.NAME] = c


def get_multi_domain_dataset(args, cfg) -> MultiDomainDataset:
    assert args.dataset in Priv_NAMES.keys()
    return Priv_NAMES[args.dataset](args, cfg)
