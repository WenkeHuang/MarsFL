import os
import inspect
import importlib
from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Datasets/federated_dataset/single_domain')
            if not model.find('__') > -1 and 'py' in model]


single_domain_dataset_name = {}
for model in get_all_models():
    mod = importlib.import_module('Datasets.federated_dataset.single_domain.' + model)
    dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'SingleDomainDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        single_domain_dataset_name[c.NAME] = c


def get_single_domain_dataset(args, cfg) -> SingleDomainDataset:
    assert args.dataset in single_domain_dataset_name.keys()
    return single_domain_dataset_name[args.dataset](args, cfg)
