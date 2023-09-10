import importlib
import inspect
import os


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Datasets/public_dataset')
            if not model.find('__') > -1 and 'py' in model]


Pub_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('Datasets.public_dataset.' + model)
    dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'PublicDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        Pub_NAMES[c.NAME] = c


def get_public_dataset(args, cfg, **kwargs):
    public_dataset_name = kwargs['public_dataset_name']
    assert public_dataset_name in Pub_NAMES.keys()
    return Pub_NAMES[public_dataset_name](args, cfg, **kwargs)
