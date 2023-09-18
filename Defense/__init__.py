import importlib
import inspect
import os


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Defense')
            if not model.find('__') > -1 and 'py' in model]


Defense_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('Defense.' + model)
    class_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'DefenseMethod' in str(
        inspect.getmro(getattr(mod, x))[1:])]
    for d in class_name:
        c = getattr(mod, d)
        Defense_NAMES[c.NAME] = c


def get_defense_method(args, cfg):
    return Defense_NAMES[args.defense](args, cfg)
