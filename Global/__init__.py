import importlib
import inspect
import os


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Global')
            if not model.find('__') > -1 and 'py' in model]


global_names = {}
for model in get_all_models():
    mod = importlib.import_module('Global.' + model)
    class_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'GlobalMethod' in str(
        inspect.getmro(getattr(mod, x))[1:])]
    for d in class_name:
        c = getattr(mod, d)
        global_names[c.NAME] = c


def get_global_method(args, cfg):
    return global_names[cfg[args.method].global_method](args, cfg)
