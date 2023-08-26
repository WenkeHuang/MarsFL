import importlib
import inspect
import os


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Local')
            if not model.find('__') > -1 and 'py' in model]


local_names = {}
for model in get_all_models():
    mod = importlib.import_module('Local.' + model)
    class_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'LocalMethod' in str(
        inspect.getmro(getattr(mod, x))[1:])]
    for d in class_name:
        c = getattr(mod, d)
        local_names[c.NAME] = c


def get_local_method(args, cfg):
    return local_names[cfg[args.method].local_method](args, cfg)
