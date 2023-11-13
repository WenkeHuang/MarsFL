import importlib
import inspect
import os

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Methods')
            if not model.find('__') > -1 and 'py' in model]

Fed_Methods_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('Methods.' + model)
    class_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'FederatedMethod' in str(
        inspect.getmro(getattr(mod, x))[1:])]
    for d in class_name:
        c = getattr(mod, d)
        Fed_Methods_NAMES[c.NAME] = c

def get_fed_method(nets_list, client_domain_list, args, cfg):
    return Fed_Methods_NAMES[args.method](nets_list, client_domain_list, args, cfg)
