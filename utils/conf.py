import numpy as np
import random
import torch
import os


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


def multi_domain_data_path() -> str:
    return '/data0/data_wk/Domain/'  # 140
    # return '/data/FL_data/Domain/' # 129


def single_domain_data_path() -> str:
    return '/data0/data_wk/'  # 140
    # return '/data/FL_data/Domain/' # 129

def log_path() -> str:
    return './data/'

def net_path() -> str:
    return './checkpoints/'


def config_path() -> str:
    return './Configs/'


def checkpoint_path() -> str:
    return './checkpoint/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
