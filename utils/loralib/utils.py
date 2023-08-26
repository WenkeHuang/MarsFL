
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer
import copy


# def unmerge_model(model:nn.Module):
#     '''
#     分离所有Lora参数
#     '''
#     for m in model.modules():
#         if isinstance(m, LoRALayer):
#             m.merge_weight(False)

def change_model_mode(mode, model: nn.Module):
    if mode not in ('lora', 'bone'):
        raise ValueError('not valid tarin part !')
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.change_mode(mode)


def reset_lora_parameter(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.reset_lora_prameters()


def fix_bone_and_lora(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = False


def mark_train_part(mode, model: nn.Module):
    '''
    根据mode选择训练的部分

    mode in ('lora','bone')
    '''
    if mode not in ('lora', 'bone'):
        raise ValueError('not valid tarin part !')

    # 训练lora，固定backbone
    if mode == 'lora':
        mark_only_lora_as_trainable(model)
    else:
        # 训练backbone，固定lora
        mark_lora_fixed(model)


def mark_lora_fixed(model: nn.Module):
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            # 这里需要保证设置为True
            p.requires_grad = True
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def all_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    my_state_dict = copy.deepcopy(my_state_dict)
    return my_state_dict


# def bone_state_dict(model: nn.Module, use_copy=True) -> Dict[str, torch.Tensor]:
#     my_state_dict = model.state_dict()
#     if use_copy:
#         my_state_dict = copy.deepcopy(my_state_dict)
#     my_state_dict = {k: v for k, v in my_state_dict.items() if 'lora_' not in k}
#     return my_state_dict


def lora_prameters(model: nn.Module):
    return {n: p for n, p in model.named_parameters() if 'lora_' in n}


# def bone_parameters(model: nn.Module):
#     return {n: p for n, p in model.named_parameters() if 'lora_' not in n}


# 只找到backbone 不要lora和分类器
def backbone_parameters(model: nn.Module):
    return {n: p for n, p in model.named_parameters() if 'lora_' not in n and 'fc_class' not in n}

def backbone_state_dict(model: nn.Module, use_copy=True) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if use_copy:
        my_state_dict = copy.deepcopy(my_state_dict)
    my_state_dict = {k: v for k, v in my_state_dict.items() if 'lora_' not in k and 'fc_class' not in k}
    return my_state_dict

# 只要lora和分类器
def lora_and_fc_state_dict(model: nn.Module, use_copy=True) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if use_copy:
        my_state_dict = copy.deepcopy(my_state_dict)
    my_state_dict = {k: v for k, v in my_state_dict.items() if 'lora_' in k or 'fc_class' in k}
    return my_state_dict


def lora_state_dict(model: nn.Module, bias: str = 'none', use_copy=True) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if use_copy:
        my_state_dict = copy.deepcopy(my_state_dict)
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
