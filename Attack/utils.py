import copy
from numpy.testing import assert_array_almost_equal
import numpy as np
import torch
from utils.utils import row_into_parameters

# attack攻击类型
attack_type_dict = {
    'PairFlip': 'dataset',
    'SymFlip': 'dataset',
    'RandomNoise': 'model_para',
    'lie_attack': 'model_para',
    'min_max': 'model_para',
    'min_sum': 'model_para',
}


def multiclass_noisify(y, P):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState()

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise


def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0):
    if noise_type == 'PairFlip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, nb_classes=nb_classes)
    elif noise_type == 'SymFlip':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, nb_classes=nb_classes)
    elif noise_type is None:
        train_noisy_labels = train_labels
        actual_noise_rate = 0
    return train_noisy_labels, actual_noise_rate


# 数据集攻击
def attack_dataset(args, cfg, private_dataset, client_type):
    # 攻击类型是数据集攻击 那么修改数据集的内容
    if attack_type_dict[cfg[args.task].evils] == 'dataset':
        for i in range(len(client_type)):
            if not client_type[i]:
                dataset = private_dataset.train_loaders[i].dataset
                train_labels = np.asarray([[dataset.targets[i]] for i in range(len(dataset.targets))])
                train_noisy_labels, actual_noise_rate = noisify(
                    train_labels=train_labels,
                    noise_type=cfg[args.task].evils,
                    noise_rate=cfg[args.task].noise_data_rate,
                    nb_classes=len(np.unique(train_labels)))

                train_noisy_labels = train_noisy_labels.reshape(-1)
                dataset.targets = train_noisy_labels


# 网络参数攻击
def attack_net_para(args, cfg, fed_method):
    temp_net = copy.deepcopy(fed_method.global_net)
    if cfg[args.task].evils == 'RandomNoise':
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                random_net = copy.deepcopy(fed_method.random_net)
                fed_method.nets_list[i] = random_net

    elif cfg[args.task].evils == 'AddNoise':
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                random_net = copy.deepcopy(fed_method.random_net)
                noise_weight = 0.5
                for name, param in sele_net.state_dict().items():
                    param += torch.tensor(copy.deepcopy(noise_weight * (random_net.state_dict()[name] - param)), dtype=param.dtype)

    elif cfg[args.task].evils == 'lie_attack':
        # 计算z的值
        n = len(fed_method.online_clients_list)
        m = n - sum(fed_method.client_type)
        s = n // 2 + 1 - m
        z = np.exp(-1 * (s ** 2)) / (2 / ((2 * np.pi) ** 0.5))

        all_net_delta = []
        with torch.no_grad():
            for i in fed_method.online_clients_list:
                if fed_method.client_type[i] == True:
                    sele_net = fed_method.nets_list[i]

                    net_delta = []
                    for name, param0 in temp_net.state_dict().items():
                        param1 = sele_net.state_dict()[name]
                        delta = (param1.detach() - param0.detach())
                        net_delta.append(copy.deepcopy(delta.view(-1)))
                    net_delta = torch.cat(net_delta, dim=0).view(1, -1)

                    all_net_delta.append(net_delta)
            all_net_delta = torch.cat(all_net_delta, dim=0)
            avg = torch.mean(all_net_delta, dim=0)
            std = torch.std(all_net_delta, dim=0)
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                row_into_parameters((avg + z * std).cpu().numpy(), sele_net.parameters())

    elif cfg[args.task].evils == 'min_sum':

        # 计算好客户端的模型变化量和对应均值
        all_net_delta = []
        with torch.no_grad():
            for i in fed_method.online_clients_list:
                if fed_method.client_type[i] == True:
                    sele_net = fed_method.nets_list[i]

                    net_delta = []
                    for name, param0 in temp_net.state_dict().items():
                        param1 = sele_net.state_dict()[name]
                        delta = (param1.detach() - param0.detach())
                        net_delta.append(copy.deepcopy(delta.view(-1)))
                    net_delta = torch.cat(net_delta, dim=0).view(1, -1)

                    all_net_delta.append(net_delta)
            all_net_delta = torch.cat(all_net_delta, dim=0)
            avg_delta = torch.mean(all_net_delta, dim=0)

        if cfg[args.task].dev_type == 'unit_vec':
            deviation = avg_delta / torch.norm(avg_delta)  # unit vector, dir opp to good dir
        elif cfg[args.task].dev_type == 'sign':
            deviation = torch.sign(avg_delta)
        elif cfg[args.task].dev_type == 'std':
            deviation = torch.std(all_net_delta, 0)

        lamda_fail = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda_succ = 0

        distances = []
        for update in all_net_delta:
            distance = torch.norm((all_net_delta - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > cfg[args.task].threshold_diff:
            mal_update = (avg_delta - lamda * deviation)
            distance = torch.norm((all_net_delta - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (avg_delta - lamda_succ * deviation)
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                row_into_parameters(mal_update.cpu().numpy(), sele_net.parameters())

    elif cfg[args.task].evils == 'min_max':
        # 计算好客户端的模型变化量和对应均值
        all_net_delta = []
        with torch.no_grad():
            for i in fed_method.online_clients_list:
                if fed_method.client_type[i] == True:
                    sele_net = fed_method.nets_list[i]

                    net_delta = []
                    for name, param0 in temp_net.state_dict().items():
                        param1 = sele_net.state_dict()[name]
                        delta = (param1.detach() - param0.detach())
                        net_delta.append(copy.deepcopy(delta.view(-1)))
                    net_delta = torch.cat(net_delta, dim=0).view(1, -1)

                    all_net_delta.append(net_delta)
            all_net_delta = torch.cat(all_net_delta, dim=0)
            avg_delta = torch.mean(all_net_delta, dim=0)

        if cfg[args.task].dev_type == 'unit_vec':
            deviation = avg_delta / torch.norm(avg_delta)  # unit vector, dir opp to good dir
        elif cfg[args.task].dev_type == 'sign':
            deviation = torch.sign(avg_delta)
        elif cfg[args.task].dev_type == 'std':
            deviation = torch.std(all_net_delta, 0)

        lamda_fail = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda_succ = 0

        distances = []
        for update in all_net_delta:
            distance = torch.norm((all_net_delta - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > cfg[args.task].threshold_diff:
            mal_update = (avg_delta - lamda * deviation)
            distance = torch.norm((all_net_delta - mal_update), dim=1) ** 2
            max_d = torch.max(distance)

            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (avg_delta - lamda_succ * deviation)
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                row_into_parameters(mal_update.cpu().numpy(), sele_net.parameters())
