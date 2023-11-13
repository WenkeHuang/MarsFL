from collections import defaultdict
import sklearn.metrics.pairwise as smp
import numpy as np


def trimmed_mean(users_grads, users_count, corrupted_count):
    number_to_consider = int(users_grads.shape[0] - corrupted_count) - 1
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)

    for i, param_across_users in enumerate(users_grads.T):
        med = np.median(param_across_users)
        good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
        current_grads[i] = np.mean(good_vals) + med
    return current_grads


def bulyan(users_grads, users_count, corrupted_count):
    assert users_count >= 4 * corrupted_count + 3
    set_size = users_count - 2 * corrupted_count
    selection_set = []

    distances = _krum_create_distances(users_grads)
    while len(selection_set) < set_size:
        currently_selected = krum(users_grads, users_count - len(selection_set), corrupted_count, distances, True)
        selection_set.append(users_grads[currently_selected])

        # remove the selected from next iterations:
        distances.pop(currently_selected)
        for remaining_user in distances.keys():
            distances[remaining_user].pop(currently_selected)

    return trimmed_mean(np.array(selection_set), len(selection_set), 2 * corrupted_count)


def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
    return distances


def krum(users_grads, users_count, corrupted_count, distances=None, return_index=False):
    if not return_index:
        assert users_count >= 2 * corrupted_count + 1, ('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        return users_grads[minimal_error_index]


def multi_krum(users_grads, users_count, corrupted_count, n):
    non_malicious_count = users_count - corrupted_count
    # minimal_error = 1e20
    # minimal_error_index = -1
    all_error = []

    distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        all_error.append(current_error)
        # if current_error < minimal_error:
        #     minimal_error = current_error
        #     minimal_error_index = user
    all_error = np.array(all_error)
    sort_index = all_error.argsort()

    mean_users_grads = np.mean(users_grads[sort_index[:n]], axis=0)
    return mean_users_grads


def fools_gold(this_delta, summed_deltas, sig_features_idx, model, topk_prop=0, importance=False, importanceHard=False, clip=0):
    # Take all the features of sig_features_idx for each clients
    sd = summed_deltas.copy()
    sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)
    epsilon = 1e-5
    n = len(this_delta)
    # if importance or importanceHard:
    #     if importance:
    #         # smooth version of importance features
    #         importantFeatures = importanceFeatureMapLocal(model, topk_prop)
    #     if importanceHard:
    #         # hard version of important features
    #         importantFeatures = importanceFeatureHard(model, topk_prop)
    #     for i in range(n):
    #         sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)

    cs = smp.cosine_similarity(sig_filtered_deltas) - np.eye(n)
    # Pardoning: reweight by the max value seen
    maxcs = np.max(cs, axis=1) + epsilon
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    # if iter % 10 == 0 and iter != 0:
    #     print maxcs
    #     print wv

    # if clip != 0:
    #     # Augment onto krum
    #     scores = get_krum_scores(this_delta, n - clip)
    #     bad_idx = np.argpartition(scores, n - clip)[(n - clip):n]
    #
    #     # Filter out the highest krum scores
    #     wv[bad_idx] = 0

    # Apply the weight vector on this delta
    # delta = np.reshape(this_delta, (n, d))

    return np.dot(this_delta.T, wv)

def weighted_average_oracle(points, weights):
    tot_weights = np.sum(weights)
    weighted_updates = np.zeros_like(points[0])

    for w, p in zip(weights, points):
        weighted_updates += (w / tot_weights) * p

    return weighted_updates


def l2dist(p1, p2):
    """L2 distance between p1, p2, each of which is a list of nd-arrays"""
    # return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])
    return np.linalg.norm(p1 - p2)


def geometric_median_objective(median, points, alphas):
    """Compute geometric median objective."""
    return sum([alpha * l2dist(median, p) for alpha, p in zip(alphas, points)])


def geometric_median_update(points, alphas, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6):
    """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
    """
    alphas = np.asarray(alphas, dtype=points[0].dtype) / sum(alphas)
    median = weighted_average_oracle(points, alphas)
    num_oracle_calls = 1

    # logging
    obj_val = geometric_median_objective(median, points, alphas)
    logs = []
    log_entry = [0, obj_val, 0, 0]
    logs.append(log_entry)
    if verbose:
        print('Starting Weiszfeld algorithm')
        print(log_entry)

    # start
    for i in range(maxiter):
        prev_median, prev_obj_val = median, obj_val
        weights = np.asarray([alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alphas, points)],
                             dtype=alphas.dtype)
        weights = weights / weights.sum()
        median = weighted_average_oracle(points, weights)
        num_oracle_calls += 1
        obj_val = geometric_median_objective(median, points, alphas)
        log_entry = [i + 1, obj_val,
                     (prev_obj_val - obj_val) / obj_val,
                     l2dist(median, prev_median)]
        logs.append(log_entry)
        if verbose:
            print(log_entry)
        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break
    return median, num_oracle_calls, logs
