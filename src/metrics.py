from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.special import kl_div as scipy_kl_div
from scipy.spatial.distance import cosine
import numpy as np


def kl_div(y_true, embeddings, symmetry=None, verbose=False):
    '''

    :param y_true:
    :param embeddings:
    :param symmetry: default=None, can be 'jeffreys' or 'jensen-shannon'
    :return:
    '''
    same_similarities = get_same_similarities(y_true, embeddings, verbose=verbose)
    false_similarities = get_false_similarities(y_true, embeddings, verbose=verbose)

    bins = np.linspace(-1, 1, 100)
    same_sim_distr, _ = np.histogram(same_similarities, bins, density=True)
    same_sim_distr += 1e-6
    false_sim_distr, _ = np.histogram(false_similarities, bins, density=True)
    false_sim_distr += 1e-6

    if symmetry == 'jeffreys':
        result = (scipy_kl_div(same_sim_distr, false_sim_distr) + scipy_kl_div(false_sim_distr, same_sim_distr)) / 2
    elif symmetry == 'jensen-shannon':
        m = (same_sim_distr + false_sim_distr) / 2
        result = (scipy_kl_div(same_sim_distr, m) + scipy_kl_div(false_sim_distr, m)) / 2
    else:
        result = scipy_kl_div(same_sim_distr, false_sim_distr)

    return sum(result)


def tpr_at_fpr(y_true, embeddings, fpr=0.01, verbose=False) -> float:
    same_similarities = get_same_similarities(y_true, embeddings, verbose=verbose)
    false_similarities = get_false_similarities(y_true, embeddings, verbose=verbose)

    threshold_similarity = np.quantile(false_similarities, q=1-fpr)
    tpr = sum(same_similarities >= threshold_similarity) / len(same_similarities)
    return tpr


def get_same_similarities(y_true, embeddings, verbose=False) -> np.ndarray:
    '''

    :param y_true: (N,)
    :param embeddings: (N, emb_size)
    :return: similarity scores of the same ids
    '''
    unique_ids = np.unique(y_true)
    same_similarities = []

    pbar = unique_ids
    if verbose:
        pbar = tqdm(pbar, desc=f"Getting same similarities")
    for id_ in pbar:
        current_embeddings = embeddings[y_true == id_, :]
        for e1 in current_embeddings:
            for e2 in current_embeddings:
                if e1 is not e2:
                    same_similarities.append(cosine(e1, e2))

    return np.array(same_similarities)


def get_false_similarities(y_true, embeddings, verbose=False) -> np.ndarray:
    '''

    :param y_true: (N,)
    :param embeddings: (N, emb_size)
    :return: similarity scores of the false ids
    '''
    false_similarities = []

    pbar = zip(y_true, embeddings)
    if verbose:
        pbar = tqdm(pbar, desc=f"Getting false similarities")
    for id1, e1 in pbar:
        for id2, e2 in zip(y_true, embeddings):
            if id1 != id2:
                false_similarities.append(cosine(e1, e2))

    return np.array(false_similarities)

