from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
import numpy as np


def tpr_at_fpr(y_true, embeddings, fpr=0.01) -> float:
    same_similarities = get_same_similarities(y_true, embeddings)
    false_similarities = get_false_similarities(y_true, embeddings)

    threshold_similarity = np.quantile(false_similarities, q=1-fpr)
    tpr = sum(same_similarities >= threshold_similarity) / len(same_similarities)
    return tpr


def get_same_similarities(y_true, embeddings) -> np.ndarray:
    '''

    :param y_true: (N,)
    :param embeddings: (N, emb_size)
    :return: similarity scores of the same ids
    '''
    unique_ids = np.unique(y_true)
    same_similarities = []
    for id_ in tqdm(unique_ids, desc=f"Getting same similarities"):
        current_embeddings = embeddings[y_true == id_, :]
        for e1 in current_embeddings:
            for e2 in current_embeddings:
                if e1 is not e2:
                    same_similarities.append(cosine(e1, e2))

    return np.array(same_similarities)


def get_false_similarities(y_true, embeddings) -> np.ndarray:
    '''

    :param y_true: (N,)
    :param embeddings: (N, emb_size)
    :return: similarity scores of the false ids
    '''
    false_similarities = []
    for id1, e1 in tqdm(zip(y_true, embeddings), desc=f"Getting false similarities"):
        for id2, e2 in zip(y_true, embeddings):
            if id1 != id2:
                false_similarities.append(cosine(e1, e2))

    return np.array(false_similarities)


