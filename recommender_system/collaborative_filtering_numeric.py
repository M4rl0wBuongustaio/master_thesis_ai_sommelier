from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pandas as pd
import numpy as np


def get_sparse_wine_user_matrix(review_pool: pd.DataFrame):
    return sparse.csr_matrix(
        (review_pool.rating, (review_pool.user_id, review_pool.wine_id))
    )


def get_sim_matrix(matrix: sparse.csr_matrix):
    return cosine_similarity(matrix, dense_output=False)


def get_n_similar_user(review_pool: pd.DataFrame, n: int, truncate: bool, input_user_id: int):
    matrix = get_sparse_wine_user_matrix(review_pool=review_pool)
    sim_matrix = get_sim_matrix(matrix=matrix)
    candidates: np.ndarray = sim_matrix[input_user_id, :].nonzero()[1]
    candidates: np.ndarray = np.delete(candidates, np.where(candidates == input_user_id)[0])

    if truncate and len(candidates) > n:
        candidates = np.random.RandomState(26).choice(candidates, size=n)

    similar_users = {
        candidate: sim_matrix[input_user_id, candidate] for candidate in candidates
    }
    similar_users = pd.DataFrame({'user_id': similar_users.keys(), 'similarity': similar_users.values()})
    similar_users.sort_values(by='similarity', ascending=False, inplace=True)
    similar_users.drop(columns='similarity', inplace=True)
    return similar_users.head(n=n)
