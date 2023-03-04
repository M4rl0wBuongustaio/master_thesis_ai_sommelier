from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pandas as pd
import numpy as np


def get_sparse_wine_user_matrix(df: pd.DataFrame):
    return sparse.csr_matrix(
        (df.rating, (df.user_id, df.wine_id))
    )


def get_sim_matrix(matrix: sparse.csr_matrix):
    return cosine_similarity(matrix, dense_output=False)


def get_top_n_similar_users(n: int, sim_matrix: sparse.csr_matrix, input_user: int):
    users = sim_matrix[input_user, :].nonzero()[1]
    users = np.delete(users, np.where(users == input_user)[0])
    similar_users = {
        user: sim_matrix[input_user, user] for user in users
    }
    similar_users = {
        k: v for k, v in sorted(similar_users.items(), key=lambda item: item[1])
    }
    return list(similar_users.keys())[-n:][::-1]


def get_n_predictions(
        input_user: int,
        similar_users: list,
        reviews_train: pd.DataFrame,
        reviews_test: pd.DataFrame,
        threshold: float,
        is_evaluation: bool
):
    input_user_avg_rating = np.round(reviews_train[reviews_train.user_id == input_user].rating.mean(), decimals=1)
    input_user_rated_wines = np.unique(reviews_train[reviews_train.user_id == input_user].wine_id)
    wine_prediction = {}
    for sim_user in similar_users:
        user_avg_rating = reviews_train[reviews_train.user_id == sim_user].rating.mean()
        if is_evaluation:
            unrated_wines: pd.DataFrame = reviews_train.loc[
                (reviews_train.user_id == sim_user) & (~reviews_train.wine_id.isin(input_user_rated_wines)) &
                (reviews_train.wine_id.isin(reviews_test.loc[reviews_test.user_id == input_user].wine_id)),
                ['user_id', 'wine_id', 'rating']
            ]
        else:
            unrated_wines: pd.DataFrame = reviews_train.loc[
                (reviews_train.user_id == sim_user) & (~reviews_train.wine_id.isin(input_user_rated_wines)),
                ['user_id', 'wine_id', 'rating']
            ]
        for wine in unrated_wines.wine_id:
            if wine not in wine_prediction:
                wine_prediction[wine] = np.round((input_user_avg_rating + (
                        unrated_wines.loc[unrated_wines.wine_id == wine].rating.iloc[0] - user_avg_rating)), decimals=1)
    return {key: val for key, val in wine_prediction.items() if val >= threshold}
