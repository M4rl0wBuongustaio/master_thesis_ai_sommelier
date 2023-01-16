import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


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


def get_n_predictions(input_user: int, similar_users: list, reviews: pd.DataFrame, threshold: float,
                      is_evaluation: bool, test_rev_df: pd.DataFrame):
    input_user_avg_rating = np.round(reviews[reviews.user_id == input_user].rating.mean(), decimals=1)
    input_user_rated_wines = np.unique(reviews[reviews.user_id == input_user].wine_id)
    wine_prediction = {}
    for sim_user in similar_users:
        user_avg_rating = reviews[reviews.user_id == sim_user].rating.mean()
        if is_evaluation:
            unrated_wines: pd.DataFrame = reviews.loc[
                (reviews.user_id == sim_user) & (~reviews.wine_id.isin(input_user_rated_wines)) & (
                    reviews.wine_id.isin(test_rev_df.loc[test_rev_df.user_id == input_user].wine_id)),
                ['user_id', 'wine_id', 'rating']
            ]
        else:
            unrated_wines: pd.DataFrame = reviews.loc[
                (reviews.user_id == sim_user) & (~reviews.wine_id.isin(input_user_rated_wines)), ['user_id',
                                                                                                  'wine_id',
                                                                                                  'rating']
            ]
        for wine in unrated_wines.wine_id:
            if wine not in wine_prediction:
                wine_prediction[wine] = round((input_user_avg_rating + (
                        unrated_wines.loc[unrated_wines.wine_id == wine].rating.iloc[0] - user_avg_rating)) * 2) / 2
    return {key: val for key, val in wine_prediction.items() if val >= threshold}


def evaluate_collaborative_filtering(df_list: list):
    df_test = df_list[0]
    df_train = df_list[1]
    input_users = df_train[df_train.user_id.isin(df_test.user_id)].user_id.unique()
    sim_matrix = get_sim_matrix(get_sparse_wine_user_matrix(df_train))
    unpredictable_users = []
    df = pd.DataFrame(data={}.items(), columns=['wine_id', 'rating', 'rating_predicted'])

    for user in input_users:
        try:
            sim_users = get_top_n_similar_users(n=10, sim_matrix=sim_matrix, input_user=user)
            preds: dict = get_n_predictions(input_user=user, similar_users=sim_users, reviews=df_train,
                                            threshold=3.0, is_evaluation=True, test_rev_df=df_test)
            df_temp = pd.DataFrame(preds.items(), columns=['wine_id', 'rating_predicted']).merge(
                df_test.loc[df_test.user_id == user, ['wine_id', 'rating']], on='wine_id', how='left')
            df = pd.concat([df, df_temp])
        except Exception as err:
            unpredictable_users.append(user)
            print('No wines could be predicted for user: ' + str(user) + ' (' + str(err) + ')')
            raise err
    return df
