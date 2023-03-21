import collaborative_filtering_numeric
import collaborative_filtering_textual
import pandas as pd
import numpy as np


def get_predictions(
        review_pool: pd.DataFrame,
        input_user_id: int,
        similar_user_id: int,
        target_wines: list = None
) -> pd.DataFrame:
    input_user_avg_rating = np.round(review_pool[review_pool['user_id'] == input_user_id]['rating'].mean(), decimals=1)
    sim_user_avg_rating = np.round(review_pool[review_pool['user_id'] == similar_user_id]['rating'].mean(), decimals=1)
    predictions_list = list()
    wines_list = list()
    input_user_list = list()

    if target_wines is None:
        target_wines: list = review_pool[
            (review_pool['user_id'] == similar_user_id) &
            (review_pool['user_id'] != input_user_id)
            ]['wine_id'].tolist()

    for target_wine in target_wines:
        sim_user_rating: float = review_pool[
            (review_pool['user_id'] == similar_user_id) & (review_pool['wine_id'] == target_wine)]['rating'].values[0]
        prediction: float = (input_user_avg_rating + (sim_user_rating - sim_user_avg_rating))

        predictions_list.append(prediction)
        wines_list.append(target_wine)
        input_user_list.append(input_user_id)
    return pd.DataFrame({'user_id': input_user_list, 'wine_id': wines_list, 'prediction': predictions_list})


def evaluate_recommender(
        args: list
) -> pd.DataFrame:
    df_train: pd.DataFrame = args[0]
    df_test: pd.DataFrame = args[1]
    input_user_list = args[2]
    n_user: int = args[3]
    type_name: str = args[4]
    is_evaluation: bool = args[5]
    truncate: bool = args[6]
    if len(args) > 7:
        model_path = args[7]

    df_results = pd.DataFrame()

    for input_user_id in input_user_list:

        input_user_reviews = df_train[df_train['user_id'] == input_user_id]
        input_user_reviews_test = df_test[df_test['user_id'] == input_user_id]

        if is_evaluation:
            # Ensure user can be evaluated against input user.
            review_pool: pd.DataFrame = df_train[df_train['wine_id'].isin(input_user_reviews['wine_id'])]

            # Reduce to input-user-reviews in training data.
            review_pool: pd.DataFrame = review_pool[review_pool['user_id'].isin(
                df_test[df_test['wine_id'].isin(input_user_reviews_test['wine_id'])]['user_id']
            )]
            if review_pool.empty:
                continue
        else:
            review_pool = df_train[
                df_train['wine_id'].isin(input_user_reviews['wine_id'])
            ]

        if type_name == 'nlp':
            similar_user: pd.DataFrame = collaborative_filtering_textual.get_n_similar_user(
                input_user_id=input_user_id, review_pool=review_pool, embedder_path=model_path,
                truncate=truncate, n=n_user
            )
        elif type_name == 'numeric':
            similar_user: pd.DataFrame = collaborative_filtering_numeric.get_n_similar_user(
                review_pool=review_pool, n=n_user, truncate=truncate, input_user_id=input_user_id
            )
        else:
            raise ValueError('Please set a valid recommender type (nlp / numeric)!')

        for similar_user_id in similar_user['user_id']:
            similar_user_reviews: pd.DataFrame = pd.concat(
                [df_test[df_test['user_id'] == similar_user_id], df_train[df_train['user_id'] == similar_user_id]])
            target_wines = list(
                set(similar_user_reviews['wine_id']) &
                set(input_user_reviews_test['wine_id'])
            )
            review_pool = pd.concat([similar_user_reviews, input_user_reviews])
            df_predictions = get_predictions(
                review_pool=review_pool, input_user_id=input_user_id, similar_user_id=similar_user_id,
                target_wines=target_wines
            )
            df_results = pd.concat([df_results, df_predictions])

    df_results = df_results.merge(df_test[['user_id', 'wine_id', 'rating']], on=['user_id', 'wine_id'])
    return df_results
