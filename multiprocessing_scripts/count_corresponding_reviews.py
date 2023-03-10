import pandas as pd


def count_corresponding_reviews(args) -> pd.DataFrame:
    df_train = args[0]
    df_test = args[1]
    user_ids = args[2]

    id_list = list()
    reviews_count = list()
    corresponding_reviews_list = list()
    corresponding_reviews_evaluation_list = list()

    for user_id in user_ids:
        input_user_reviews = df_train[df_train['user_id'] == user_id]
        input_user_reviews_test = df_test[df_test['user_id'] == user_id]

        df_temp = df_train[df_train['wine_id'].isin(input_user_reviews['wine_id'])]

        corresponding_reviews_list.append(len(df_temp))

        df_temp = df_temp[df_temp['user_id'].isin(
            df_test[df_test['wine_id'].isin(input_user_reviews_test['wine_id'])]['user_id']
        )]

        id_list.append(user_id)
        corresponding_reviews_evaluation_list.append(len(df_temp))
        reviews_count.append(len(input_user_reviews) + len(input_user_reviews_test))

    data = pd.DataFrame({
        'id': id_list,
        'reviews_count': reviews_count,
        'corresponding_reviews': corresponding_reviews_list,
        'corresponding_reviews_evaluation': corresponding_reviews_evaluation_list
    })
    return data
