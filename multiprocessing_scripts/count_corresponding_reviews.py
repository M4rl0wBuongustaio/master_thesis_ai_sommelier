import pandas as pd


def count_corresponding_reviews(args) -> list:
    df_train = args[0]
    df_test = args[1]
    user_ids = args[2]

    corresponding_reviews_evaluation_list = list()
    median_note_length = list()

    for user_id in user_ids:
        input_user_reviews = df_train[df_train['user_id'] == user_id]
        input_user_reviews_test = df_test[df_test['user_id'] == user_id]

        df_temp = df_train[df_train['wine_id'].isin(input_user_reviews['wine_id'])]

        df_temp = df_temp[df_temp['user_id'].isin(
            df_test[df_test['wine_id'].isin(input_user_reviews_test['wine_id'])]['user_id']
        )]

        corresponding_reviews_evaluation_list.append(len(df_temp))
        median_note_length.append(df_temp['note_length'].median())

    data = pd.DataFrame({
        'corresponding_reviews_evaluation': corresponding_reviews_evaluation_list,
        'median_note_length_cr': median_note_length
    })
    return data
