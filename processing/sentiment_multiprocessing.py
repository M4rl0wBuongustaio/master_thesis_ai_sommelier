from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch.multiprocessing import Pool, set_start_method
from datetime import datetime
import pandas as pd
import sqlite3

set_start_method("spawn", force=True)

model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

SA = None


def get_sa():
    global SA
    if SA is None:
        SA = pipeline(
            'sentiment-analysis',
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            model=AutoModelForSequenceClassification.from_pretrained(model_name),
            max_length=512,
            truncation=True
        )
    return SA


def get_sentiments(df):
    sa = get_sa()
    for i in df.index:
        result = sa(df['note'][i])
        if result[0]['label'] == 'negative':
            df.loc[i, 'sentiment'] = result[0]['score'] * -1
        else:
            df.loc[i, 'sentiment'] = result[0]['score']
    df = df.drop(columns='note')
    return df


def process(df):
    df.dropna(inplace=True, how='any')

    core_count = 10
    len_df_rev = int(len(df) / core_count)
    df_frames = [df.iloc[i * len_df_rev:(i + 1) * len_df_rev].copy() for i in range(core_count + 1)]

    start = datetime.now()
    multi_pool = Pool(processes=core_count)
    predictions = multi_pool.map(get_sentiments, df_frames)
    multi_pool.close()
    multi_pool.join()
    print(datetime.now() - start)
    return predictions
