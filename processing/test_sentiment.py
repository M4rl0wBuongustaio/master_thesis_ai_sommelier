from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import concurrent.futures
import pandas as pd
import sqlite3
import os

SA = None


def get_sa():
    if SA is not None:
        return SA
    sa = pipeline(
        'sentiment-analysis',
        tokenizer=AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest'),
        model=AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    )
    return sa


def get_sentiments(df):
    sa = get_sa()
    for i in df.index:
        result = sa(df['note'][i])
        if result[0]['label'] == 'negative':
            df.loc[i, 'sentiment'] = result[0]['score'] * -1
        else:
            df.loc[i, 'sentiment'] = result[0]['score']
    return df.drop(columns='note', inplace=True)


'''
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context("fork"))):
    Process = NoDaemonProcess
'''

if __name__ == "__main__":
    con_rev = sqlite3.connect('../database/review_en.db')
    df_rev = pd.read_sql_query("SELECT id, note from review_en", con_rev)
    df_rev.note.dropna(inplace=True)
    df_rev = df_rev.head(n=1000)

    start = datetime.now()
    df_rev.note.dropna(inplace=True)
    df_results = pd.DataFrame()
    core_count = 10
    len_df_rev = int(len(df_rev) / core_count)
    df_frames = [df_rev.iloc[i * len_df_rev:(i + 1) * len_df_rev].copy() for i in range(core_count + 1)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(get_sentiments, df_frames[i]) for i in range(core_count)]

        for result in concurrent.futures.as_completed(results):
            df_results = pd.concat([df_results, result.result()])

    print(datetime.now() - start)
    print(df_results)
