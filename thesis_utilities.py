import pandas as pd
import sqlite3


def load_from_database(
        db: str,
        table: str,
        columns: str,
        with_condition: bool,
        condition: list
):
    df = pd.DataFrame()
    connection = sqlite3.connect('database/' + db + '.db')
    try:
        if with_condition:
            df = pd.read_sql_query(
                str('SELECT ' + columns + ' FROM ' + table + ' WHERE ' + condition[0] + ' ' + condition[1] + ' ' +
                    condition[2]), connection, index_col='index'
            )
        else:
            df = pd.read_sql_query(
                str('SELECT ' + columns + ' FROM ' + table), con=connection, index_col='index'
            )
    except Exception as e:
        raise e
    connection.close()
    return df


def save_to_database(
        db: str,
        table: str,
        df: pd.DataFrame
):
    connection = sqlite3.connect('database/' + db + '.db')
    try:
        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)
        if 'level_0' in df.columns:
            df.drop(columns='level_0', inplace=True)
        df.to_sql(name=table, con=connection, if_exists='replace')
    except Exception as e:
        raise e
    print('DataFrame has been saved successfully to: ' + db)
