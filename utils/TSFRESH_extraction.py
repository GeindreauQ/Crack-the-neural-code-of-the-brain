import numpy as np
import pandas as pd
from tsfresh import extract_features

def compute(df,set_):
    filtre = lambda x:'time' in x
    df[list(filter(filtre,df.columns))] = df[list(filter(filtre,df.columns))].diff(axis=1)
    df.dropna(axis=1,inplace=True)
    df_long = pd.wide_to_long(df,stubnames='timestamp_',i='ID',j='ep').reset_index().rename(columns={'timestamp_':'timestamp'}).drop('neuron_id',axis=1)
    aug_df = extract_features(df_long, column_id='ID', column_sort='ep',n_jobs=5)
    aug = aug_df.loc[:,aug_df.var()>0]
    aug = aug.loc[:,(aug_df.isna().sum()<len(aug_df)*0.5)]
    aug.to_csv(f'DATA/TSFRESH/tsfresh_aug_{set_}.csv')
    return


def compute_global_tsfresh():
    compute(pd.read_csv('DATA/RAW/train.csv'),"train")
    compute(pd.read_csv('DATA/RAW/test.csv'),"test")
    pd.concat(
        [pd.read_csv(f'DATA/TSFRESH/tsfresh_aug_train.csv',index_col=0),
         pd.read_csv(f'DATA/TSFRESH/tsfresh_aug_test.csv',index_col=0)],axis=0).rename(
        columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).to_csv('DATA/TSFRESH/tsfresh_aug.csv')
    return