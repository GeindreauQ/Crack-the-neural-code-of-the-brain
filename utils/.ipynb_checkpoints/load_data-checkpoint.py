import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def transform(df1,diff=1,to_add=None):
    raw = df1.drop(['neuron_id'],axis=1).diff(diff,axis=1).dropna(axis=1)
    new = pd.DataFrame()
    for func in [pd.Series.min,pd.Series.max,pd.Series.kurt,pd.Series.skew,pd.Series.std,pd.Series.sum]:
        new[func.__name__]=raw.apply(func,axis=1)
    for i in range(1,20):
        new['Q'+str(i)]=raw.quantile(0.05*i,axis=1)
    features_basics = new.columns.tolist()
    new = new.join(df1.neuron_id).join(new.join(df1.neuron_id).groupby('neuron_id').mean(),on='neuron_id',rsuffix='batch')
    new.loc[:,list(map(lambda x: f'n{x}batch',features_basics))] = new[features_basics].values/ new.loc[:,list(map(lambda x: f'{x}batch',features_basics))].values
    
    new = new.join(new.groupby('neuron_id').count()['min'],on = 'neuron_id',rsuffix='count')
    
    return new.drop('neuron_id',axis=1)

def l_raw():
    import random

    X = pd.read_csv('DATA/RAW/train.csv')
    X['TARGET'] = pd.read_csv('DATA/RAW/target.csv').TARGET
    
    neurons_ids = list(set(X.neuron_id))
    random.shuffle(neurons_ids)
    X=X.drop('ID',axis=1)
    
    train = X[X.neuron_id.isin(neurons_ids[:-40])]
    test  = X[X.neuron_id.isin(neurons_ids[-40:])]
    
    return train.drop('TARGET',axis=1),train.TARGET,test.drop('TARGET',axis=1),test.TARGET