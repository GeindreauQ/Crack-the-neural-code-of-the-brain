import featuretools as ft
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('DATA/TSFEL/tsfel_aug.csv')
df = df.loc[:,df.var()>0]
df['neuron_id'] = pd.read_csv('DATA/RAW/train.csv').neuron_id.tolist()+pd.read_csv('DATA/RAW/test.csv').neuron_id.tolist()
df['ID'] = df.index



if __name__=='__main__':
    es = ft.EntitySet(id = 'raw')
    es.add_dataframe(dataframe_name="raw",
                    dataframe=df,
                    index="ID",)
    
    es.normalize_dataframe(base_dataframe_name='raw',
                           new_dataframe_name='neuron',
                           index='neuron_id',)
    
    trans_features =['modulo_by_feature',
             'percentile',
             'natural_logarithm',
             'cosine',
             'square_root',
             'modulo_by_feature',
             'sine',]
    
    
    
    feature_matrix, feature_names = ft.dfs(
        entityset = es,
        target_dataframe_name = 'raw',
        agg_primitives = ['max','skew','mean','min','std',],
        trans_primitives =trans_features,
        max_depth = 2,
        verbose=True,
        n_jobs=1,)
    feature_matrix.to_csv('DATA/TSFEL/features_tools_aug.csv')
