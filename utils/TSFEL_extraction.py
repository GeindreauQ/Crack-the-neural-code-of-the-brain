import pandas as pd
import tsfel

df = pd.concat([pd.read_csv('DATA/RAW/train.csv'),pd.read_csv('DATA/RAW/test.csv')])

cfg = tsfel.get_features_by_domain()


def compute_global_tsfel():
    res = tsfel.time_series_features_extractor(cfg, df.drop(['ID','neuron_id'],axis=1).diff().values,njobs=-1)
    res = res.loc[:,res.var()>0]
    res = res.rename(columns=lambda x:x.replace(' ','_'))
    res.to_csv('DATA/TSFEL/tsfel_aug_diff.csv')
