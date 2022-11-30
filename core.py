
import optuna
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.linear_model import RidgeClassifier,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from copy import deepcopy
optuna.logging.set_verbosity(optuna.logging.FATAL)


class Feature_generator(TransformerMixin):
    def __init__(self,estimator:BaseEstimator=Ridge(),max_iter:int=20,scoring=None,split:str='cv') -> None:
        super().__init__()
        self.estimator = estimator
        self.max_iter = max_iter
        self._ex = []
        self.scoring = scoring

    def get_new_ft(self,X,y):
        study = optuna.create_study()
        def to_opt(trial):
            available_features = X.columns.tolist()
            ft1 = trial.suggest_categorical('ft1',available_features)
            ft2 = trial.suggest_categorical('ft2',available_features)
            ft3 = trial.suggest_categorical('ft3',available_features)
            a = trial.suggest_float('a', -2,2)
            b = trial.suggest_float('b', -2,2)
            a2 = trial.suggest_float('a2', -2,2)
            b2 = trial.suggest_float('b2', -2,2)

            a3 = trial.suggest_float('a3', -2,2)
            b3 = trial.suggest_float('b3', -2,2)
            val = (a*X[ft1]).apply(lambda x: (x**b).real)*(a2*X[ft2]).apply(lambda x: (x**b2).real)*(a3*X[ft3]).apply(lambda x: (x**b3).real)
            temp = X.copy()
            temp['temp'] = StandardScaler().fit_transform(val.values.reshape(-1,1)).reshape(-1)
            try:
                res = -CVS(self.estimator,temp,y,scoring=self.scoring,cv=20).mean()
                return res
            except Exception as e:
                print(e)
                return np.inf
        study.optimize(to_opt, n_trials=50)
        return study.best_params
    def func(self,X,a,b,a2,b2,a3,b3,ft1,ft2,ft3,scaler=None):
        val = (a*X[ft1]).apply(lambda x: (x**b).real)*(a2*X[ft2]).apply(lambda x: (x**b2).real)*(a3*X[ft3]).apply(lambda x: (x**b3).real)
        if not scaler:
            scaler = StandardScaler()
            val = scaler.fit_transform(val.values.reshape(-1,1)).reshape(-1)
            
        else:
            val = scaler.transform(val.values.reshape(-1,1)).reshape(-1)
        return val,scaler
        
    def fit_transform(self,X,y):
        X=X.copy()
        for count in tqdm(range(self.max_iter)):
            hyperparam = self.get_new_ft(X,y)
            val,scaler = self.func(X,**hyperparam)
            ft_name = f"{hyperparam['ft1']}and{hyperparam['ft2']}and{hyperparam['ft3']}"
            X[ft_name] = val
            dico = deepcopy(hyperparam)
            self._ex.append((ft_name,deepcopy(scaler),dico))
        return X

    def transform(self,X):
        X=X.copy()
        for ft_name,scaler,dico in self._ex:
            val,_ = self.func(X,scaler=scaler,**dico)
            X[ft_name]=val
        return X
