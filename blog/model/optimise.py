# Code to optimise the parameters of the model using Optuna.

import xgboost as xgb
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

class Objective(object):
    """
    Define an objective class to be able to pass parameters to the objective function.
    
    """
    def __init__(self, X, y):
        # Hold the X,y data.
        self.X = X
        self.y= y

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        param_grid = {
            "verbosity": [-1],
            "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=10),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.3),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
            }
        clf = lgb.LGBMClassifier(objective='binary', **param_grid)
        roc_auc_scores = cross_val_score(clf, self.X, self.y, cv=3,scoring='roc_auc')
        average_precision_scores = cross_val_score(clf, self.X, self.y, cv=3,scoring='average_precision')
        
        return roc_auc_scores.mean() , average_precision_scores.mean()
        


