# Code to optimise the parameters of the model using Optuna.

import xgboost as xgb
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import numpy as np
from loguru import logger
import optuna.integration.lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from lightgbm import early_stopping
from lightgbm import log_evaluation

class MultiObjective(object):
    """
    Define an objective class to be able to pass parameters to the objective function.

    """

    def __init__(self, X, y):
        # Hold the X,y data.
        self.X = X
        self.y = y

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=10),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        }
        clf = lgb.LGBMClassifier(objective="binary", verbosity=-1, **param_grid)
        roc_auc_scores = cross_val_score(clf, self.X, self.y, cv=3, scoring="roc_auc")
        average_precision_scores = cross_val_score(
            clf, self.X, self.y, cv=3, scoring="average_precision"
        )

        return np.round(roc_auc_scores.mean(), 5), np.round(
            average_precision_scores.mean(), 5
        )


class ROCAUCObjective(object):
    """
    Define an objective class to be able to pass parameters to the objective function.

    """

    def __init__(self, X, y, monotone_constraints=None):
        # Hold the X,y data.
        self.X = X
        self.y = y
        self.monotone_constraints = monotone_constraints

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        if self.monotone_constraints is not None:
            # Create a parameter grid with monotonic constraints
            logger.info("Monotone constraints")
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=10),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "monotone_constraints": self.monotone_constraints,
            }
        else:
            logger.info("No monotone constraints")
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=10),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            }
        clf = lgb.LGBMClassifier(objective="binary", verbosity=-1, **param_grid)
        roc_auc_scores = cross_val_score(clf, self.X, self.y, cv=5, scoring="roc_auc")

        objective_value = roc_auc_scores.mean()

        """
        # Return the objective value.
        # Note that the higher the better.  
        # """
        if np.isnan(objective_value):
            return 0.5
        else:
            return objective_value


class PRAUCObjective(object):
    """
    Define an objective class to be able to pass parameters to the objective function.

    """

    def __init__(self, X, y):
        # Hold the X,y data.
        self.X = X
        self.y = y

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=10),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        }
        clf = lgb.LGBMClassifier(objective="binary", verbosity=-1, **param_grid)
        average_precision_scores = cross_val_score(
            clf, self.X, self.y, cv=3, scoring="average_precision"
        )

        return np.round(average_precision_scores.mean(), 5)


def getObjective(X, y, objective_type):
    """
    Return the objective function.

    """
    if objective_type == "multi":
        return MultiObjective(X, y)
    elif objective_type == "rocauc":
        return ROCAUCObjective(X, y)
    elif objective_type == "prauc":
        return PRAUCObjective(X, y)
    else:
        raise ValueError("Unknown objective type")
    
    
    
def tune_model(X, y, objective, n_trials=100, n_jobs=-1, monotone_constraints=None):
    """
    Tune the model using Optuna.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    objective : string
        The objective function to be used.
    n_trials : int, optional (default=100)
        The number of trials to run.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel.
    monotone_constraints : list, optional (default=None)
        The monotone constraints to be used.

    Returns
    -------
    best_params : dict
        The best parameters found.

    """
    
    objective = getObjective(X, y, objective)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    
    dtrain = lgb.Dataset(X, label=y)
    tuner = lgb.LightGBMTunerCV(
        params,
        dtrain,
        folds=StratifiedKFold(n_splits=3),
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    tuner.run()

    print("Best score:", tuner.best_score)
    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))
    
    return best_params