# Code to optimise the parameters of the model using Optuna.

import xgboost as xgb
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import numpy as np
from loguru import logger


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
                "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "monotone_constraints": self.monotone_constraints,
            }
        else:
            logger.info("No monotone constraints")
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=10),
                "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            }
        clf = lgb.LGBMClassifier(objective="binary", verbosity=-1, **param_grid)
        roc_auc_scores = cross_val_score(clf, self.X, self.y, cv=5, scoring="roc_auc")

        objective_value = np.round(roc_auc_scores.mean(), 5)

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
