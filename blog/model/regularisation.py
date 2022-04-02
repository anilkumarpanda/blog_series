"""
Code for model regularisation.
"""

# Code to tune the models
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from probatus.feature_elimination import EarlyStoppingShapRFECV
from yellowbrick.classifier import DiscriminationThreshold
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from loguru import logger


def select_features(data, n_features, verbose=1):
    """
    Method for feature selection.
    Args:
        data(dict) : Dictionary containing the training and testing data.
        n_features (int): Number of features to return.
    """
    # Simple feature selection strategy to ensure that the features used in the model are good.

    clf = XGBClassifier(
        max_depth=3, use_label_encoder=False, objective="binary:logistic"
    )
    fs_param_grid = {
        "n_estimators": [5, 7, 10, 15],
        "num_leaves": [3, 5, 7, 10],
    }
    search = RandomizedSearchCV(clf, fs_param_grid)

    # Run feature elimination
    shap_elimination = EarlyStoppingShapRFECV(
        clf=search,
        step=0.2,
        cv=3,
        scoring="roc_auc",
        early_stopping_rounds=5,
        n_jobs=4,
        verbose=verbose,
    )

    report = shap_elimination.fit_compute(data["xtrain"], data["ytrain"])
    # Select the best features based on validation score.
    if n_features is None:
        selected_feats = report[["features_set"]].head(1).values[0]
    else:
        selected_feats = shap_elimination.get_reduced_features_set(
            num_features=n_features
        )
    logger.info(f"Selected features : {selected_feats} ")
    return selected_feats, shap_elimination.plot()


def get_monotone_constraints(data_dict, target, corr_threshold=0.1):
    """
    Method to get monotone constraints.

    Args:
        data_dict(dict) : Dictionary containing the training and testing data.
        target(str) : Target variable.
        corr_threshold(float) : Correlation threshold.
        Returns:
            monotone_constraints(dict) : Dictionary containing the monotone constraints.
    """
    data = data_dict["xtrain"].copy()
    data[target] = data_dict["ytrain"]

    corr = pd.Series(data.corr(method="spearman")[target]).drop(target)
    monotone_constraints = tuple(
        np.where(corr < -corr_threshold, -1, np.where(corr > corr_threshold, 1, 0))
    )
    return monotone_constraints
