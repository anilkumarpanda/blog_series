"""
Code to evaluate the model.
"""

from yellowbrick.classifier import DiscriminationThreshold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from blog.utils.utils import get_key


def show_model_results(data, model, feature_names=None, calc_rocauc=True):
    """
    Show the model results.
    Args :
        data(dict) : Dictionary containing the training and testing data.
        model(xgboost.XGBClassifier) : Model used to train and evaluate data.
        metrics(dict): Metrics used to evaluate the model.Default is roc_auc.
    Returns :
        model(xgboost.XGBClassifier) : Return the trained model.
    """
    # Add cross validation.
    # roc_auc = cross_val_score(
    #         model, data["xtrain"], data["ytrain"], cv=3, scoring="roc_auc"
    #     )
    # print(f'Cross Validation ROC AUC Score : {roc_auc.mean()}')
    # Add feature names is not none.
    # Add other metrics as required.

    if feature_names is not None:
        model.fit(data["xtrain"], data["ytrain"], feature_names=feature_names)
    else:
        model.fit(data["xtrain"], data["ytrain"])

    # Show the model results.
    if calc_rocauc:
        y_train_proba = model.predict_proba(data["xtrain"])[:, 1]
        y_test_proba = model.predict_proba(data["xtest"])[:, 1]
        print(f"Train ROC-AUC score : {roc_auc_score(data['ytrain'],y_train_proba)}")
        print(f"Test ROC-AUC score : {roc_auc_score(data['ytest'],y_test_proba)}")
        print(
            f"Test PR AUC socre : {average_precision_score(data['ytest'],y_test_proba)}"
        )

    # Caluclate the Predictions.
    y_test_pred = model.predict(data["xtest"])
    # Create a list of threshold values with a step size of 0.1
    theshold_list = np.arange(0.01, 0.99, 0.1)
    proba_dict = {}
    # convert probabilities to prediction based on threshold
    for threshold in theshold_list:
        predictions = np.where(y_test_pred > threshold, 1, 0)
        balanced_accuracy = balanced_accuracy_score(data["ytest"], predictions)
        proba_dict[threshold] = balanced_accuracy

    # Find the maximum value in the dictionary proab_dict
    max_accuracy = max(proba_dict.values())
    # Find key for a value in dictionary
    threshold = get_key(proba_dict, max_accuracy)
    print(
        f"Best balanced accouracy of {max_accuracy} obtained at threshold {threshold} "
    )

    return model

    # Get the key from dict, with a given value.
