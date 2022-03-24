"""
Code to evaluate the model.
"""
from yellowbrick.classifier import DiscriminationThreshold
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def show_model_results(data, model,calc_threshold=False):
    """
    Show the model results.
    Args :
        data(dict) : Dictionary containing the training and testing data.
        model(xgboost.XGBClassifier) : Model used to train and evaluate data.
        metrics(dict): Metrics used to evaluate the model.Default is roc_auc.
    Returns :
        model(xgboost.XGBClassifier) : Return the trained model.
    """
    # Todo : Add cross validation.

    model.fit(data["xtrain"], data["ytrain"])

    y_train_proba = model.predict_proba(data["xtrain"])[:, 1]
    y_test_proba = model.predict_proba(data["xtest"])[:, 1]
    y_test_pred = model.predict(data["xtest"])

    print(f"Train ROC-AUC score : {roc_auc_score(data['ytrain'],y_train_proba)}")
    print(f"Test ROC-AUC score : {roc_auc_score(data['ytest'],y_test_proba)}")
    print(f"Test PR AUC socre : {average_precision_score(data['ytest'],y_test_proba)}")
    # Add other metrics as required.

    # The outputs are probabilites, however we would like to work with predictions.
    # Hence, lets convert the probas to predictions.
    if calc_threshold:
        visualizer = DiscriminationThreshold(model, quantiles=np.array([0.25, 0.5, 0.75]),exclude=['queue_rate'])
        visualizer.fit(data["xtrain"], data["ytrain"])
        visualizer.show()

    return model
