"""
Code to evaluate the model.
"""

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from blog.utils.utils import get_key
from loguru import logger
import seaborn as sns
import matplotlib.pylab as plt

def show_model_results(data, model, feature_names=None, calc_rocauc=True, **kwargs):
    """
    Show the model results.
    Args :
        data(dict) : Dictionary containing the training and testing data.
        model(xgboost.XGBClassifier) : Model used to train and evaluate data.
        metrics(dict): Metrics used to evaluate the model.Default is roc_auc.

    Returns :
        model(xgboost.XGBClassifier) : Return the trained model.
    """
    
    if calc_rocauc:
        #Add cross validation.
        roc_auc = cross_val_score(
                model, data["xtrain"], data["ytrain"], cv=3, scoring="roc_auc"
            )
        print(f'Cross Validation ROC AUC Score : {roc_auc.mean()}')
    
    # Add feature names is not none.
    # Add other metrics as required.

    if feature_names is not None:
        model.fit(data["xtrain"], data["ytrain"], feature_names=feature_names)
    else:
        model.fit(data["xtrain"], data["ytrain"], **kwargs)

    # Show the model results.
    if calc_rocauc:
        # Add cross validation.
        # roc_auc = cross_val_score(
        #         model, data["xtrain"], data["ytrain"], cv=3, scoring="roc_auc"
        #     )
        # print(f'Cross Validation ROC AUC Score : {roc_auc.mean()}')
        # Add feature names is not none.
        # Add other metrics as required.
        y_train_proba = model.predict_proba(data["xtrain"])[:, 1]
        y_test_proba = model.predict_proba(data["xtest"])[:, 1]
        print(f"Train ROC-AUC score : {roc_auc_score(data['ytrain'],y_train_proba)}")
        print(f"Test ROC-AUC score : {roc_auc_score(data['ytest'],y_test_proba)}")
        print(
            f"Test PR AUC socre : {average_precision_score(data['ytest'],y_test_proba)}"
        )
        
        theshold_list = np.arange(0.01, 0.99, 0.10)
        proba_dict = {}
        # convert probabilities to prediction based on threshold
        for threshold in theshold_list:
            predictions = np.where(y_test_proba > threshold, 1, 0)
            balanced_accuracy = balanced_accuracy_score(data["ytest"], predictions)
            proba_dict[threshold] = balanced_accuracy
            #print(f"Balanced Accuracy : {balanced_accuracy} at threshold : {threshold} ")
        # Find the maximum value in the dictionary proab_dict
        max_accuracy = max(proba_dict.values())
        # Find key for a value in dictionary
        threshold = get_key(proba_dict, max_accuracy)
        print(
            f"Best balanced accuracy of {np.round(max_accuracy,3)} obtained at threshold {np.round(threshold,3)} "
        )

    else :
        # Calculate the Predictions.
        y_test_pred = model.predict(data["xtest"])
        balanced_accuracy = balanced_accuracy_score(data["ytest"], y_test_pred)
        print(
            f"Test balanced accouracy of {balanced_accuracy} obtained "
        )
       
    return model

def get_segment_rocauc(columns, model, data_dict):
    """
    Segment the data based on the ROC-AUC score.
    Args:
        columns (_type_): _description_
        model (_type_): _description_
        data_dict (_type_): _description_
    """
    # Calculate the predictions.
    pred_proba = model.predict_proba(data_dict["xtest"])[:,1]
    xtest = data_dict["xtest"].copy()
    
    xtest['pred_proba'] = pred_proba
    xtest['target'] = data_dict['ytest']
    
    #For each column in column list
    for col in columns :
        print(f"\nCreating segment wise ROC for : {col}")
        #Get the list of unique values in the column
        unique_vals = xtest[col].unique()
        roc_auc_dict= {}
        for val in unique_vals:
            y_test_segment = xtest[xtest[col] == val]['target']
            y_proba_segment = xtest[xtest[col] == val]['pred_proba']
            roc_auc_score_segment = roc_auc_score(y_test_segment, y_proba_segment)
            roc_auc_dict[val] = roc_auc_score_segment
        #Convert the dictionary to dataframe
        seg_roc_auc_df = pd.DataFrame.from_dict(roc_auc_dict,columns=['roc-auc'],orient='index')
        # Sort the dataframe by index
        seg_roc_auc_df = seg_roc_auc_df.sort_index()
        # Create a sns bar plot with angled labels at 45 degrees
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=seg_roc_auc_df.index, y=seg_roc_auc_df['roc-auc'])
        plt.xticks(rotation=70)
        plt.show()
            
    