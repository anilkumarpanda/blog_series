#Code for various visualisations.
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

def plot_roc(models,data_dict):
    """
    Plot the ROC curve for the model for test data.

    Args:
        models (list): List of fitted models
        data_dict (dict): data dictionary containing the training and testing data.
    """

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    # Train the models and record the results
    for cls in models:
        yproba = cls.predict_proba(data_dict['xtest'])[::,1]
        fpr, tpr, _ = roc_curve(data_dict['ytest'],  yproba)
        auc = roc_auc_score(data_dict['ytest'], yproba)
        
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)
    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                result_table.loc[i]['tpr'], 
                label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()
