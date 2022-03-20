"""
Code for simple model training and evaluation using XGBoost.
"""

from tabnanny import verbose
from xml.sax.handler import feature_namespace_prefixes
import pandas as pd 
from loguru import logger 
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from blog.data.data_cleaner_factory import DataCleanerFactory
from blog.model.regularisation import select_features,get_monotone_constraints
from blog.model.optimise import ROCAUCObjective
from blog.model.evaluation import show_model_results
from probatus.interpret import ShapModelInterpreter

dataset_name = "lnt"
target ='loan_default'
dcf = DataCleanerFactory()
dataset  = dcf.getDataset(dataset_name)
X,y = dataset.get_data(path='data/lnt_dataset.csv',dropna=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=65)
logger.info(f"Train shape : {X_train.shape} , {y_train.shape}")
logger.info(f"Test shape : {X_test.shape},{y_test.shape} ")

data_dict = {'xtrain': X_train, 'ytrain': y_train,'xtest' : X_test, 'ytest' : y_test}
logger.info("Selecting features")
# Select the best features based on SHAPRFE CV
#selected_features,fs_plot = select_features(data=data_dict,n_features=12,verbose=1)

selected_features = ['VoterID_flag', 'Aadhar_flag', 'NO.OF_INQUIRIES', 
'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'ltv', 'PRI.SANCTIONED.AMOUNT', 
'Employment.Type', 'PRI.OVERDUE.ACCTS', 'manufacturer_id', 'PRI.NO.OF.ACCTS', 
'disbursed_amount', 'PERFORM_CNS.SCORE']

# Subset the dataset with the selected features.
data_dict['xtrain'] = data_dict['xtrain'][selected_features]
data_dict['xtest'] = data_dict['xtest'][selected_features]

# Tune model parameters with Optuna.
# Once the parameters are tuned, the model is trained and the results are evaluated.
# We need not run this again.

# study = optuna.create_study(study_name='auc_objective',direction="maximize")
# study.optimize(ROCAUCObjective(X_train,y_train), n_trials=20, timeout=300)
# print("Number of finished trials: {}".format(len(study.trials)))
# print("Best trial:")
# trial = study.best_trial
# print("  Value: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

model_params = {
    'n_estimators': 430,
    'learning_rate': 0.031,
    'num_leaves': 30,
    'feature_fraction': 0.40
}

# # We can do further analysis on top 5 features.
top_5_features = ['PRI.SANCTIONED.AMOUNT','ltv','PERFORM_CNS.SCORE','PRI.NO.OF.ACCTS','disbursed_amount']

logger.info("Training non-monotonic model")
# Train a LGB model without monotonicity constraints.
logger.info(f"Creating model with params : {model_params}")
non_mono_lgb = lgb.LGBMClassifier(**model_params)
non_mono_model = show_model_results(data=data_dict,model=non_mono_lgb,calc_threshold=False)

# Train ShapModelInterpreter
shap_interpreter = ShapModelInterpreter(non_mono_model)
feature_importance = shap_interpreter.fit_compute(data_dict['xtrain'],data_dict['xtest'],data_dict['ytrain'],data_dict['ytest'])
#ax1 = shap_interpreter.plot('importance')
for feature in top_5_features:
    fig = plt.figure()
    ax3 = shap_interpreter.plot('dependence', target_columns=[feature],show=False)
    fig.suptitle('Non Monotonic Model', fontsize=12)
    fig.savefig(f'figures/{dataset_name}_shap_dependence_non_mono_{feature}.png')


# Train a LGB model with monotonicity constraints.

logger.info("Training monotonic model")
model_params['monotone_constraints']=get_monotone_constraints(data_dict=data_dict,target=target)
logger.info(f"Creating model with params : {model_params}")
mono_lgb = lgb.LGBMClassifier(**model_params)
mono_model = show_model_results(data=data_dict,model=mono_lgb,calc_threshold=False)

# Train ShapModelInterpreter
shap_interpreter = ShapModelInterpreter(mono_model)
feature_importance = shap_interpreter.fit_compute(data_dict['xtrain'],data_dict['xtest'],data_dict['ytrain'],data_dict['ytest'])
# Save the plots for comparision
for feature in top_5_features:
    fig = plt.figure()
    ax3 = shap_interpreter.plot('dependence', target_columns=[feature],show=False)
    fig.suptitle('Monotonic Model', fontsize=12)
    fig.savefig(f'figures/{dataset_name}_shap_dependence_mono_{feature}.png')