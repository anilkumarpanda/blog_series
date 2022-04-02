"""
Code for simple model training and evaluation using LGBM.
"""

import pandas as pd 
from loguru import logger 
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from blog.data.data_cleaner_factory import DataCleanerFactory
from blog.model.regularisation import select_features,get_monotone_constraints
from blog.model.optimise import ROCAUCObjective,MultiObjective
from blog.model.evaluation import show_model_results
from probatus.interpret import ShapModelInterpreter
from blog.data.lendingclub_dataset import LendingClubDataset
from blog.utils.df_utils import split_cols_by_type
from blog.model.sampling import oot_split
from blog.model.regularisation import get_monotone_constraints

dataset_name = "lendingclub"
target ='loan_status'
path = 'data/lc_2007_2017.csv'
final_features = ['addr_state','issue_d','annual_inc', 'emp_length', 'emp_title', 
                  'fico_range_high', 'fico_range_low', 'home_ownership', 'application_type',
                  'initial_list_status', 'int_rate', 'loan_amnt', 'num_actv_bc_tl', 'loan_status', 
                  'mort_acc', 'tot_cur_bal', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 
                  'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 
                  'total_acc', 'verification_status']

lcd = LendingClubDataset()
X,y = lcd.get_data(path=path,use_cols=final_features,dropna=False,sample=-1)
logger.info(f'\nTarget distribution:\n{y.value_counts(normalize=True)}')

# Convert string columns to categorical.
cat_cols,num_cols = split_cols_by_type(X)
for col in cat_cols:
    X[col] = X[col].astype('category')
data = X.copy()
data[target] = y

X_it,y_it,X_oot,y_oot = oot_split(df=data,split_date='Jan-2017',split_col='issue_d',target_col=target)
data_dict = {'xtrain': X_it, 'ytrain': y_it,'xtest' : X_oot, 'ytest' : y_oot}

#print(X_it.dtypes)
# # Select the best features based on SHAPRFE CV
# selected_features,fs_plot = select_features(data=data_dict,n_features=11,verbose=1)

# study = optuna.create_study(study_name='auc_objective',direction="maximize")
# study.optimize(MultiObjective(data_dict['xtrain'],data_dict['ytrain']), n_trials=15, timeout=300)
# print("Number of finished trials: {}".format(len(study.trials)))
# print("Best trial:")
# trial = study.best_trial
# print("  Value: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

#logger.info("Training monotonic model")
#monotonic_constraints=get_monotone_constraints(data_dict=data_dict,target=target)

# Tune model parameters with Optuna.
# Once the parameters are tuned, the model is trained and the results are evaluated.
# We need not run this again.

study = optuna.create_study(study_name='auc_objective',direction="maximize")
study.optimize(ROCAUCObjective(data_dict['xtrain'],data_dict['ytrain']),n_trials=10, timeout=300)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# model_params = {
#     'n_estimators': 230,
#     'learning_rate': 0.05,
#     'num_leaves': 188,
#     'feature_fraction': 0.7,
#     #'monotone_constraints': monotonic_constraints
# }
logger.info(f"Creating model with params : {trial.params.items()}")
mono_lgb = lgb.LGBMClassifier(**trial.params.items())
mono_model = show_model_results(data=data_dict,model=mono_lgb,calc_threshold=False)