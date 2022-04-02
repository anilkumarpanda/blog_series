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
from blog.model.optimise import ROCAUCObjective,MultiObjective,PRAUCObjective
from blog.model.evaluation import show_model_results
from probatus.interpret import ShapModelInterpreter
from blog.data.lendingclub_dataset import LendingClubDataset
from blog.utils.df_utils import split_cols_by_type
from blog.model.sampling import oot_split
from blog.model.regularisation import get_monotone_constraints

dataset_name = "lendingclub"
target ='loan_status'
path = 'data/lc_2007_2017.csv'
initial_features = ['addr_state','issue_d','annual_inc', 'emp_length', 'emp_title', 
                  'fico_range_high', 'fico_range_low', 'home_ownership', 'application_type',
                  'initial_list_status', 'loan_amnt', 'num_actv_bc_tl', 'loan_status', 
                  'mort_acc', 'tot_cur_bal', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 
                  'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 
                  'total_acc', 'verification_status']

lcd = LendingClubDataset()
X,y = lcd.get_data(path=path,use_cols=initial_features,dropna=False,sample=-1)
logger.info(f'\nTarget distribution:\n{y.value_counts(normalize=True)}')

# Convert string columns to categorical.
cat_cols,num_cols = split_cols_by_type(X)
logger.info(f"No.of categorical columns: {len(cat_cols)}, No.of numerical columns: {len(num_cols)}")

for col in cat_cols:
    X[col] = X[col].astype('category')
data = X.copy()
data[target] = y

X_it,y_it,X_oot,y_oot = oot_split(df=data,split_date='Sep-2016',split_col='issue_d',target_col=target)
data_dict = {'xtrain': X_it, 'ytrain': y_it,'xtest' : X_oot, 'ytest' : y_oot}

#Select the best features based on SHAPRFE CV
#selected_features,fs_plot = select_features(data=data_dict,n_features=16,verbose=100)
selected_features = ['pub_rec_bankruptcies', 'purpose', 'emp_length', 'loan_amnt', 'title', 
                   'tot_cur_bal', 'emp_title', 'term', 'total_acc', 'home_ownership', 
                   'sub_grade', 'addr_state', 'mort_acc']

#Update the data dic
data_dict['xtrain'] = data_dict['xtrain'][selected_features]
data_dict['xtest'] = data_dict['xtest'][selected_features]

# logger.info("Training monotonic model")
# monotonic_constraints=get_monotone_constraints(data_dict=data_dict,target=target)
# print(monotonic_constraints)

# study = optuna.create_study(study_name='auc_objective',direction="maximize")
# study.optimize(ROCAUCObjective(data_dict['xtrain'],data_dict['ytrain']), n_trials=10, timeout=300)
# print("Number of finished trials: {}".format(len(study.trials)))
# print("Best trial:")
# trial = study.best_trial
# print("  Value: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))


model_params = {'n_estimators': 40, 'learning_rate': 0.04026317553428693, 'num_leaves': 10, 'feature_fraction': 0.8106789070836655}
logger.info(f"Creating model with params : {model_params}")
mono_lgb = lgb.LGBMClassifier(**model_params)
mono_model = show_model_results(data=data_dict,model=mono_lgb,calc_threshold=False)

# Train ShapModelInterpreter
shap_interpreter = ShapModelInterpreter(mono_model)
feature_importance = shap_interpreter.fit_compute(data_dict['xtrain'],data_dict['xtest'],data_dict['ytrain'],data_dict['ytest'])
# Save the plots for comparision
for feature in selected_features:
    fig = plt.figure()
    ax3 = shap_interpreter.plot('dependence', target_columns=[feature],show=False)
    fig.suptitle('Model', fontsize=12)
    fig.savefig(f'figures/{dataset_name}_shap_dependence_mono_{feature}.png')