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
from blog.model.optimise import ROCAUCObjective
from blog.model.evaluation import show_model_results
from probatus.interpret import ShapModelInterpreter
from blog.data.lendingclub_dataset import LendingClubDataset
from blog.utils.df_utils import split_cols_by_type

dataset_name = "lendingclub"
target ='loan_status'
path = 'data/lc_2007_2017.csv'
final_features = ['addr_state', 'annual_inc', 'earliest_cr_line', 'emp_length', 'emp_title', 
                  'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'application_type',
                    'initial_list_status', 'int_rate', 'loan_amnt', 'num_actv_bc_tl', 'loan_status', 
                    'mort_acc', 'tot_cur_bal', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 
                    'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 
                    'total_acc', 'verification_status']

lcd = LendingClubDataset()
X,y = lcd.get_data(path=path,use_cols=final_features,dropna=False)
logger.info(f'\nTarget distribution:\n{y.value_counts(normalize=True)}')

# Convert string columns to categorical.
cat_cols,num_cols = split_cols_by_type(X)
for col in cat_cols:
    X[col] = X[col].astype('category')


