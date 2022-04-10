# Code to perform RuleFit on the Lending club data.

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
from blog.model.regularisation import select_features, get_monotone_constraints
from blog.model.optimise import ROCAUCObjective, MultiObjective, PRAUCObjective
from blog.model.evaluation import show_model_results
from probatus.interpret import ShapModelInterpreter
from blog.data.lendingclub_dataset import LendingClubDataset
from blog.model.sampling import oot_split
from blog.model.regularisation import get_monotone_constraints
from blog.model.feature_transformations import get_simple_feature_transformation
from blog.model.optimise import getObjective, tune_model

dataset_name = "lendingclub"
target = "loan_status"
path = "data/lc_2007_2017.csv"


initial_features = ['verification_status', 'emp_title', 'int_rate', 'loan_amnt', 'total_rec_int',
                    'total_acc', 'tot_cur_bal', 'fico_range_low', 'fico_range_high', 'grade',
                    'total_rev_hi_lim', 'sub_grade', 'initial_list_status', 'purpose', 'issue_d', 
                    'emp_length', 'pub_rec_bankruptcies', 'last_pymnt_amnt', 'num_actv_bc_tl', 'total_pymnt',
                    'loan_status', 'term', 'home_ownership', 'revol_util', 'application_type', 'addr_state', 
                    'inq_last_6mths', 'pub_rec', 'dti', 'mort_acc', 'revol_bal', 'title', 'annual_inc',
                    'out_prncp', 'open_acc']

# combine two lists without duplicates
initial_features = list(set(initial_features))
print(initial_features)
# ================== Read and clean the dataset.===========================================
logger.info(f"1.Read data")
lcd = LendingClubDataset()
X, y = lcd.get_data(path=path, use_cols=initial_features, dropna=True, sample=-1)
logger.info(f"\nTarget distribution:\n{y.value_counts(normalize=True)}")

# Split data into IT & OOT datasets.
data = X.copy()
data[target] = y

X_it, y_it, X_oot, y_oot = oot_split(
    df=data, split_date="Jun-2016", split_col="issue_d", target_col=target
)
data_dict = {"xtrain": X_it, "ytrain": y_it, "xtest": X_oot, "ytest": y_oot}

# =================== Apply feature transformation =========================================
logger.info(f"2.Transform features")
data_dict = get_simple_feature_transformation(data_dict)

# =================== Feature Selection =========================================
logger.info(f"3.Starting feature selection.")
# Select the best features based on SHAPRFE CV
#selected_features, fs_plot = select_features(data=data_dict, verbose=100)

selected_features = [
    "emp_length",
    "addr_state",
    "revol_util",
    "revol_bal",
    "term",
    "num_actv_bc_tl",
    "title",
    "total_acc",
    "application_type",
    "purpose",
    "sub_grade",
]
logger.info(f"Final features :  {selected_features}")
# Update the data dictionary with the selected features.
data_dict["xtrain"] = data_dict["xtrain"][selected_features]
data_dict["xtest"] = data_dict["xtest"][selected_features]


# =================== Train the model =========================================


## ===================================== Rule Fit =========================================
##We see that LGBM model relies on a few features to predict the loan status.
##Can we achive similar results by using the Rulefit model?

logger.info(f"7.Rule Fit")

import numpy as np
import pandas as pd
from rulefit import RuleFit
from sklearn.ensemble import GradientBoostingRegressor


features = selected_features
rf_data_dict = {
    "xtrain": data_dict["xtrain"].values,
    "ytrain": data_dict["ytrain"].values,
    "xtest": data_dict["xtest"].values,
    "ytest": data_dict["ytest"].values,
}


#gb = GradientBoostingRegressor(n_estimators=100, max_depth=10, learning_rate=0.01)
rf = RuleFit(max_iter=1000,n_jobs=6,rfmode="classify")
# rf.fit(rf_data_dict['xtrain'], rf_data_dict['ytrain'], feature_names=features)
rf_model = show_model_results(data=rf_data_dict, model=rf,feature_names=features, calc_rocauc=False)
## Get the rules.
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
## Save the rules dataframe to markdown file.
rules.to_csv(f"assets/tables/{dataset_name}_rules.csv",index=False)