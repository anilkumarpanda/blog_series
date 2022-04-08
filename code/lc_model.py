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
initial_features = [
    "addr_state",
    "issue_d",
    "annual_inc",
    "emp_length",
    "emp_title",
    "fico_range_high",
    "fico_range_low",
    "home_ownership",
    "application_type",
    "initial_list_status",
    "loan_amnt",
    "num_actv_bc_tl",
    "loan_status",
    "mort_acc",
    "tot_cur_bal",
    "open_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "purpose",
    "revol_bal",
    "revol_util",
    "sub_grade",
    "term",
    "title",
    "total_acc",
    "verification_status",
]

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
# selected_features,fs_plot = select_features(data=data_dict,n_features=16,verbose=100)

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

model_params = tune_model(data_dict["xtrain"], data_dict["ytrain"], "rocauc")
model_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "feature_pre_filter": False,
    "lambda_l1": 0.2147440348264995,
    "lambda_l2": 2.951088160975595e-07,
    "num_leaves": 32,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6816921315176412,
    "bagging_freq": 4,
    "min_child_samples": 25,
}


# =================== Train Model =========================================
logger.info(f"5.Train Model ")
logger.info(f"Creating model with params : {model_params}")
mono_lgb = lgb.LGBMClassifier(**model_params)
mono_model = show_model_results(data=data_dict, model=mono_lgb)

# ===================Interpret Model =========================================
# logger.info(f"6.Interpret Model")

# ## Train ShapModelInterpreter
# shap_interpreter = ShapModelInterpreter(mono_model)
# feature_importance = shap_interpreter.fit_compute(
#     data_dict["xtrain"], data_dict["xtest"], data_dict["ytrain"], data_dict["ytest"]
# )

# fig = plt.figure()
# ax1 = shap_interpreter.plot("importance", show=False)
# fig.suptitle("Feature Importance Plot", fontsize=12)
# fig.savefig(f"assets/figures/{dataset_name}_feature_importance.png")
# plt.close(fig)

# fig = plt.figure()
# ax2 = shap_interpreter.plot("summary", show=False)
# fig.suptitle("Feature Summary Plot", fontsize=12)
# fig.savefig(f"assets/figures/{dataset_name}_feature_summary.png")
# plt.close(fig)

# # Save the plots for comparision
# for feature in selected_features:
#     fig = plt.figure()
#     ax3 = shap_interpreter.plot("dependence", target_columns=[feature], show=False)
#     fig.suptitle(f"Dependence Plot : {feature}", fontsize=12)
#     fig.savefig(f"assets/figures/{dataset_name}_shap_dependence_mono_{feature}.png")
#     plt.close(fig)

# ===================================== Rule Fit =========================================
# We see that LGBM model relies on a few features to predict the loan status.
# Can we achive similar results by using the Rulefit model?

# logger.info(f"7.Rule Fit")

# import numpy as np
# import pandas as pd
# from rulefit import RuleFit
# from sklearn.ensemble import GradientBoostingRegressor


# features = selected_features
# rf_data_dict = {
#     "xtrain": data_dict["xtrain"].values,
#     "ytrain": data_dict["ytrain"].values,
#     "xtest": data_dict["xtest"].values,
#     "ytest": data_dict["ytest"].values,
# }


# #gb = GradientBoostingRegressor(n_estimators=100, max_depth=10, learning_rate=0.01)
# rf = RuleFit(max_iter=1000,n_jobs=6,rfmode="classify")
# # rf.fit(rf_data_dict['xtrain'], rf_data_dict['ytrain'], feature_names=features)
# rf_model = show_model_results(data=rf_data_dict, model=rf,feature_names=features, calc_rocauc=False)
# ## Get the rules.
# rules = rf.get_rules()
# rules = rules[rules.coef != 0].sort_values("support", ascending=False)
# ## Save the rules dataframe to markdown file.
# rules.to_csv(f"assets/tables/{dataset_name}_rules.csv",index=False)
