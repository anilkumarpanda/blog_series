"""
Code for testing the DataProfiler package

"""

import pandas as pd
from loguru import logger
from blog.data.lendingclub_dataset import LendingClubDataset
from blog.model.sampling import oot_split
import json
import dataprofiler as dp


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

for key, val in data_dict.items():
    logger.info("Creating Data Profiler for {}".format(key))
    profile = dp.Profiler(
        data_dict[key]
    )  # Calculate Statistics, Entity Recognition, etc
    readable_report = profile.report(report_options={"output_format": "pretty"})
    with open(f"assets/tables/{key}_profile.json", "w") as outfile:
        json.dump(readable_report, outfile)
