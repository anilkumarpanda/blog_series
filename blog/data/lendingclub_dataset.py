"""
Python class to read and process the Lending Club dataset.
"""

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from blog.utils.df_utils import split_cols_by_type


class LendingClubDataset:
    def __init__(self) -> None:
        self.path = None
        self.df = None
        self.use_cols = None
        # Define the target column name.
        self.target_column = "loan_status"

    def _read_data(self):
        """
        Reads the dataset form the pre-specified path.
        """
        try:
            logger.info(f"Reading data from path {self.path}")
            self.df = pd.read_csv(
                self.path, sep=",", usecols=self.use_cols, parse_dates=["issue_d"]
            )
            logger.info(f"Read data with shape {self.df.shape}")
        except:
            logger.debug(f"Error in reading dataset. Check path or reading.")

    def _process_data(self):
        """
        Process and clean data.
        """

        # Clean data.
        # Remove verh high vlaues for annual income and current balance
        self.df = self.df[self.df["annual_inc"] <= 250000]
        self.df = self.df[self.df["tot_cur_bal"] < 1000001]

        # Create features related of the FICO scores.
        self.df = self.df[self.df["pub_rec"] < 3]
        self.df["fico_diff"] = self.df["fico_range_high"] - self.df["fico_range_low"]
        self.df["fico_mean"] = (
            self.df["fico_range_high"] + self.df["fico_range_low"]
        ) / 2

        self.df = self.df.drop(["fico_range_high", "fico_range_low"], axis=1)
        # Map employment length to number of years.

        self.df["emp_length"] = self.df["emp_length"].map(
            {
                "< 1 year": 0,
                "1 year": 1,
                "2 years": 2,
                "3 years": 3,
                "4 years": 4,
                "5 years": 5,
                "6 years": 6,
                "7 years": 7,
                "8 years": 8,
                "9 years": 9,
                "10+ years": 10,
            }
        )

        # We want to predict if a particular loan was fully paid or charged off
        # (if it was paid or the customer was not able to pay back the loan amount)
        # We will drop rows which have loan_status other than Fully Paid and Charged Off
        # Map the target data.
        logger.info("Creating target data.")
        self.df = self.df[
            (self.df["loan_status"] == "Fully Paid")
            | (self.df["loan_status"] == "Charged Off")
        ]
        self.df["loan_status"] = self.df["loan_status"].map(
            {
                "Fully Paid": 0,
                "Charged Off": 1,
            }
        )

    def get_data(self, path, use_cols, dropna, sample=-1):
        """
        Get the data from the path.
        """
        self.path = path
        self.use_cols = use_cols
        self._read_data()
        self._process_data()

        if dropna:
            self.df = self.df.dropna()

        y = self.df[self.target_column]
        X = self.df.drop([self.target_column], axis=1)

        # Convert string columns to categorical.
        cat_cols, num_cols = split_cols_by_type(X)

        for col in cat_cols:
            X[col] = X[col].fillna("missing")
            X[col] = X[col].astype("category")

        if sample > 0:
            trainx, testx, trainy, testy = train_test_split(
                X, y, train_size=sample, stratify=y
            )
            logger.info(
                f"Shape of training data X :{ trainx.shape}, y : {trainy.shape}."
            )
            return trainx, trainy
        else:
            logger.info(f"Shape of training data X :{ X.shape}, y : {y.shape}.")
            return X, y
