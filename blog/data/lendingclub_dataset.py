"""
Python class to read and process the Lending Club dataset.
"""

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


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
            self.df = pd.read_csv(self.path, sep=",", usecols=self.use_cols)
            logger.info(f"Read data with shape {self.df.shape}")
        except:
            logger.debug(f"Error in reading dataset. Check path or reading.")

    def _process_data(self):
        """
        Process and clean data.
        """

        # Map the target data.

        logger.info("Mapping target variable to binary values.")
        self.df["loan_status"] = self.df["loan_status"].map(
            {
                "Fully Paid": 0,
                "Current": 0,
                "Charged Off": 1,
                "Late (31-120 days)": 1,
                "In Grace Period": 0,
                "Late (16-30 days)": 0,
                "Does not meet the credit policy. Status:Fully Paid": 0,
                "Does not meet the credit policy. Status:Charged Off": 1,
                "Default": 1,
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
