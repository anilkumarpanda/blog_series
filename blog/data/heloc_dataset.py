"""
Python class to read and process the Heloc dataset used in the FICO explainability challenge.
"""

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class HelocDataset:
    def __init__(self) -> None:
        self.path = None
        self.df = None
        # Define the target column name.
        self.target_column = "RiskPerformance"

    def _read_data(self):
        """
        Reads the dataset form the pre-specified path.
        """
        try:
            logger.info(f"Reading data from path {self.path}")
            self.df = pd.read_csv(self.path, sep=",")
            logger.info(f"Read data with shape {self.df.shape}")
        except:
            logger.debug(f"Error in reading dataset. Check path or reading.")

    def _process_data(self):
        """
        Process and clean data.
        """
        # Data Cleaning based on Error analysis.
        logger.info("Cleaning data based on error analysis.Removing rows")
        self.df = self.df[self.df["NumSatisfactoryTrades"] >= 0]
        self.df = self.df[self.df["ExternalRiskEstimate"] >= 0]

        logger.info(" Converting to numeric values for target column")
        # Convert the target columns to numeric.
        self.df[self.target_column] = self.df["RiskPerformance"].apply(
            lambda x: 1 if "Bad" in x else 0
        )

    def get_data(self, path, dropna, sample=-1):
        """
        Get the data from the path.
        """
        self.path = path
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
