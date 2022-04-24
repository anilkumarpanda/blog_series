# Code for simple feature transformations
from itertools import groupby
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from feature_engine.encoding import RareLabelEncoder, WoEEncoder
from blog.utils.df_utils import split_cols_by_type
from loguru import logger
import pandas as pd


def get_simple_feature_transformation(data_dict):
    """

    A simple feature transformation that includes:
    a) rare label encoding
    b) WOE encoding for categorical features
    c) Identity transformation for numerical features

    Args:
        data_dict (_type_): _description_
    """
    # Convert string columns to categorical.
    cat_cols, num_cols = split_cols_by_type(data_dict["xtrain"])
    print("Length of cat_cols: ", len(cat_cols))
    print("Length of num_cols: ", len(num_cols))

    numerical_preprocessor = Pipeline(
        steps=[
            ("num_cols", FunctionTransformer()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            (
                "rare",
                RareLabelEncoder(
                    tol=0.03,
                    n_categories=2,
                ),
            ),
            ("woe", WoEEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_preprocessor", numerical_preprocessor, num_cols),
            ("categorical_preprocessor", categorical_preprocessor, cat_cols),
        ]
    )

    columns = [*num_cols, *cat_cols]

    logger.info(f"Transforming data...")
    data_dict["xtrain"] = preprocessor.fit_transform(
        data_dict["xtrain"], data_dict["ytrain"]
    )
    data_dict["xtest"] = preprocessor.transform(data_dict["xtest"])

    logger.info("Converting to Pandas dataframes...")
    data_dict["xtrain"] = pd.DataFrame(data_dict["xtrain"], columns=columns)
    data_dict["xtest"] = pd.DataFrame(data_dict["xtest"], columns=columns)

    return data_dict


def create_bunch_feats(data_dict):
    """
    Simple feature engineering method that includes creation of features
    of the type
    feature_1 = groupby(feat_x)[feat_y].mean()
    feature_2 = feat_y-groupyby(feat_x)[feat_y].mean()

    Easiest implementation is if feat_x is categorical and feat_y is numerical.

    """

    # Split cols by type
    cat_cols, num_cols = split_cols_by_type(data_dict["xtrain"])
    logger.info(f"Length of cat_cols: {len(cat_cols)}")
    logger.info(f"Length of num_cols: {len(num_cols)}")

    # Create a bunch of features

    for cat_col in cat_cols:
        for num_col in num_cols:
            logger.info(f"Creating feature {cat_col}_{num_col}")
            data_dict["xtrain"][f"{cat_col}_{num_col}_mean"] = data_dict["xtrain"].groupby(cat_col)[num_col].mean()
            data_dict["xtest"][f"{cat_col}_{num_col}_mean"] = data_dict["xtest"].groupby(cat_col)[num_col].mean()

            data_dict["xtrain"][f"{cat_col}_{num_col}_std"] = data_dict["xtrain"].groupby(cat_col)[num_col].std()
            data_dict["xtest"][f"{cat_col}_{num_col}_std"] = data_dict["xtest"].groupby(cat_col)[num_col].std()

            data_dict["xtrain"][f"{cat_col}_{num_col}_skew"] = data_dict["xtrain"].groupby(cat_col)[num_col].skew()
            data_dict["xtest"][f"{cat_col}_{num_col}_skew"] = data_dict["xtest"].groupby(cat_col)[num_col].skew()

            # Difference with the actual value

            data_dict["xtrain"][f"{cat_col}_{num_col}_mean"] = (
                data_dict["xtrain"][num_col]-data_dict["xtrain"].groupby(cat_col)[num_col].mean()
                )
            data_dict["xtest"][f"{cat_col}_{num_col}_mean"] = (
                data_dict["xtest"][num_col]- data_dict["xtest"].groupby(cat_col)[num_col].mean()
                )

    return data_dict     