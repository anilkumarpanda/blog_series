# Utility functions for dataframe manipulation
import numpy as np


def split_cols_by_type(df):
    """
    Split columns by type.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    return cat_cols, num_cols
