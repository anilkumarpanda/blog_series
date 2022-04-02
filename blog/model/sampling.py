# Sampling functions
from loguru import logger as logging
import pandas as pd
import numpy as np

def oot_split(df, split_date,split_col,target_col):
    """Splits dataset into in-time and out-of-time samples.

    Args:
        df (pd.DataFrame): Full model dev sample, incl. col `rqs_dat_rqs` (application date)
        split_date (str): Date to split on.
        split_col (str): Column to split on.
        target_col (str): Target column.
        other_cols(string): If 'filter' only the features are selected. Features are columns that start with an 'f'
        if 'passthrough', columns are not filtered.

    Returns:
        df_it (pd.DataFrame): in-time data sample
        df_oot (pd.DataFrame): out-of-time data sample
    """
    split_date = pd.Timestamp(split_date)
    
    df_it = df[(df[split_col] < split_date)].copy()
    df_oot = df[(df[split_col] >= split_date)].copy()
    
    oot_pct = round(df_oot.shape[0] / (df_oot.shape[0] + df_it.shape[0]) * 100, 1)
    it_pct = round(df_it.shape[0] / (df_oot.shape[0] + df_it.shape[0]) * 100, 1)

    defaults_it = df_it[df_it[target_col] == 1].shape[0]
    defaults_oot = df_oot[df_oot[target_col] == 1].shape[0]
    
    target = target_col

    X_it = df_it.copy()
    y_it = df_it[target]
    X_oot = df_oot.copy()
    y_oot = df_oot[target]
    

    it_stat_dict = {
        "name": "IT",
        "X_shape": X_it.shape,
        "y_shape": y_it.shape,
        "default_rate": np.sum(y_it) / y_it.shape[0] * 100,
        "pct_of_data": it_pct,
        "earliest_disbursed_date": df_it[split_col].min(),
        "latest_disbursed_date": df_it[split_col].max(),
        "default_count": defaults_it,
    }

    oot_stat_dict = {
        "name": "OOT",
        "X_shape": X_oot.shape,
        "y_shape": y_oot.shape,
        "default_rate": np.sum(y_oot) / y_oot.shape[0] * 100,
        "pct_of_data": oot_pct,
        "earliest_disbursed_date": df_oot[split_col].min(),
        "latest_disbursed_date": df_oot[split_col].max(),
        "default_count": defaults_oot,
    }

    # Combine the information in to dataframe.
    logging.info(f"IT data info \n : {it_stat_dict}")
    logging.info(f"OOT data info \n : {oot_stat_dict}")

    
    logging.info(
        f"OOT sample contains {defaults_oot} defaults, "
        f"{round(defaults_oot / (defaults_it + defaults_oot) * 100, 1)}% of total defaults "
        f"({defaults_it + defaults_oot})"
    )

    # Drop the split column.
    X_it.drop(columns=[split_col,target_col], axis=1, inplace=True)
    X_oot.drop(columns=[split_col,target_col], axis=1, inplace=True)
    
    return X_it, y_it, X_oot, y_oot
