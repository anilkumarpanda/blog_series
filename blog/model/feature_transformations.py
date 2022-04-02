# Code for simple feature transformations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import FunctionTransformer
from feature_engine.encoding import RareLabelEncoder,WoEEncoder
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
    cat_cols,num_cols = split_cols_by_type(data_dict['xtrain'])
    print("Length of cat_cols: ",len(cat_cols))
    print("Length of num_cols: ",len(num_cols))

    numerical_preprocessor = Pipeline(
    steps=[('num_cols', FunctionTransformer()),])

    categorical_preprocessor = Pipeline(steps=[
        ("rare", RareLabelEncoder(tol=0.03,n_categories=2,)),
        ("woe", WoEEncoder())
    ])

    preprocessor = ColumnTransformer(   
        transformers=[
            ("numerical_preprocessor", numerical_preprocessor, num_cols),
            ("categorical_preprocessor", categorical_preprocessor, cat_cols)
        ]
    )

    columns = data_dict['xtrain'].columns
    print("Length of columns: ", len(columns))
    
    logger.info(f"Transforming data...")
    data_dict['xtrain'] = preprocessor.fit_transform(data_dict['xtrain'], data_dict['ytrain'])
    data_dict['xtest'] = preprocessor.transform(data_dict['xtest'])

    print("Shape of transformed train data: ", data_dict['xtrain'].shape)
    print("Shape of transformed test data: ", data_dict['xtest'].shape)
    
    logger.info("Converting to Pandas dataframes...")
    data_dict['xtrain']= pd.DataFrame(data_dict['xtrain'], columns=columns)
    data_dict['xtest']= pd.DataFrame(data_dict['xtest'], columns=columns)
    
    return data_dict