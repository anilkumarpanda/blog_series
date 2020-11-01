# Code to explore constrastive explaination.
# Compare this with Shap explaination.

import pandas as pd 
from loguru import logger 
from blog.data.data_cleaner_factory import DataCleanerFactory

dcf = DataCleanerFactory()
lnt_dataset  = dcf.getDataset('lnt')
lnt_dataset.get_data(path='data/lnt_dataset.csv')

