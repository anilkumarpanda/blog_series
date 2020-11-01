#Python class to process the LnT Finance data.

import pandas as pd 
from loguru import logger

class LnTDataset():

    def __init__(self):
        self.path = None
        self.df = None
        self.target_column = 'loan_default'

    def _read_data(self):
        """
        Reads the dataset form the pre-specified path.    
        """
        try:
            logger.info(f'Reading data from path {self.path}')
            self.df = pd.read_csv(self.path,sep=',')
            logger.info(f'Read data with shape {self.df.shape}')
        except :
            logger.debug(f'Error in reading dataset. Check path or reading.')

    def _process_data(self):
        """
        Process and clean data.
        """
        #Drop all Id columns.
        logger.info(f'Dropping all id columns')
        self.df=self.df.drop([
            'UniqueID', 'branch_id','supplier_id', 
            'Current_pincode_ID','State_ID','Employee_code_ID', 
            'MobileNo_Avl_Flag'
            ],axis=1)

        logger.info(f'Calculating customer age')
        self._calc_age()

        logger.info(f'Calculating financial age of customer')
        self._finance_age()

    def _age(self,dur):
        """
        Function to split the string age to
        numbers.
        """    
        yr = int(dur.split('-')[2])
        if yr >=0 and yr<=19:
            return yr+2000
        else:
            return yr+1900

    def _calc_age(self):
        """
        Calculate the age of the customer.

        """
        self.df['Date.of.Birth'] = self.df['Date.of.Birth'].apply(self._age)
        self.df['DisbursalDate'] = self.df['DisbursalDate'].apply(self._age)
        self.df['Age']=self.df['DisbursalDate']-self.df['Date.of.Birth']
        self.df=self.df.drop(['DisbursalDate','Date.of.Birth'],axis=1)

    def _string_age(self,age_str):
        """
        Some columns have age as strings.
        e,g "0yrs 0mon"    
        """
        age_str = age_str.replace("mon",'')
        age_str = age_str.split("yrs")
        age = int(age_str[0])*12 + int(age_str[1])

        return age

    def _finance_age(self):
        """
        Calculate age for finace related data 
        e.g credit history and avg account age in yrs.
        """
        self.df['AVERAGE.ACCT.AGE'] = self.df['AVERAGE.ACCT.AGE'].apply(self._string_age)
        self.df['CREDIT.HISTORY.LENGTH'] = self.df['CREDIT.HISTORY.LENGTH'].apply(self._string_age)


    def get_data(self,path,dropna=True):
        """
        Returns the processed X, y.
        This dataset can then be used directly.
        """
        self.path = path
        #Read the data.    
        self._read_data()
        #Process he data.
        self._process_data()

        if dropna:
            logger.info(f'Dropping na rows.')
            self.df.dropna(inplace=True)
    
        y = self.df[self.target_column]
        X = self.df.drop([self.target_column],axis=1)

        logger.info(f'Shape of training data X :{ X.shape}, y : {y.shape}.')
        return X,y



