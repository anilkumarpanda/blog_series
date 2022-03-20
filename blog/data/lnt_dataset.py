#Python class to process the LnT Finance data.

import pandas as pd 
from loguru import logger
from sklearn.model_selection import train_test_split
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

        logger.info(f'Calculating risk score of customer')
        self._map_credit_risk()

        logger.info(f'Calculating emplyment status customer')
        self._map_employment()


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

    def _map_credit_risk(self):
        """
        Process the credit risk information.
        """

        #Replacing all the values into Common Group

        self.df['PERFORM_CNS.SCORE.DESCRIPTION'].replace({'C-Very Low Risk':'Very Low Risk',
                                                    'A-Very Low Risk':'Very Low Risk',
                                                    'D-Very Low Risk':'Very Low Risk',
                                                    'B-Very Low Risk':'Very Low Risk',
                                                    'M-Very High Risk':'Very High Risk',
                                                    'L-Very High Risk':'Very High Risk',
                                                    'F-Low Risk':'Low Risk',
                                                    'E-Low Risk':'Low Risk',
                                                    'G-Low Risk':'Low Risk',
                                                    'H-Medium Risk':'Medium Risk',
                                                    'I-Medium Risk':'Medium Risk',
                                                    'J-High Risk':'High Risk',
                                                    'K-High Risk':'High Risk'},
                                                    inplace=True)

        
        risk_map = {'No Bureau History Available':-1, 
                    'Not Scored: No Activity seen on the customer (Inactive)':-1,
                    'Not Scored: Sufficient History Not Available':-1,
                    'Not Scored: No Updates available in last 36 months':-1,
                    'Not Scored: Only a Guarantor':-1,
                    'Not Scored: More than 50 active Accounts found':-1,
                    'Not Scored: Not Enough Info available on the customer':-1,
                    'Very Low Risk':4,
                    'Low Risk':3,
                    'Medium Risk':2, 
                    'High Risk':1,
                    'Very High Risk':0}

        self.df['PERFORM_CNS.SCORE.DESCRIPTION'] = self.df['PERFORM_CNS.SCORE.DESCRIPTION'].map(risk_map)                                            

    def _map_employment(self):
        """
        Map employment status to numbers.
        """
        employment_map = {'Self employed':0, 'Salaried':1, 'Not_employed':-1}
        self.df['Employment.Type'] = self.df['Employment.Type'].apply(lambda x: employment_map[x])

    def get_data(self,path,dropna=False,sample=-1):
        """
        Returns the processed X, y.
        This dataset can then be used directly.
        """
        self.path = path
        #Read the data.    
        self._read_data()
        #Drop Na   
     
        if dropna:
            logger.info(f'Dropping na rows.')
            self.df.dropna(inplace=True)
        else :
            ##TODO :Ways of handling Na values.
            pass

        #Process the data.
        self._process_data()
    
        y = self.df[self.target_column]
        X = self.df.drop([self.target_column],axis=1)
        
        if sample > 0 :
            trainx,testx,trainy,testy = train_test_split(X,y,train_size=sample,
                                                        stratify=y)
            logger.info(f'Shape of training data X :{ trainx.shape}, y : {trainy.shape}.')
            return trainx,trainy
        else :
        
            logger.info(f'Shape of training data X :{ X.shape}, y : {y.shape}.')
            return X,y



