#Factory class to clean the dataset follows the factory pattern.
from blog.data.lnt_dataset import LnTDataset

class DataCleanerFactory():
    
    def __init__(self):
        self.dataset_name = None

    def getDataset(self,dataset):
        """
        Get dataset.
        """
        self.dataset_name=dataset.lower()
        if self.dataset_name=='lnt':
            return LnTDataset()
        else:
            return NotImplementedError()
    