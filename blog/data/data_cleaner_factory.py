#Factory class to clean the dataset follows the factory pattern.
from blog.data.lnt_dataset import LnTDataset
from blog.data.heloc_dataset import HelocDataset

class DataCleanerFactory():
    
    def __init__(self):
        self.dataset_name = None

    def getDataset(self,dataset_name):
        """
        Get the dataset object based on name.
        """
        self.dataset_name=dataset_name.lower()
        if self.dataset_name=='lnt':
            return LnTDataset()
        elif self.dataset_name=='heloc':
            return HelocDataset()
        else:   
            return NotImplementedError()
    