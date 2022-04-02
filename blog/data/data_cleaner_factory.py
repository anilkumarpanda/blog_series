# Factory class to clean the dataset follows the factory pattern.
from blog.data.lnt_dataset import LnTDataset
from blog.data.heloc_dataset import HelocDataset
from blog.data.lendingclub_dataset import LendingClubDataset


class DataCleanerFactory:
    def __init__(self):
        self.dataset_name = None

    def getDataset(self, dataset_name, **kwargs):
        """
        Get the dataset object based on name.
        """
        self.dataset_name = dataset_name.lower()
        if self.dataset_name == "lnt":
            return LnTDataset(**kwargs)
        elif self.dataset_name == "heloc":
            return HelocDataset(**kwargs)
        elif self.dataset_name == "lendingclub":
            return LendingClubDataset(**kwargs)
        else:
            return NotImplementedError()
