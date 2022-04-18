#Code for target transformations
from loguru import logger

def get_sample_weights(data_dict):
    """
    Get sample weights for training data.

    Args :
        data_dict (dict): Data dictionary.

    """
    
    weight_map ={ 
        'a' : 17,
        'b':7 ,
        'c':4, 
        'd': 3,
        'e':2,
        'f':1,
        'g':1
        }

    #Map grade to dictionary values
    sample_weights = data_dict['xtrain']['grade'].map(weight_map)
    logger.info('Length of sample weights: {}'.format(len(sample_weights)))
    return sample_weights
