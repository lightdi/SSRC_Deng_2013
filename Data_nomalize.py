"""
Normalize Data form matrix
@author:Lightdi
"""

import numpy as np

def data_normalize(data):
    '''
    normalize data as in matlab toolbox usingo l2 norm

    Parameters
    ----------
    data : array
        data to be normalized.

    Returns
    -------
    normalized : TYPE
        Normalized data.

    '''
    normalized = data/np.matlib.repmat(np.sqrt(np.sum(np.power(data,2), axis=0)),len(data),1)
    return normalized