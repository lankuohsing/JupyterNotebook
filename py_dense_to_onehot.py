
# coding: utf-8

# In[1]:

import numpy as np


# In[ ]:

def dense_to_one_hot(labels_dense, num_classes):  
    num_labels = labels_dense.shape[0]  
    index_offset = np.arange(num_labels) * num_classes  
    labels_one_hot = np.zeros((num_labels, num_classes))  
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  
    return labels_one_hot 

