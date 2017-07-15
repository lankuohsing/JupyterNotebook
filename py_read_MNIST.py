
# coding: utf-8

# In[31]:

import os as os
import numpy as np
from PIL import Image as im
'''
dataset_file: 'trainimg' or 'testimage'
'''

# In[32]:

def read_mnist_pictures(dataset_file):
    images=[]
    labels=[]
    i=0
    for label in range(10):
        pathDir=os.listdir('/Dataset_for_dl/MNIST图片库/'+str(dataset_file)+'/pic2/'+str(label)+'/')
        for allDir in pathDir:
            child=os.path.join('%s%s' % ('/Dataset_for_dl/MNIST图片库/'+str(dataset_file)+'/pic2/'+str(label)+'/',allDir))
            img=im.open(child)
            #img=img.resize((16,16))#将图片缩小为16x16
            img_array=np.array(img)
            img_array=img_array[:,:,0]
            images.append
            img_array=np.reshape(img_array,-1)
            images.append(img_array)
            labels.append(label)
            i=i+1
    images=np.array(images)
    labels=np.array(labels)
    print("The size of "+dataset_file+" images:",i)
    return images,labels,i


