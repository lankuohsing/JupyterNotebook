
# coding: utf-8

# In[1]:

#packages
import tensorflow as tf
import pandas as pd
from numpy import random as nr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  
import types


# In[2]:

#一些常量的定义
img_size=28*28
training_iters=200000
train_size=0.8
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
training_epochs=15
display_step=10
learning_rate=1e-3
batch_size=128
dropout1=0.75


# In[3]:

def read_data(filename):#用于读取数据的函数
    labeled_images = pd.read_csv(filename)
    [num_sample,num_feature]=labeled_images.shape
    images = labeled_images.iloc[:,1:]
    labels = labeled_images.iloc[:,:1]
    return (images,labels,num_sample,num_feature)


# In[4]:

filename='/Dataset_for_dl/input/train.csv'
(images,labels,num_sample,num_feature)=read_data(filename)


# In[5]:

#切割数据集
(train_images, test_images,train_labels, test_labels) = train_test_split(images, labels, train_size=train_size, random_state=0)
num_train=int(num_sample*train_size)


# In[6]:

#将DataFrame转化为Matrix
training_images=train_images.as_matrix()
training_labels=train_labels.as_matrix()
testing_images=test_images.as_matrix()
testing_labels=test_labels.as_matrix()


# In[7]:

training_images=training_images.astype('float32')#类型转换，以防matmul函数参数类型不匹配


# In[8]:

#OneHot encoding
enc = OneHotEncoder()  
enc.fit(training_labels)  
print("enc.n_values_ is:",enc.n_values_  )
num_class=enc.n_values_[0]#类别数量
training_labels_onehot=enc.transform(training_labels).toarray()
testing_labels_onehot=enc.transform(testing_labels).toarray()


# In[9]:

#Create some wrappers for simplicity
def conv2d(x,W,b,strides=1):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')#SAME表示卷积核可以停留在图像边缘
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

#Create model
def conv_net(x,W,b,keep_prob=1.):
    #Reshape input image
    x=tf.reshape(x,shape=[-1,28,28,1])
    
    #Convolution Layer 
    conv1=conv2d(x,W['wc1'],b['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # Convolution Layer
    conv2 = conv2d(conv1, W['wc2'],b['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, W['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, W['wd1']), b['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, W['out']), b['out'])
    return out


# In[10]:

#Weights and bias
W={
    #5x5 conv,1 input, 32outputs
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
    #5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    #Fully connected, 7x7x64 inputs, 2014 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64,1024])),
    'out': tf.Variable(tf.random_normal([1024,num_class]))
}
b = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_class]))
}
x=tf.placeholder(tf.float32,[None,img_size])
y_ = tf.placeholder(tf.float32, [None,num_class])#正确的标签
keep_prob = tf.placeholder(tf.float32)


# In[11]:

# Construct model
y = conv_net(x, W, b, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# In[12]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_indices=nr.choice(range(num_train),batch_size)
        batch_x=training_images[batch_indices,:]
        batch_y=training_labels_onehot[batch_indices,:]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y,keep_prob: dropout1})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y_: batch_y,
                                                             keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={x: testing_images,
                                      y_: testing_labels_onehot,
                                     keep_prob: 1.}))

