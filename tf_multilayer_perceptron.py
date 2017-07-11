# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:39:04 2017
@author: lankuohsing
"""
from __future__ import print_function
import tensorflow as tf
import pandas as pd
from numpy import random as nr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import types


img_size=28*28
learning_rate=0.01
batch_size=100
num_data=42000

labeled_images = pd.read_csv('./input/train.csv')
#images = labeled_images.iloc[0:5000,1:]
#labels = labeled_images.iloc[0:5000,:1]

images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]

(train_images, test_images,train_labels, test_labels) = train_test_split(images, labels, train_size=0.8, random_state=0)
#将DataFrame转换为matrix
training_images=train_images.as_matrix()
training_labels=train_labels.as_matrix()
testing_images=test_images.as_matrix()
testing_labels=test_labels.as_matrix()

enc = OneHotEncoder()
enc.fit(training_labels)
print("enc.n_values_ is:",enc.n_values_  )
training_labels_onehot=enc.transform(training_labels).toarray()
testing_labels_onehot=enc.transform(testing_labels).toarray()

testing_labels_onehot.shape

x=tf.placeholder(tf.float32,[None,img_size])#[none,img_size]表示在tf运行时能输入任意形数量的images,每张image尺寸为img_size
#模型参数，初始值均为零
W = tf.Variable(tf.zeros([784,10]))#权重值
b = tf.Variable(tf.zeros([10]))#偏置量
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])#正确的标签
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
for i in range(1000):
    batch_indices=nr.choice(range(int(num_data*0.8)),batch_size)
    batch_xs=training_images[batch_indices,:]
    batch_ys=training_labels_onehot[batch_indices,:]

    _,c=sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    if i % 50 == 0:
            print("Epoch:", '%04d' % (i+1), "cost=", \
                "{:.9f}".format(c))
print(batch_xs)
print(batch_ys)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: testing_images, y_: testing_labels_onehot}))


