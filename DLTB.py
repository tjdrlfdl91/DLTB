# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:49:41 2018

@author: mogae
"""

"""
HeuristicTinkers Project No.003 "DLTB"

CONTENTS
    0. Settings
        0.1 Directory Settings
        0.2 Package Imports

    1. Import Data
        1.1 Data Check
        1.2 Merge Train and Test
    
    2. Preprocessing
        2.1 Data Overview and Distribution
        2.2 Correlations
        2.3 NaN Handling
        2.4 Removing Garbage Variables
            2.4.1 Variance Threshold
        2.5 Encoding
            2.5.1 One Hot Encoding
            2.5.2 Label Encoding
        2.6 
    
     
"""

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

### 0. Settings
### 0.1 Directory Settings

# cwd에 본인 working directory 적어넣으세요

cwd = "C:/Users/mogae/Documents/Python_Scripts/DeepLearningTutorial/DLTB"
cwd = "C:/Python/git/dltb/DLTB"

import os
os.getcwd()
os.chdir(cwd)

### 0.2 Package Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

### 1. Import Data
### 1.1 Data Check

# 이미지 데이터가 npy형식으로 저장되어 있습니다

x_l = np.load('X.npy')
Y_l = np.load('Y.npy')

# plt로 간단하게 이미지를 그려볼 수 있습니다

img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[1026].reshape(img_size, img_size))
plt.axis('off')

# 데이터의 [204:409]가 0의 손 기호, [822:1027]가 1의 손 기호입니다
# 일단은 이 두 종류만 불러옵시다

X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

# train test를 갈라줍시다
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1126)
number_of_train = X_train.shape[0] #348개
number_of_test = X_test.shape[0] #62개

# 데이터 인식을 위해 3차원 데이터를 flattening 해줍시다
# 2차원 이미지 데이터(64 * 64 )를 1차원(4096)으로 바꿔주면 됩니다

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

# 아무런 이유는 없다는데 이새끼가 Transpose 시키라고 합니다
# 병신새끼

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

# Hidden layer는 1층, 노드는 3개로 해줍시다
# 랜덤으로 initial weight를 줍니다
# initial bias는 0으로 줍시다

def initialize_parameters(x_train, y_train):
    np.random.seed(1126)
    parameters = {"w1": np.random.rand(3, x_train.shape[0]) * 2 - 1,
                  "b1": np.zeros((3,1)),
                  "w2": np.random.randn(y_train.shape[0],3) * 2 - 1,
                  "b2": np.zeros((y_train.shape[0],1))}
    return parameters

# sigmoid 함수를 정의해줍니다

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# 
def forward_propagation(x_train, parameters):
    #forward propagation
    z1 = np.dot(parameters["w1"],x_train) + parameters["b1"]
    a1 = np.tanh(z1)
    z2 = np.dot(parameters["w2"],a1) + parameters["b2"]
    a2 = sigmoid(z2)

    coeff = {"z1": z1,
             "a1": a1,
             "z2": z2,
             "a2": a2}

    return a2, coeff

# Backward Propagation
def backward_propagation(parameters, coeff, X, Y):

    dz2 = coeff["a2"]-Y
    dw2 = np.dot(dz2, coeff["a1"].T)/X.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True)/X.shape[1]
    dz1 = np.dot(parameters["w2"].T,dz2)*(1 - np.power(coeff["a1"], 2))
    dw1 = np.dot(dz1,X.T)/X.shape[1]
    db1 = np.sum(dz1,axis = 1, keepdims=True)/X.shape[1]
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 0.01):
    parameters = {"w1": parameters["w1"]-learning_rate*grads["dw1"],
                  "b1": parameters["b1"]-learning_rate*grads["db1"],
                  "w2": parameters["w2"]-learning_rate*grads["dw2"],
                  "b2": parameters["b2"]-learning_rate*grads["db2"]}

def predict(parameters,x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
    return parameters
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    error = y_train - y_head
    loss = np.square(error)
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    derivative_weight = (np.dot(x_train,(error.T)))/x_train.shape[1]
    derivative_bias = np.sum(error)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

num_iterations = 500

# 2 - Layer neural network
def neural_network(x_train, y_train, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        a2, coeff = forward_propagation(x_train, parameters)
        # compute cost
        #cost = np.square(np.sum(y_train - a2))/x_train.shape[1]
        logprobs = np.multiply(np.log(a2),y_train)
        cost = -np.sum(logprobs)/y_train.shape[1]
         # backward propagation
        grads = backward_propagation(parameters, coeff, x_train, y_train)
         # update parameters
        parameters = {"w1": parameters["w1"]-0.01*grads["dw1"],
                  "b1": parameters["b1"]-0.01*grads["db1"],
                  "w2": parameters["w2"]-0.01*grads["dw2"],
                  "b2": parameters["b2"]-0.01*grads["db2"]}
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))

    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()

    print("train accuracy: {} %".format(100 - np.mean(np.abs(a2.round() - y_train)) * 100))
        
    return a2

neural_network(x_train, y_train, num_iterations=2000)
print("a2:", a2, "y:", y_train)

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))