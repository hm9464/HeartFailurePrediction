# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:13:44 2017

@author: Himanshu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:32:36 2017

@author: Himanshu
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# activating python 3.5 environment: activate py35/deactivate py35
# running script in virtual environment: 
# spyder --new-instance to activate python 3.5 in spyder (from cmd: tools>open cmd)

'''
Data description:
    Age: number
    Sex: Binary (male=1)
    Cp: chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
    Trestbps: resting BP (in mm HG)
    Cholesterol: serum cholestrol in mg/dl
    Fbs: fasting blood sugar > 120mg/dl (1=yes)
    RestECG: 0 = normal; 1 = having ST-T; 2 = hypertrophy
    Thalach: max heart rate achieved
    Exang: exercise induced angina (1 = yes; 0 = no)
    Oldpeak: ST depression induced by exercise relative to rest
    Slope: slope of the peak exercise 
    Ca: number of major vessels (0-3) colored by flourosopy
    Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
'''

# Part 1 - Data Preprocessing ############################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Himanshu/Desktop/HeartDisease2.csv')
X = dataset.iloc[:, 0:13].values #index of columns in the independent (predictor) variables
y = dataset.iloc[:, 13:18].values #col 13 (what we are predicting)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# test size=0.2 means 20% of total rows is test (8000 train, 2000 test)

# Feature Scaling - MUST scale for any NN model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN! ##############################################

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # used to initialize NN
from keras.layers import Dense # model to create different layers in NN

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
# dense helps to put an initial weight (needs to start somewhere)
# add (layer) will add a layer
# 6 nodes in the hidden layer (tip: input nodes + output nodes /2), and tells next layer no. of nodes to expect
# uniform is to randomly initialize the weights to a uniform distribution
# activation is the function you will use (relu is rectifier)
# input dim --> number of inputs from input layer

# Adding the second hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
# knows what inputs to expect because there is already an input layer created

# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# using adam optimizer --> algorithm to use to find the optimal weights
# loss: need to have a loss function (which you are trying to minimize), binary crossentropy for binary output
                                     
# Fitting the ANN to the Training set and training the ANN
classifier.fit(X_train, y_train, batch_size = 15, epochs = 100)
# fit(training set, ouput of training set, batch size, epochs)
# batch size: how many observations pass through before we update the weights
# epochs: number of times the whole training set goes through ANN

# Part 3 - Making predictions and evaluating the model ############################

# Predicting the Test set results
y_pred = classifier.predict(X_test) # gives prediction for each observation in test set
# use higher threshold for sensitive info (like medicine)
# now in y_pred dataframe, it gives answer as true/false, rather than just probability
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

####################################################################

'''checking HF severity for a new row: e.g. patient with:
Age: 54
Sex: Male
Cp: 4
Trestbps: 168
Cholesterol: 350
Fbs: no (0)
RestECG: 2
Thalach: 167
Exang: yes (1)
Oldpeak: 2.8
Slope: 2
Ca: 2
Thal: 7
'''

sample_patient = sc.transform(np.array([[54,1,4,168,350,0,2,167,1,2.8,2,2,7]]))
sample_pred = classifier.predict(sample_patient)