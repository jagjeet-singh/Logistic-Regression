from __future__ import division
import numpy as np
from LogReg import LogisticRegression

#Read the data

X_Train = np.genfromtxt('../data/XTrain.csv', delimiter=",")
Y_Train = np.genfromtxt('../data/YTrain.csv',delimiter=",")
Y_Train = Y_Train.reshape((Y_Train.shape[0],1))
X_Test = np.genfromtxt('../data/XTest.csv', delimiter=",")
Y_Test = np.genfromtxt('../data/YTest.csv', delimiter=",")
Y_Test = Y_Test.reshape((Y_Test.shape[0],1))

#X_Train = X_Train[0:500,0:1000]
#Y_Train = Y_Train[0:500,:]
#X_Test = X_Test[0:10,0:1000]
#Y_Test = Y_Test[0:10,:]

LogReg=LogisticRegression()

#Uncomment this only after your Logistic Regression Class has been Completed

weight=LogReg.train(X_Train,Y_Train)
print('weights calculated')
Y_predict=np.array(LogReg.predict_label(X_Test,weight))
print('y predicted')
print(Y_predict)


print (LogReg.calculateAccuracy(Y_predict,Y_Test))
