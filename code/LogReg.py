#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    

    
    
    def calculateGradient(self, weight, X, Y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
        
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is (d+1)-by-1 dimensional numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''
        h = self.sigmoid(np.dot(X,weight))
        w = weight
        w = np.array(np.zeros(np.shape(weight)))
        w[0] = weight[0]
        Gradient = np.dot(np.transpose(X),h-Y) + regLambda*weight
        Gradient = Gradient - regLambda*w
    
        return Gradient    

    def sigmoid(self, Z):
        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
            '''
        s = 1/(1+np.exp(-Z))
         
        return s

    def update_weight(self,X,Y,weight):
        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        gradient = self.calculateGradient(weight, X, Y, self.regLambda)
        new_weight = weight - self.alpha*gradient
        
        return new_weight
    
    def check_conv(self,weight,new_weight,epsilon):
        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged, otherwise False

        '''
        difference = new_weight-weight
        difference_square = np.square(difference)
        l2_norm = np.sum(difference_square)
#        l2_norm = np.linalg.norm(new_weight-weight,2)
        
        if l2_norm <= epsilon:
            return True
        else:
            return False
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape
        
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1))
        
        iter = 0
        
        while(iter<self.maxNumIters):
        
        
        
#        for iter in range(1,self.maxNumIters+1):
            
            
            self.new_weight=self.update_weight(X,Y,self.weight)
            conv = self.check_conv(self.weight,self.new_weight,self.epsilon)
            if conv == True:
                break
            self.weight = self.new_weight
            iter = iter+1
        return self.new_weight

    def predict_label(self, X,weight):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        
        h = self.sigmoid(np.dot(X,weight))
        np.place(h,h>0.5,1)
        np.place(h,h<=0.5,0)
        result=h
        
        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''
        correct = np.sum(Y_predict==Y_test)
        Accuracy = correct/np.size(Y_test)*100
        
        return Accuracy
    
        