"""
This algorithm is based on Deng 2013 - SSRC

@author: Lighti
"""

#Imports 
import numpy as np
from Eigenface_f import Eigenface_f
from Data_nomalize import data_normalize
import solvehomotopy




#Algorithm 1. Superposed Sparse Representation based Classification (SSRC)
def ssrc(generic, train, test, lamb, pca_dim):
    """
    Function calculates the accuracy of ssrc on a given datasets

    Parameters
    ----------
    train : Databse
        Contains Database objects with train data.
    test : TYPE
        Contains Database objects with test data.
    lamb : float
        Regularization parameter λ > 0.
    pca_dim : TYPE
        Dimension of PCA.

    Returns
    -------
    float
        Accuracy of model on data.

    """
    if pca_dim <= 0: 
        raise Exception("PCA dimension must be greater than 0")
    
    # Creating generic variation matriz V from generic dataset

    classes = np.unique(generic.Y)
    class_num = len(classes)
    dim = len(generic.X)
    generic_num = len(generic.X[0])
    
    generic.V = np.zeros((dim,generic_num),dtype=np.double)
    generic.P = np.zeros((dim,generic_num),dtype=np.double)

    for j in range(0, class_num):
        print (j)
        idx =  np.where(generic.Y == classes[j])[1]
        data = generic.X[:,idx]
        
        centroid = np.sum(data, axis=1) / len(data[0]) 
        #Step 1: Compute the prototype matrix P according to (9)
        generic.P[:,j] = centroid

        # calculate variation matrix V
         # calculate a logmap of centroide of each sample
        l = len(idx)
        for k in range (0, l):
            diff = generic.X[:,idx[k]] - centroid
            #Step 1: Compute  the variation matrix V according to (10)
            generic.V[:,idx[k]] = np.real(diff)
        print('# Genered generic IntraVariDict for class ', j)

    
    #Generate Prototype dictionary  
    classes = np.unique(train.Y)
    class_num = len(classes)
    dim = len(train.X)
    train_num = len(train.X[0])
    test_num = len(test.X[0])
    
    #Initialize variation matrix and Prototype matrix
    train.V = np.zeros((dim,train_num),dtype=np.double)
    train.P = np.zeros((dim,class_num),dtype=np.double)
    
    for j in range(0, class_num):
        print (j)
        idx =  np.where(train.Y == classes[j])[1]
        data = train.X[:,idx]
        
        centroid = np.sum(data, axis=1) / len(data[0]) 
        #Step 1: Compute the prototype matrix P according to (9)
        train.P[:,j] = centroid
        
        # calculate a logmap of centroide of each sample
        l = len(idx)
        for k in range (0, l):
            diff = train.X[:,idx[k]] - centroid
            #Step 1: Compute  the variation matrix V according to (10)
            train.V[:,idx[k]] = np.real(diff)
        print('# Genered IntraVariDict for class ', j)
    
    #Step 2: Derive the projection matrix by applying PCA on the 
    #training samples A, and project the prototype and variation 
    #matrices to the p-dimensional space
    
    #Calculate projection matrix
    disc_set, a, b = Eigenface_f(train.X, pca_dim)
    #Applying the projection matrix
    train.P = np.matmul(disc_set.T,train.P)
    #train.V = np.matmul(disc_set.T,train.V)
    #generic.P = np.matmul(disc_set.T,generic.P)
    generic.V = np.matmul(disc_set.T,generic.V)
    test.X = np.matmul(disc_set.T,test.X)
    
    
    #Step 3: Normalize data
    test.X = data_normalize(test.X)
    train.P = data_normalize(train.P)
    #train.V = data_normalize(train.V)
    #generic.P = data_normalize(generic.P)
    generic.V = data_normalize(generic.V)
    
    
    #Step 4: Compute the residuals
    
    PX = np.concatenate((train.P, generic.V),axis=1)
    
    PY = classes
    
    #Prepare Class array
    classes = np.unique(PY)
    
    #prepare predicted label array
    identity = np.zeros((1,test_num))
    for i in range(0, test_num):
            
        y = test.X[:,i]
        
        #Calculate Sparse code
        maxiter = 50000
        lambda_coef = lamb #1e-2
        tolerance = 0.05
        stoppingCriterion = True
        
        P = np.matmul(PX.T,PX)
        P = np.linalg.inv(P + 0.001*np.eye(len(PX[0])))
        P = np.matmul(P,PX.T)
        
        x0 = np.matmul(P,y)
        
        alpha_beta = solvehomotopy.SolveHomotopy(PX,
                                             y,
                                             lambda_coef,
                                             tolerance,
                                             stoppingCriterion,
                                             maxiter, 
                                             x0)
        
        #Prepare redisual array
        residuals = np.zeros((1,class_num))
        
        #Calculate redisual for each class
        for j in range(0,class_num):
            
            non_idx =  np.where(PY != classes[j])
            sc = alpha_beta.copy()
            sc[non_idx] = 0
            
            residuals[0,j] = np.linalg.norm(y - np.matmul( PX,sc),2)
            
        c = np.amin(residuals)
        label = np.where(residuals == c)[1][0] + 1
        identity[0,i] = label
         
        
        correct = (label == test.Y[0, i])
        print("# SSRC: test:%04d, predict class: %03d --> ground truth :%03d (%d)\n" % ( i, label, test.Y[0, i], correct))
            
    # calculate accuracy
    correct_num = np.sum(identity == test.Y)
    accuracy = correct_num/test_num


    #Step 5: Result
    
    return accuracy