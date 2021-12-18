# -*- coding: utf-8 -*-
"""
Generate PCA of data (Eigenface)

@author: lightdi
"""

#imports
import math
import numpy as np
import scipy.linalg as la


def find_K_Max_Egin(Matrix, Eigen_NUM):
    """
    
    The function return the N max Eigen values and Eigen vecotr of the matrix

    Parameters
    ----------
    Matrix : numpy.ndarray
        Matriz with Egien values.
    Eigen_NUM : int
        Number of values to be returned.

    Returns
    -------
    Matrix of Eigen vector and Eigen Values.

    """
    NN, NN = Matrix.shape
    
    S, V = la.eigh(Matrix)
    
    Eigen_Vector = np.zeros((NN,Eigen_NUM),dtype=np.double)
    Eigen_Value  = np.zeros((1,Eigen_NUM),dtype=np.double)
    
    p = (NN -1)
    
    for i in range(0,Eigen_NUM):
        #print(i)
        #print (p)
        Eigen_Value[0,i] = S[p]    
        Eigen_Vector[:,i] = V[:,p]
        p = p - 1
        
    return Eigen_Vector, Eigen_Value

def Eigenface_f(Train, Eigen_NUM):
    
    NN, Train_NUM = Train.shape
    
    
    if NN <= Train_NUM:
        
        Mean_Image = np.mean(Train,1)
        Mean_Image = Mean_Image.reshape((len(Mean_Image),1))
        Train = Train - Mean_Image * np.ones((1,Train_NUM))
        R = np.matmul(Train.T,Train)/(Train_NUM-1)
        
        V, S = find_K_Max_Egin(R, Eigen_NUM)
        
        disc_value = S
        disc_set = V
        
    else:
        Mean_Image = np.mean(Train,1)
        Mean_Image = Mean_Image.reshape((len(Mean_Image),1))
        Train = Train - Mean_Image * np.ones((1,Train_NUM))
        
        R = np.matmul(Train.T,Train)/(Train_NUM-1)
        
        V, S = find_K_Max_Egin(R, Eigen_NUM)
        
        disc_value = S
        disc_set = np.zeros((NN,Eigen_NUM),dtype=np.double)
        
        Train = Train/math.sqrt((Train_NUM-1))

        for k in range(0,Eigen_NUM):
            a = np.matmul(Train,V[:,k])
            b = (1/math.sqrt(disc_value[:,k]))
            disc_set[:,k] = b*a
    
    return disc_set, disc_value, Mean_Image

        