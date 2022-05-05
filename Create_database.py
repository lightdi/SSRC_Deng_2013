"""
Load images files and create Train and Test datasets

"""

#Imports
import numpy as np
import cv2
import os 
from sklearn import preprocessing as pp
from Database import Database
from Data_nomalize import data_normalize
import numpy.matlib
import scipy.io as sio

def load_AR_from_mat_file(train_file:str, test_file:str):
    """
    This function load data from matlab file .mat, this 
    function was used to compare this code with matlab code.
    """
    
    train_matrix = sio.loadmat(train_file)
    test_matrix = sio.loadmat(test_file)

    train = Database()

    train.X = train_matrix['train'][0][0][0]
    train.Y = np.matrix['train'][0][0][1]

    test = Database()

    test.X = test_matrix['test'][0][0][0]
    test.Y = train_matrix['test'][0][0][1]

    return train, test

def load_AR_files_from_folder(folder:str):
    """
        This function load AR dataset images to Numpy Matrix. 
        Was used OpenCV to load imagem to matrix and conver to grayscale.
        All images will be reshaped to 1D vector, which will be each column of the new matrix.
    
    """
    # Image index on folder
    i = 0

    allImages = Database()

    for filename in os.listdir(folder):
        image = cv2.imread(folder + "/" + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_vector = image.reshape(-1)

        #if empty will creante a zeros matrix for all images on folder
        if allImages.X is None:
            allImages.X = np.empty([image.size, len(os.listdir(folder))])

        allImages.X[:,i] = img_vector
        
        i+=1

    return make_AR_train_test_dataset(allImages)

def make_AR_train_test_dataset(dataset:Database()):
    '''
    Parameters
    ----------
    dataset : BaseDados() BaseDados class with all images matrix
       Create Train and Test dataset from BaseDados class with all images matrix

    Returns
    -------
    Train BaseDados.
    Test BaseDados
    '''
    train = Database()
    train.X = np.empty ([len(dataset.X),int(len(dataset.X[1])/2)])
    train.Y = np.ones ([1,int(len(dataset.X[1])/2)])

    test = Database()
    test.X = np.empty ([len(dataset.X),int(len(dataset.X[1])/2)])
    test.Y = np.ones ([1,int(len(dataset.X[1])/2)])

    idx = 0 
    l = 1

    # Each subject on AR dataset has twenty-six imagens, 
    # these images will be separate random in thirteen for train as thirteen for test
    for i in range(1, len(dataset.X[1]),26):
        X = dataset.X[:,(i-1):(i-1)+26]
        X = X[:,np.random.permutation(X.shape[1])]

        train.X[:, idx:(idx+13)] = X[:,0:13]
        test.X[:,idx:(idx+13)] = X[:,13:26]

        train.Y[:,idx:(idx+13)] = train.Y[:,idx:(idx+13)] * l
        test.Y[:,idx:(idx+13)] = test.Y[:,idx:(idx+13)] * l

        idx = idx + 13

        l = l + 1

    return train, test
    
def load_FRGC_files_from_folder(folder:str):
    """
        This function load AR dataset images to Numpy Matrix. 
        Was used OpenCV to load imagem to matrix and conver to grayscale.
        All images will be reshaped to 1D vector, which will be each column of the new matrix.
    
    """
    # Image index on folder
    # Image index on folder
    i = 0
    y = 0

    allImages = Database()
    last_subject = ""
    for filename in os.listdir(folder):
        if not (filename.endswith(".jpg") or  filename.endswith(".png") \
            or  filename.endswith(".JPG") or  filename.endswith(".PNG")):
            continue 
        image = cv2.imread(folder + "/" + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_vector = image.reshape(-1)
        
        Subject_Id = filename.split("d")[0]
        
        #if the subject is the first
        if last_subject == "":
            last_subject = Subject_Id
        
        #if the subject is not the same as the last subject
        if Subject_Id != last_subject:
            y = y + 1
            last_subject = Subject_Id

        #if empty will creante a zeros matrix for all images on folder
        if allImages.X is None:
            allImages.X = np.empty([image.size, len(os.listdir(folder))])

        allImages.X[:,i] = img_vector
        
        #Create Y dataset
        if allImages.Y is None:
            allImages.Y = np.zeros([1, len(os.listdir(folder))])
        
        allImages.Y[0,i] = y
        
        i+=1
    
    
    return allImages

def load_FERET_files_from_folder(folder:str, dictionary:dict):
    """
        This function load FERET dataset images to Numpy Matrix. 
        Was used OpenCV to load imagem to matrix and conver to grayscale.
        All images will be reshaped to 1D vector, which will be each column of the new matrix.

        Returns 
            allImage: Database with all images matrix and Y ground truth.
            dictionary: Dictionary with all images and ground truth.


    
    """
    # Image index on folder
    # Image index on folder
    i = 0
    y = 0

    dictionary = {}
    
    allImages = Database()
    last_subject = ""
    for filename in os.listdir(folder):
        if not (filename.endswith(".jpg") or  filename.endswith(".png") \
            or  filename.endswith(".JPG") or  filename.endswith(".PNG")):
            continue 
        image = cv2.imread(folder + "/" + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_vector = image.reshape(-1)
        
        Subject_Id = filename.split("f")[0]

        if 
        
        #if the subject is the first
        if last_subject == "":
            last_subject = Subject_Id
        
        #if the subject is not the same as the last subject
        if Subject_Id != last_subject:
            y = y + 1
            last_subject = Subject_Id

        #if empty will creante a zeros matrix for all images on folder
        if allImages.X is None:
            allImages.X = np.empty([image.size, len(os.listdir(folder))])

        allImages.X[:,i] = img_vector
        
        #Create Y dataset
        if allImages.Y is None:
            allImages.Y = np.zeros([1, len(os.listdir(folder))])
        
        allImages.Y[0,i] = y
        
        i+=1
    
    print(y)
    
    return allImages

