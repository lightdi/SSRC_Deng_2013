# -*- coding: utf-8 -*-
"""
Algoritmo SSRC

@author: lightdi
"""
#imports
#~~ Load Import function for load image files
import time
from Data_nomalize import data_normalize
from Create_database import load_AR_files_from_folder
from SSRC import ssrc


#Get the execution time


# stat load images from dataset folder
print('# Load images files ')
start = time.time()
train, test =load_AR_files_from_folder("dataset/AR-Cropped/")
end = time.time()
print("The time of execution of Load Images :", round((end-start)/60,3), "Minutos")

train.X = data_normalize(train.X)
test.X = data_normalize(test.X)

lamb = 1e-2
verbose = True
eigenface_flag = True
pca_dim = 300

print('# Start SRC')
start = time.time()
accuracy_ssrc = ssrc(train, test, lamb, pca_dim)
end = time.time()

print('# SSRC Accuracy =', round(accuracy_ssrc,5), '% \n')

print("The time of execution of function ssrc :", round((end-start)/60,3), "Minutos")


