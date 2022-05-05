"""
This algorithm is based on Deng 2013 - SSRC

@author: Lighti
"""

#Imports 
import numpy as np
import time
#from Data_nomalize import data_normalize
from Create_database import load_FRGC_files_from_folder
from Create_database import load_FERET_files_from_folder
from Data_nomalize import data_normalize
from SSRC_Single import ssrc


print('# Load images files ')
start = time.time()
allIFRGCImage = load_FRGC_files_from_folder("d:\\FRGC\\FRGC-2.0-dist\\nd1\\Gallery\\")

allIFERETGallery = load_FERET_files_from_folder("d:\\Temp\\FeretGray\\Gallery\\")

allIFERETfb = load_FERET_files_from_folder("d:\\Temp\\FeretGray\\fb\\")

allIFERETfc = load_FERET_files_from_folder("d:\\Temp\\FeretGray\\fc\\")

allIFERETdup1 = load_FERET_files_from_folder("d:\\Temp\\FeretGray\\dup1\\")

allIFERETdup2 = load_FERET_files_from_folder("d:\\Temp\\FeretGray\\dup2\\")

end = time.time()
print("The time of execution of Load Images :", round((end-start)/60,3), "Minutos")

#Normalize the datasets
print('# Normalize images files ')
start = time.time()
allIFRGCImage.X = data_normalize(allIFRGCImage.X)
allIFERETGallery.X = data_normalize( allIFERETGallery.X ) 
allIFERETfb.X = data_normalize( allIFERETfb.X) # not ok
allIFERETfc.X = data_normalize( allIFERETfc.X)
allIFERETdup1.X = data_normalize( allIFERETdup1.X)
allIFERETdup2.X = data_normalize( allIFERETdup2.X)

end = time.time()
print("The time of execution of Normalize Images :", round((end-start)/60,3), "Minutos")

lamb = 1e-2
verbose = True
eigenface_flag = True
pca_dim = 300

print('# Start SRC in fb')
start = time.time()
accuracy_ssrc = ssrc( allIFRGCImage, allIFERETGallery, allIFERETfb, lamb, pca_dim)
end = time.time()

print('# SSRC fb Accuracy =', round(accuracy_ssrc,5), '% \n')

print("The time of execution of function ssrc in fb :", round((end-start)/60,3), "Minutos")


 
input("Press Enter to continue...")