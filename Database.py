"""
Class that keep the matrix of images. 
X: is the matrix of images, which each column represents an image.
Y: is the matriz of labels, which each column represents a numeric label for subject.
P: is the matriz of Prototypes, which each column represents mean of imanges from the same subject.
P: is the matriz of Prototypes, which each column represents diferences of the mean of imanges from the all subject.
@author: Lightdi
"""

class Database:
    X = None
    Y = None
    P = None
    V = None
