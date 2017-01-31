import numpy as np
import scipy.ndimage
import scipy.stats
import matplotlib.pyplot as plt
import random
import time
import cPickle
from Linear_Regression import Data_Base, Linear_Classifier_nD, make_data_set

imdb = Data_Base('training_set5a.txt','cropped/',
                 {'Drescher':[1,0,0,0,0,0], 'Ferrera':[0,1,0,0,0,0], 'Chenoweth':[0,0,1,0,0,0],
                  'Baldwin':[0,0,0,1,0,0], 'Hader':[0,0,0,0,1,0], 'Carell':[0,0,0,0,0,1]})
classifier = Linear_Classifier_nD(imdb)
h = np.zeros((1025,6))
h[2][2] = 0.0001

error_a = classifier.error(np.dot(classifier.x,classifier.linear_weights+h))
error_b = classifier.error(np.dot(classifier.x,classifier.linear_weights))

#error_a = classifier.y-np.dot(classifier.x, classifier.linear_weights+h)
#error_b = classifier.y-np.dot(classifier.x, classifier.linear_weights)
true_grad = (error_a-error_b)/0.0001
cl_grad = classifier.grad(np.dot(classifier.x, classifier.linear_weights))
print true_grad, cl_grad[2][2]

classifier.grad_descent()
print 'DONE'

