import numpy as np
import scipy.ndimage
import scipy.stats
import matplotlib.pyplot as plt
import random
import time
import os


class Data_Base(object):
    def __init__(self, image_source, image_dir, classes):
        self.image_dir = image_dir
        self.image_source = image_source
        self.classes = classes
        self.num_class = len(self.classes)
        self.im_count = self.image_count()
        self.imdb, self.image_size = self.load_images()  # list of tuples (label_as_class_index, image_as_1D_array)

    def image_count(self):
        with open(self.image_source) as f:
            for i, l in enumerate(f):
                pass
        return i+1

    def load_images(self):
        f = open(self.image_source)
        i = 0
        for line in f:
            ndimage = scipy.ndimage.imread(self.image_dir+line.split('\t')[0], True)
            ndimage.resize(ndimage.size)
            ndimage /= 255.0
            label = self.classes[line.split('\t')[1].strip('\n')]
            if i == 0:
                image_db = [(label,ndimage)]
            else:
                image_db = image_db + [(label, ndimage)]
            i += 1
        return image_db, ndimage.size


class Linear_Classifier(object):
    def __init__(self, db):
        self.step_size = 0.001
        self.EPS = 5e-5
        self.max_iter = 1e5
        self.report_iter = 500
        self.subtract_image_mean = True
        self.db = db
        self.input_size = db.image_size
        self.linear_weights = self.init_weights()
        self.x, self.y = self.load()
        self.f_t = np.ndarray(int(self.max_iter))

    def init_weights(self):
        linear_weights = scipy.stats.norm.rvs(size = self.input_size+1, random_state=1)
        linear_weights /= 1024.0
        return linear_weights

    def forward(self):  #compute outputs of forward pass
        z = np.dot(self.x, self.linear_weights)
        '''
        z = np.ndarray(z_reg.shape)
        for i in range(z_reg.size):
            if z_reg[i] >= 0:
                z[i] = 1
            else:
                z[i] = 0
        '''
        return z

    def error(self,z):  #compute errors given regression, z is X*theta
        return np.sum(((self.y-z)**2))/(2*len(self.db.imdb))

    def grad(self,z): #compute gradient of error, z is X*theta
        return -1 * np.sum((self.y - z) * self.x.transpose(), 1) / len(self.db.imdb)

    def load(self):
        x = np.zeros((len(self.db.imdb),self.input_size+1,))
        y = np.zeros((len(self.db.imdb)))
        i = 0
        for im in self.db.imdb:
            x[i] = np.insert(im[1], 0, 1.)
            y[i] = im[0]
            i +=1
        if self.subtract_image_mean:
            x[:, 1:] = x[:, 1:] - np.average(x[:, 1:])
        return x, y

    def grad_descent(self):
        prev_t = self.linear_weights - 10 * self.EPS
        t = self.linear_weights.copy()
        iter = 0
        while np.linalg.norm(self.linear_weights - prev_t) > self.EPS and iter < self.max_iter:
            prev_t = self.linear_weights.copy()
            z = self.forward()
            error = self.error(z)
            grad = self.grad(z)
            self.linear_weights -= self.step_size * grad
            if iter % self.report_iter == 0:
                print "Iter", iter
                print "f(x) = %.5f" % (error)
                print "Gradient Magnitude: ", np.linalg.norm(grad), "\n"
                self.f_t[int(iter//self.report_iter)] = error
            iter += 1
        self.f_t = self.f_t[:int(iter//self.report_iter)]
        np.save(self.db.image_dir.strip('/')+'_'+self.db.image_source.split('.')[0],self.linear_weights)
        #self.linear_weights = t


class Linear_Classifier_nD(object):
    def __init__(self, db):
        self.step_size = 0.001
        self.EPS = 5e-5
        self.max_iter = 1e5
        self.report_iter = 500
        self.subtract_image_mean = True
        self.db = db
        self.input_size = db.image_size
        self.linear_weights = self.init_weights()
        self.x, self.y = self.load()
        self.f_t = np.ndarray(int(self.max_iter))

    def init_weights(self):
        linear_weights = np.zeros((self.input_size+1,self.db.num_class))
        return linear_weights
        #linear_weights = np.ndarray((self.db.num_class, self.input_size+1))
        #for i in range(0,self.db.num_class):
            #linear_weights[i] = scipy.stats.norm.rvs(size = self.input_size+1,random_state=1)
        #linear_weights /=1024.
        #return linear_weights.transpose()

    def forward(self):  #compute outputs of forward pass
        z = np.dot(self.x, self.linear_weights)
        '''
        z = np.ndarray(z_reg.shape)
        for i in range(z_reg.size):
            if z_reg[i] >= 0:
                z[i] = 1
            else:
                z[i] = 0
        '''
        return z

    def error(self,z):  #compute errors given regression
        return np.sum(np.sum(((self.y-z)**2)))/(2.*len(self.db.imdb))

    def grad(self,z):
        return np.dot(self.x.transpose(), (z-self.y))/float(len(self.db.imdb))

    def load(self):
        x = np.zeros((len(self.db.imdb),self.input_size+1,))
        y = np.zeros((len(self.db.imdb),self.db.num_class))
        i = 0
        for im in self.db.imdb:
            x[i] = np.insert(im[1], 0, 1.)
            y[i] = im[0]
            i +=1
        if self.subtract_image_mean:
            x[:, 1:] = x[:, 1:] - np.average(x[:, 1:])
        return x, y

    def grad_descent(self):
        prev_t = self.linear_weights - 10 * self.EPS
        t = self.linear_weights.copy()
        iter = 0
        while np.linalg.norm(self.linear_weights - prev_t) > self.EPS and iter < self.max_iter:
            prev_t = self.linear_weights.copy()
            z = self.forward()
            error = self.error(z)
            grad = self.grad(z)
            self.linear_weights -= self.step_size * grad
            if iter % self.report_iter == 0:
                print "Iter", iter
                print "f(x) = %.5f" % (error)
                print "Gradient Magnitude: ", np.linalg.norm(grad), "\n"
                self.f_t[int(iter//self.report_iter)] = error
            iter += 1
        self.f_t = self.f_t[:int(iter//self.report_iter)]
        np.save(self.db.image_dir.strip('/')+'_'+self.db.image_source.split('.')[0],self.linear_weights)


def make_data_set(act, set_id, size=100):
    all_images = os.listdir('cropped/')

    train_set = open('training_set%s.txt' % (set_id),'w+')
    validation_set = open('validation_set%s.txt' % (set_id),'w+')
    testing_set = open('testing_set%s.txt' % (set_id),'w+')
    count = np.zeros(len(act))
    for image in all_images:
        i = 0
        for actor in act:
            if image[0:4] == actor.split()[1].lower()[0:4]:
                if count[i]<size:
                    train_set.write(image+'\t'+actor.split()[1]+'\n')
                elif count[i]<size+10:
                    validation_set.write(image+'\t'+actor.split()[1]+'\n')
                elif count[i]<size+20:
                    testing_set.write(image+'\t'+actor.split()[1]+'\n')
                count[i] += 1
            i += 1
    train_set.close()
    validation_set.close()
    testing_set.close()