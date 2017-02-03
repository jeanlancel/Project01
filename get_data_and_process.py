from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from PIL import Image as image
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


#act = list(set([a.split("\t")[0] for a in open("facescrub_actors.txt").readlines()]))
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell','Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


testfile = urllib.URLopener()

# Note: you need to create the uncropped folder first in order
# for this to work

try:
    os.mkdir("uncropped/")
except:
    print 'Directory Exists, continueing'
crop_list = open("actors_list.txt", "w+")
for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("facescrub_actresses.txt"):
        if a in line:
            filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
            # A version without timeout (uncomment in case you need to
            # unsupress exceptions, which timeout() does)
            # testfile.retrieve(line.split()[4], "uncropped/"+filename)
            # timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/" + filename), {}, 30)
            if not os.path.isfile("uncropped/" + filename):
                continue
            crop_list.write(filename+'\t'+line.split('\t')[4]+'\n')
            print filename+'\t'+line.split('\t')[4]+'\n'
            #print filename
            i += 1

crop_list.close()

crop_list = open("actors_list.txt","r")
fromdir = 'uncropped/'
todir = 'cropped/'
try:
    os.mkdir(todir)
except:
    print 'Directory Exists, continuing'
for line in crop_list:
    line = line.strip('\n')
    imname = line.split('\t')[0]
    try:
        Im = image.open(fromdir+imname).convert('L')
    except:
        print "broken image " + imname
        continue
    box = [float(i) for i in line.split('\t')[1].split(',')]
    crop_im = Im.crop(box)
    crop_im = crop_im.resize((32,32))
    crop_im.save(todir+imname)
