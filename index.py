import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import warnings
def noop(*args, **kargs): pass
warnings.warn = noop
#with warnings.catch_warnings():
    #warnings.simplefilter('ignore', category=DeprecationWarning)
import skimage
from skimage import metrics
from skimage import io
import skimage.feature as feature
import pickle
import os



class index:
    def __init__(self, name, color, color2, shape, texture, point):
        self.n = name
        self.c = color
        self.c2 = color2
        self.s = shape
        self.t = texture
        self.p = point

def color_moments(filename):
    img = cv.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    #color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    #color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    #color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    color_feature=[h_mean, s_mean, v_mean, h_std, s_std, v_std, h_thirdMoment, s_thirdMoment, v_thirdMoment]
    return color_feature

def create_index(name,indloc):
    print(name)



    img=cv.imread(name)
    #print(img)

    # Calculate histogram without mask

    hist1 = cv.calcHist([img], [0], None, [256], [0, 256])

    #print(hist1)
    colmom=color_moments(name)

    lower = 0.66 * np.mean(img)
    upper = 1.33 * np.mean(img)
    edges = cv.Canny(img, lower, upper)

    # print(edges)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    graycom = skimage.feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

  #  contrast = feature.graycoprops(graycom, 'contrast')
  #  dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
  #  homogeneity = feature.graycoprops(graycom, 'homogeneity')
  #  energy = feature.graycoprops(graycom, 'energy')
  #  correlation = feature.graycoprops(graycom, 'correlation')
  #  ASM = feature.graycoprops(graycom, 'ASM')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)

    kplist = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        kplist.append(temp)


    #colmom=None
    #kplist=None
    fff=index(name,hist1,colmom,edges,graycom,kplist)





   # fff.n = name
   # fff.c = hist1
   # fff.s = edges
   # fff.t = graycom
   # fff.p = kp

    pickle.dump(fff, open(indloc, "ab"))

    print("success "+name)


import os
#os.chdir('Covid19-dataset/test/Covid/')
ff='torso'
for file in os.listdir('static/dataset/'+ff+'/'):
    file_path = f"{'static/dataset/'+ff+'/'}{file}"
    #print(file_path)
    create_index(file_path,ff+'.dat')

#objects = []
#with (open("variableStoringFile.dat", "rb")) as openfile:
#    while True:
#        try:
#            objects.append(pickle.load(openfile))
           # print(objects)
#        except EOFError:
#            break

#print(objects[2].n)
#print(objects[3].n)