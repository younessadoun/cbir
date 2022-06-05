import string

import cv2 as cv
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage import metrics
from skimage import io
#from google.colab.patches import cv2_imshow
import skimage.feature as feature
import pickle
import os
from scipy.spatial import distance




class index:
    n = None
    c = None
    c2= None
    s = None
    t = None
    p = None
    d = None

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

def extract_features(name):
    print(name)


    img=cv.imread(name)
    #print(img)

    # Calculate histogram without mask


    hist1 = cv.calcHist([img], [0], None, [256], [0, 256])

    #print(hist1)
    colmom = color_moments(name)

    lower = 0.66 * np.mean(img)
    upper = 1.33 * np.mean(img)
    edges = cv.Canny(img, lower, upper)

    # print(edges)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    graycom = feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

  #  contrast = feature.graycoprops(graycom, 'contrast')
  #  dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
  #  homogeneity = feature.graycoprops(graycom, 'homogeneity')
  #  energy = feature.graycoprops(graycom, 'energy')
  #  correlation = feature.graycoprops(graycom, 'correlation')
  #  ASM = feature.graycoprops(graycom, 'ASM')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)
    #kp = sift.detect(gray, None)



    feat=index
    feat.n = name
    feat.c = hist1
    feat.c2= colmom
    feat.s = edges
    feat.t = graycom
    feat.p = keypoints_1
    feat.d = descriptors_1

    return feat

def search(quer,reg):
    objects = []
    order = []
    #quer='images/test/1.2.826.0.1.3680043.8.498.11678170878548215953866689093584664340-c.png'
    querf=extract_features(quer)
    qc=querf.c
    qc2=querf.c2
    qs=querf.s
    qt=querf.t
    qkp=querf.p
    qd=querf.d
    with (open(reg+".dat", "rb")) as openfile:
        while True:
            try:
                tem=pickle.load(openfile)

                cd = metrics.hausdorff_distance(qc, tem.c)

                #print(qc2)
                #print(tem.c2)
                cd2=distance.euclidean(qc2,tem.c2)
                #cd2 = metrics.hausdorff_distance(qc, tem.c2)

                #sd = metrics.hausdorff_distance(qs, tem.s)
                sd=cv.matchShapes(qs,tem.s,1,0.0)*10000

                td = metrics.hausdorff_distance(qt, tem.t)

                imgc=cv.imread(tem.n)
                gray = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)

                sift = cv.SIFT_create()

                keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)
                #kpc = sift.detect(gray, None)
                bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
                matches = bf.match(descriptors_1, qd)
                kpd=0
                if len(descriptors_1)<=len(qd) :
                    kpd=100-((len(matches)*100)/len(descriptors_1))
                else:
                    kpd =100-( (len(matches) * 100) / len(qd))
             #   print(len(descriptors_1))
             #   print(len(qd))
             #   print(len(matches))
             #   matches = sorted(matches, key=lambda x: x.distance)
                #kpd = metrics.hausdorff_distance(qkp, tem.p)

                ttd = cd*0.5 + cd2*0.5 + sd*1 + td*0.1 + kpd*10
              #  ttd = sd

                o=(ttd,tem.n)

                order.append(o)

                objects.append(tem)
               # print(objects)
            except EOFError:
                break
        print("end")


    order.sort()
    resultlist=[None,None,None,None,None]
    for i in range(5):
        resultlist[i]=order[i][1]

    return resultlist

def emptyF(path):
    import os
    import glob

    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f):
            os.remove(f)



#w = 10
#h = 10
#fig = plt.figure(figsize=(8, 8))
#columns = 5
#rows = 3
#img = cv.imread(quer)
#fig.add_subplot(rows, columns, 3)
#plt.imshow(img)
#for i in range(1, columns*rows-4 ):
    #    img = cv.imread(order[i][1])
    #    fig.add_subplot(rows, columns, i+5)
#    plt.imshow(img)
#plt.show()
#img = cv.imread(quer)
#fig.add_subplot(rows, columns, 3)
#lower = 0.66 * np.mean(img)
#upper = 1.33 * np.mean(img)
#edges = cv.Canny(img, lower, upper)
#plt.imshow(edges)
#for i in range(1, columns*rows-4 ):
#    img = cv.imread(order[i][1])
#    lower = 0.66 * np.mean(img)
#    upper = 1.33 * np.mean(img)
#    edges = cv.Canny(img, lower, upper)
#    fig.add_subplot(rows, columns, i+5)
#    plt.imshow(edges)
#plt.show()



#print(len(objects))



#print(metrics.hausdorff_distance(edges, edges2))