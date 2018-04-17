
# coding: utf-8

# In[10]:


import os  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import math


# In[11]:


PATH = "E:\Datasets\ADL_Dataset\HMP_Dataset\\"
NEWPATH = "E:\Datasets\ADL_Dataset\\"


# In[16]:


def Distance2dim(a,b):
    return pow(pow(a[1]-b[1],2)+pow(a[0]-b[0],2), 0.5)
def Cosin2vec(a,b):
    return (a[1]*b[1]+a[0]*b[0])/(pow(pow(a[1],2) + pow(a[0],2) , 0.5)*pow(pow(b[1],2) + pow(b[0],2) , 0.5)) 
def WeightAngle(a,b):
    return math.exp(5*(1.1 - Cosin2vec(a,b)))
def varRP(data, dim):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(150):
            x.append(data[j][0])
    elif dim == 'y':
        for j in range(150):
            x.append(data[j][1])
    elif dim == 'z':
        for j in range(150):
            x.append(data[j][2])
    
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    dimR = len(x)-1
    R = np.zeros((dimR,dimR))

    for i in range(dimR):
        for j in range(dimR):
            R[i][j] = WeightAngle(s[i],[1,0])*Distance2dim(s[i],s[j])
    return R
def RP(data, dim):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(150):
            x.append(data[j][0])
    elif dim == 'y':
        for j in range(150):
            x.append(data[j][1])
    elif dim == 'z':
        for j in range(150):
            x.append(data[j][2])
    
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    dimR = len(x)-1
    R = np.zeros((dimR,dimR))

    for i in range(dimR):
        for j in range(dimR):
            R[i][j] = Distance2dim(s[i],s[j])
    return R
def RemoveZero(l):
    nonZeroL = []
    for i in range(len(l)):
        if l[i] != 0.0:
            nonZeroL.append(l[i])
    return nonZeroL
#a = [0,-1,0.02,3]
#print RemoveZero(a)
def NormalizeMatrix(_r):
    dimR = _r.shape[0]
    h_max = []
    for i in range(dimR):
        h_max.append(max(_r[i]))
    _max =  max(h_max)
    h_min = []
    for i in range(dimR):
        h_min.append(min(RemoveZero(_r[i])))
    _min =  min(h_min)
    _max_min = _max - _min
    _normalizedRP = np.zeros((dimR,dimR))
    for i in range(dimR):
        for j in range(dimR):
            _normalizedRP[i][j] = _r[i][j]/_max_min
    return _normalizedRP
def RGBfromRPMatrix_of_XYZ(X,Y,Z):
    if X.shape != Y.shape or X.shape != Z.shape or Y.shape != Z.shape:
        print('XYZ should be in same shape!')
        return 0
    
    dimImage = X.shape[0]
    newImage = np.zeros((dimImage,dimImage,3))
    for i in range(dimImage):
        for j in range(dimImage):
            _pixel = []
            _pixel.append(X[i][j])
            _pixel.append(Y[i][j])
            _pixel.append(Z[i][j])
            newImage[i][j] = _pixel
    return newImage
def SaveRP(x_array,y_array,z_array):
    _r = RP(x_array)
    _g = RP(y_array)
    _b = RP(z_array)
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
    plt.imshow(newImage)
    plt.savefig('D:\Datasets\ADL_Dataset\\'+action+'\\'+'RP\\''{}{}.png' .format(action, subject[15:]),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
def SaveRP_XYZ(x,action, normalized):
    _r = RP(x,'x')
    _g = RP(x,'y')
    _b = RP(x,'z')
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+action+'\\RP\\'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+action+'\\RP\\'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')

def SavevarRP_XYZ(x,action, normalized):
    _r = varRP(x,'x')
    _g = varRP(x,'y')
    _b = varRP(x,'z')
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+action+'\\varRP\\'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+action+'\\varRP\\'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')


# In[17]:


actions = os.listdir(PATH)
for action in actions:
    segments = np.load(NEWPATH+action+'.npy')
    for i in range(len(segments)):
        SavevarRP_XYZ(segments[i], action, normalized = 1)

