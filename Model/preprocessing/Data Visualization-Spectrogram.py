
# coding: utf-8

# In[2]:


import os  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams


# In[3]:


PATH = "D:\Datasets\MotionData\\"
NEWPATH = "D:\Datasets\MotionData_Mag\\"
SPECTPATH = "D:\Datasets\MotionData_Spectrogram\\"
actionTags = ['0Walking','1Sitting','2Standing','3Squating','4Lying']


# In[4]:


def GetLeftRightCode(path):
    _lrcode = []
    lrcode = []
    for line in open(path,'r'):
        _lrcode.append(line)
    for i in range(len(_lrcode)):
        lrcode.append(_lrcode[i][7:-2])
    return lrcode
#print GetLeftRightCode("D:\Datasets\MotionData\MotionDataB\wb_pos_info.txt")


# In[5]:


def AverageFilter(l, windowsize):#list windowsize default is 3,if change needs to change weights and N
    _l = []
    N = (windowsize-1)/2
    for i in range(len(l)):
        if i >= N and i <= len(l)-1-N:
            _l.append(np.average(l[i-N:i+N+1],weights=[1 for j in range(windowsize)]))
        else:
            _l.append(l[i])
    return _l


# In[6]:


def DownSampling(x, rate):
    downsampling_x = []
    for i in range(len(x)):
        if i%rate == rate-1:
            downsampling_x.append(x[i])
    return downsampling_x


# In[9]:


def SaveSpectrogramFig(motion_data, subject, num, lrcode, action):
    l_ing_data_x = []
    r_ing_data_x = []
    l_ing_data_y = []
    r_ing_data_y = []
    l_ing_data_z = []
    r_ing_data_z = []
    for i in range(data.shape[0]):
         if motion_data[i][6] == action[1:]:
                if motion_data[i][2] == lrcode[0]: 
                    l_ing_data_x.append(motion_data[i][3])
                    l_ing_data_y.append(motion_data[i][4])
                    l_ing_data_z.append(motion_data[i][5])
                elif motion_data[i][2] == lrcode[1]:
                    r_ing_data_x.append(motion_data[i][3])
                    r_ing_data_y.append(motion_data[i][4])
                    r_ing_data_z.append(motion_data[i][5])
    print("Subject: %s Num: %s Action: %s Left: %d Right: %d" % (subject[-1], num[-5],action, len(l_ing_data_x), len(r_ing_data_x)))
    xyz = ['x','y','z']
    
    if len(l_ing_data_x) > 130:
        for j in range(1):    
            f1 = plt.figure(1)
            NFFT = 128       # the length of the windowing segments, len must be longer than 128
            Fs = int(52)  # the sampling frequency
            dimT = 31#(len-NFFT)/(NFFT-overlap)+1=dimT
            noverlap = NFFT - (len(l_ing_data_x)-NFFT)/(dimT-1)
            if noverlap >= NFFT:
                noverlap = 127
            Pxx, freqs, bins, im = plt.specgram(l_ing_data_x-np.mean(l_ing_data_x), NFFT = NFFT, Fs = Fs, noverlap = noverlap)
            plt.axis('off')
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            #plt.show()
            plt.savefig(SPECTPATH+action[1:]+'\\'+'{}{}{}{}{}.png' .format(subject[-1], num[-5], 'Left', xyz[j], action),bbox_inches='tight',pad_inches = 0)
            plt.close('all')
    
    if len(r_ing_data_x) > 130:
        for j in range(1):    
            f1 = plt.figure(1)
            NFFT = 128       # the length of the windowing segments, len must be longer than 128
            Fs = int(52)  # the sampling frequency
            dimT = 31#(len-NFFT)/(NFFT-overlap)+1=dimT
            noverlap = NFFT - (len(r_ing_data_x)-NFFT)/(dimT-1)
            if noverlap >= NFFT:
                noverlap = 127
            Pxx, freqs, bins, im = plt.specgram(r_ing_data_x-np.mean(r_ing_data_x),  NFFT = NFFT, Fs = Fs, noverlap = noverlap)
            plt.axis('off')
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            #plt.show()
            plt.savefig(SPECTPATH+action[1:]+'\\'+'{}{}{}{}{}.png' .format(subject[-1], num[-5], 'Left', xyz[j], action), bbox_inches='tight',pad_inches = 0)
            plt.close('all')
    
    return 0


# In[12]:


subjects = os.listdir(PATH)


# In[ ]:


l_len_total = {x[1:] : [] for x in actionTags}
r_len_total = {x[1:] : [] for x in actionTags}


# In[ ]:


# save figs
for subject in subjects:
    nums = os.listdir(PATH+subject+'\\')
    for num in nums:
        if num[0] =='L':
            csvPATH = PATH+subject+'\\'+'LabelledData{}{}.csv'.format(subject[-1], num[-5])
            data = pd.read_csv(csvPATH)
            lrcode = GetLeftRightCode(PATH+subject+'\\'+'wb_pos_info.txt')
            motion_data = []# 0 time w_id x y z activies
            for i in range(data.shape[0]):
                motion_data.append(map(lambda x : x, data.iloc[i]))
            for action in actionTags:
                #SaveAveragedFig(motion_data, subject, num, lrcode, action)
                #SaveMagFig(motion_data, subject, num, lrcode, action)
                #SaveSpectrogramFig(motion_data, subject, num, lrcode, action)
                l_len, r_len = SaveDataLen(motion_data, subject, num, lrcode, action)
                _l_len_total = l_len_total[action[1:]]
                _r_len_total = r_len_total[action[1:]]
                _l_len_total.append(l_len)
                _r_len_total.append(r_len)
                l_len_total[action] = _l_len_total
                r_len_total[action] = _r_len_total


# In[21]:


segments = np.load('D:\Datasets\\' + 'rawX3dim_longerthan256.npy')
labels = np.load('D:\Datasets\\' + 'rawY3dim_longerthan256.npy')
print segments.shape


# In[22]:


segmentsSitting = []
nSitting = 0
for i in range(len(labels)):
    if labels[i] == 'Sitting':
        segmentsSitting.append(segments[i])
        if len(segments[i]) > 256:
            nSitting += 1
print nSitting
segmentsStanding = []
nSitting = 0
for i in range(len(labels)):
    if labels[i] == 'Standing':
        segmentsStanding.append(segments[i])
        if len(segments[i]) > 256:
            nSitting += 1
print nSitting
segmentsLying = []
nLying = 0
for i in range(len(labels)):
    if labels[i] == 'Lying':
        segmentsLying.append(segments[i])
        if len(segments[i]) > 256:
            nLying += 1
print nLying

segmentsSquating = []
nSquating = 0
for i in range(len(labels)):
    if labels[i] == 'Squating':
        segmentsSquating.append(segments[i])
        if len(segments[i]) >= 256:
            nSquating += 1
print nSquating

segmentsWalking = []
nWalking = 0
for i in range(len(labels)):
    if labels[i] == 'Walking':
        segmentsWalking.append(segments[i])
        if len(segments[i]) > 256:
            nWalking += 1
print nWalking

