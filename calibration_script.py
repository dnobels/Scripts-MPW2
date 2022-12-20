#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import scipy
import scipy.stats
from scipy.optimize import curve_fit
from pylab import exp
from scipy import optimize
from math import sqrt
import glob
import seaborn as sns
import h5py   
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import interp1d


# In[2]:


def deltat(time,ch1,ch2):
    half1= max(ch1)/2
    half2= max(ch2)/2
    i=0
    while ch1[i]<half1:
        i=i+1
    j=0
    while ch2[j]<half2:
        j=j+1
    t1= time[i]
    t2= time[j]
    dt= time[j]-time[i]
    return dt, t1 ,t2

def tot(time, ch):
    half= (max(ch)-min(ch))/2 + min(ch) #find half point of the rise
    i=0
    while ch[i]<half: #find the element which is higher then the halfpoint
        i=i+1
    x1= [time[i-1],time[i]]
    y1= [ch[i-1],ch[i]]
    xnew1= np.linspace(time[i-1],time[i],200)
    f1 = interp1d(x1, y1)#linearly interpolate of these two points
    ynew1= f1(xnew1)
    k=0
    while ynew1[k]<half:
        k=k+1#find the right point of the interpolation
    j=i+30 #no do the same for the falling edge
    while ch[j]>half:
        j=j+1
    
    x2= [time[j-1],time[j]]
    y2= [ch[j-1],ch[j]]
    xnew2= np.linspace(time[j-1],time[j],200)
    f2 = interp1d(x2, y2)
    ynew2= f2(xnew2)
    z=0
    while ynew2[z]<half:
        z=z+1
    tott= xnew2[z]- xnew1[k]#calculate the tot
    return tott


# In[3]:


a=glob.glob("/data/detrd/ukraemer/osci_data/cal_scan_P0_59V_P16_6ms_W50us_B100V/*")

a= sorted(a)


# In[7]:


pixelnames=[]
temp=[]
for i in range(len(a)):
    if i==0:
        temp.append(a[i])
    elif a[i]==a[-1]:
        temp.append(a[i])
        pixelnames.append(temp)
    elif a[i][130:136]==a[i-1][130:136]:
        temp.append(a[i])
    else:
        pixelnames.append(temp)
        temp=[]
        temp.append(a[i])


# In[9]:


pixelsdt=[]
pixelstot=[]
time=np.array(dset["Time"])

for i in range(len(pixelnames)):
    temp2=[]
    temp3=[]
    for j in range(len(pixelnames[i])):
        temp1=h5py.File(pixelnames[i][j])
        ch1= temp1["Voltage_CHAN1"]
        ch1= np.array(ch1)
        ch2= temp1["Voltage_CHAN2"]
        ch2= np.array(ch2)
        if max(ch2)<0.1:
            empty1.append(max(ch2))
        else:
            dt,t2,t3 =deltat(time,ch1,ch2)
            temp2.append(dt)
            temp3.append(tot(time,ch2))
        temp1.close()
    pixelsdt.append(temp2)
    pixelstot.append(temp3)


# In[10]:


datafile=[]

for i in range(len(pixelstot)):
    temp=[]
    for j in range(len(pixelstot[i])):
        temp.append(pixelstot[i][j])
    datafile.append(temp)
    
datafile= np.transpose(datafile)

np.savetxt("cal_scan_P0_59V_P16_6ms_W50us_B100V_tot",datafile, fmt='%s')


# In[11]:


datafile1=[]

for i in range(len(pixelsdt)):
    temp=[]
    for j in range(len(pixelsdt[i])):
        temp.append(pixelsdt[i][j])
    datafile1.append(temp)
    
datafile1= np.transpose(datafile1)

np.savetxt("cal_scan_P0_59V_P16_6ms_W50us_B100V_deltat",datafile1, fmt='%s')


# In[ ]:




