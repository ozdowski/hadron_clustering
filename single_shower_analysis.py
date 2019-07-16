#########################################################
#INSTRUCTIONS FOR SETUP NECESSARY FOR THIS FILE         #
#########################################################
#Run single 50 GeV pion events:                         #
#log into cmssw2.princeton.edu and look in directory    #
#/tigress/cgtully/cms/CMSSW_10_2_4/src                  #
#Create a CMSSW_10_2_4 area                             #
#########################################################
#Driver for single pion samples:                        #
#SinglePiPt50ieta23_pythia8_cfi_GEN_SIM.py              #
#Runs:                                                  #
#step2_DIGI_L1_HLT_DIGI2RAW.py                          #
#step3_RAW2DIGI_L1Reco_RECO_RECOSIM_EI_PAT.py           #
#Finally, use pfhcal.py to make the .tif files          #
#***IMPORTANT: Use cmsRun to run these files!***        #
#***IMPORTANT: Change output file name in each file     #
#########################################################

# imports 
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np 
from skimage import measure 
import tifffile 


import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
matplotlib.rcParams['axes.grid'] = False

x = torch.ones((100,200))#.cuda()
#x = torch.ones((1,200))
#y = torch.ones((200,400000))
y = torch.ones((200,400000))#.cuda()

def getz(x, y):
  z = torch.mm(x,y)
  z = z.cpu().numpy()
  return z

# utils
def connected_components(boundaries):
#2D
#    assert(len(boundaries.shape) == 3)
    assert(len(boundaries.shape) == 4)
    boundaries = boundaries.astype(np.int64)
    ccs = np.zeros_like(boundaries)
    for z in range(ccs.shape[0]):
        ccs[z] = measure.label(boundaries[z])
        
    return ccs

def randomize_ids(segs, seed=None, lim=256):
    """Randomly change segment ids. Useful for visualization purposes"""
    np.random.seed(seed)
    
    segs = np.copy(segs).astype(np.int64)
    size = np.random.randint(lim,2*lim)
    remap = np.random.randint(1,lim+1,size)
    segs[segs != 0] = remap[segs[segs != 0] % size] # dont change 0s
    
    np.random.seed(None)

    return segs

imgs = tifffile.imread('mdepth.tif').astype(np.float32)

#tests 
print(imgs.shape)
print(imgs[200,3,2,20])
print(imgs.shape[0])

#create arrays for each dimension
depths = np.arange(imgs.shape[1])
phis = np.arange(imgs.shape[3])
etas = np.arange(imgs.shape[2])
energies = np.zeros(imgs.shape[1]).astype(np.float32)
events = np.arange(imgs.shape[0])
maxdepths = np.zeros(imgs.shape[0])
energyplot = np.zeros(imgs.shape[3])

#find the average maximum energy in each depth
for j in range (imgs.shape[1]):     #execute for each depth
  avg = 0
  tot = 0
  for i in range(imgs.shape[0]):       #scan over events
    emax = -1.0
    ehere = -1.0
    maxiphi = 0
    maxieta = 0
    for k in range(imgs.shape[2]):     #scan over eta
      for l in range(imgs.shape[3]):   #scan over phi
        ehere = imgs[i,j,k,l]
        if ehere > emax:
          emax = ehere
          maxieta = k
          maxiphi = l
    tot += emax
  avg = tot/(imgs.shape[0])
  energies[j] = avg
  print(str(avg)+",("+str(maxieta)+","+str(maxiphi)+")")

#find the depth where the energy is maxed in each event 
for a in range(imgs.shape[0]):
  maxdepth = 0
  eventmax = -1.0
  eventhere = 0
  for b in range(imgs.shape[1]):
    for c in range(imgs.shape[2]):
      for d in range(imgs.shape[3]):
        eventhere = imgs[a,b,c,d]
        if eventhere > eventmax:
          eventmax = eventhere
          deltamax = b
  #print(deltamax)
  maxdepths[a] = deltamax
  #print(maxdepths[a])

#examine the energies as you rotate in phi  
for n in range(imgs.shape[3]):
  energyplot[n] = imgs[200,1,2,n]
  
smallerscan = energyplot[10:36]
smallerphi = phis[10:36]
    
#Max Energy per Depth Plot
plt.title("Depth and Max Energy")
plt.xlabel("Depth (0-5)")
plt.ylabel("Max Energy")
plt.plot(depths,energies)
plt.show()  

#Where is max energy histogram 
plt.title("Depth of Max Energy")
plt.xlabel("Depth (0-5)")
plt.ylabel("Number of Events")
plt.hist(maxdepths,(0,1,2,3,4,5,6), edgecolor='black',align="left")
plt.show()  

#plot of energies through phi, center eta
plt.title("Energies vs Phi")
plt.xlabel("Phi")
plt.ylabel("Energies")
plt.plot(phis,energyplot)
plt.show() 

#Take a closer look at the phi plot
plt.title("energies vs phi")
plt.xlabel("phi")
plt.ylabel("energies")
plt.plot(smallerphi,smallerscan)
plt.show() 

maxima = []


#for c in range(imgs.shape[2]):
 # print("Hi")
  #if (c < 0):
   # continue

kmax = 4
kmin = 0
lmax = 35
lmin = 0
    
#loop over events
for i in range(imgs.shape[0]):
  #loop over depths
  for j in range(imgs.shape[1]):
    allmax = []
    #loop over ieta
    for k in range(imgs.shape[2]):
        #loop over iphi 
        for l in range(imgs.shape[3]):
          #examine energy at k, l
          ehere = imgs[i, j, k, l]
          emax = -1.0
          #Booleans to check whether ehere is greater than surrounding pixels
          e1 = False
          e2 = False
          e3 = False
          e4 = False
          e5 = False
          e6 = False
          e7 = False
          e8 = False
          #Booleans to check whether surrounding pixels are zero
          zero1 = False
          zero2 = False
          zero3 = False
          zero4 = False
          zero5 = False
          zero6 = False
          zero7 = False
          zero8 = False
          #Top left
          if k-1 < kmax and k-1 > kmin and l+1 < lmax and l+1 > lmin:
            if ehere > imgs[i,j,k-1,l+1]:
              e1 = True
            #check for zero
            if imgs[i,j,k-1,l+1] == 0:
              zero1 = True
          #edge condition
          if k-1 > kmax or k-1 < kmin or l+1 > lmax or l+1 < lmin:
            e1 = True
            zero1 = True
          #Top
          if k < kmax and k > kmin and l+1 < lmax and l+1 > lmin:
            if ehere > imgs[i,j,k,l+1]:
              e2 = True
            #check for zero
            if imgs[i,j,k,l+1] == 0:
              zero2 = True
          #Edge condition
          if k > kmax or k < kmin or l+1 > lmax or l+1 < lmin:
            e2 = True 
            zero2 = True
          #Top right
          if k+1 < kmax and k+1 > kmin and l+1 < lmax and l+1 > lmin:
            if ehere > imgs[i,j,k+1,l+1]:
              e3 = True
            #check for zero
            if imgs[i,j,k+1,l+1] == 0:
              zero3 = True
          #Edge condition
          if k+1 > kmax or k+1 < kmin or l+1 > lmax or l+1 < lmin:
            e3 = True
            zero3 = True
          #Middle left
          if k-1 < kmax and k-1 > kmin and l < lmax and l > lmin:
            if ehere > imgs[i,j,k-1,l]:
              e4 = True
            #check for zero
            if imgs[i,j,k-1,l] == 0:
              zero4 = True
          #Edge Condition
          if k-1 > kmax or k-1 < kmin or l > lmax or l < lmin:
            e4 = True
            zero4 = True
          #Middle right
          if k+1 < kmax and k+1 > kmin and l < lmax and l > lmin:
            if ehere > imgs[i,j,k+1,l]:
              e5 = True
            #check for zero
            if imgs[i,j,k+1,l+1] == 0:
              zero5 = True
          #Edge Condition
          if k+1 > kmax or k+1 < kmin or l > lmax or l < lmin:
            e5 = True
            zero5 = True
          #Bottom left
          if k-1 < kmax and k-1 > kmin and l-1 < lmax and l-1 > lmin:
            if ehere > imgs[i,j,k-1,l-1]:
              e6 = True
            #check for zero
            if imgs[i,j,k-1,l-1] == 0:
              zero6 = True
          #Edge Condition
          if k+1 > kmax or k+1 < kmin or l > lmax or l < lmin:
            e6 = True
            zero6 = True
          #Bottom 
          if k < kmax and k > kmin and l-1 < lmax and l-1 > lmin:
            if ehere > imgs[i,j,k,l-1]:
              e7 = True
            #check for zero
            if imgs[i,j,k,l-1] == 0:
              zero7 = True
          #Edge Condition
          if k > kmax or k < kmin or l > lmax or l < lmin:
            e7 = True
            zero7= True
          #Bottom Right
          if k+1 < kmax and k+1 > kmin and l-1 < lmax and l-1 > lmin:
            if ehere > imgs[i,j,k+1,l-1]:
              e8 = True
            #check for zero
            if imgs[i,j,k+1,l-1] == 0:
              zero8 = True
          #Edge Condition
          if k+1 > kmax or k+1 < kmin or l > lmax or l < lmin:
            e8 = True
            zero8 = True
          if e1 and e2 and e3 and e4 and e4 and e5 and e6 and e7 and e8 and not(zero1 and zero2 and zero3 and zero4 and zero5 and zero6 and zero7 and zero8):
            allmax.append([i,j,k,l])
    etaprime = 0
    phiprime = 0
    totale = 0
    for b in range(len(allmax)):
      etaprime += imgs[i,j,allmax[b][2],allmax[b][3]]*allmax[b][2]
      phiprime += imgs[i,j,allmax[b][2],allmax[b][3]]*allmax[b][3]
      totale += imgs[i,j,allmax[b][2],allmax[b][3]]
    if totale != 0:
      etaprime = etaprime / totale
      phiprime = phiprime /totale
    if etaprime != 0 and phiprime != 0:
      maxima.append([i,j,etaprime,phiprime])
      
            
            
            
roundedmax = []    

avgmax = []

for a in range(len(maxima)):
  i = maxima[a][0]
  j = maxima[a][1]
  k = maxima[a][2]
  l = maxima[a][3]
  k = round(k)
  l = round(l)
  roundedmax.append([i,j,k,l])

for a in range(400):
  print("Entry "+str(a)+": "+str(roundedmax[a]))


