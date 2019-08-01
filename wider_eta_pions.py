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

imgs = tifffile.imread('wideretapion.tif').astype(np.float32)

#tests 
print(imgs.shape)
print(imgs[200,0,0,0])
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


#Visualize Data
bdys = tifffile.imread('wideretapion.tif')

bdys = (bdys > 0).astype(np.float32)

segs = connected_components(bdys)

imgs_trn = imgs[:20]
bdys_trn = bdys[:20]
segs_trn = segs[:20]

imgs_val = imgs[20:40]
bdys_val = bdys[20:40]
segs_val = segs[20:40]

print(imgs_trn.shape, imgs_val.shape)

# visualize data 
from matplotlib.colors import LogNorm
for idepth in range(6):
  plt.figure(figsize=(15,15))
  plt.subplot(131)
  plt.title("energy idepth="+str(idepth))
  plt.ylabel("ieta-16")
  plt.yticks([0,5,10,15])
#2D
  plt.imshow(imgs_trn[0,idepth], cmap='gray')
  plt.subplot(132)
  plt.title("true idepth="+str(idepth))
  plt.yticks([0,5,10,15])
#2D
  plt.imshow(bdys_trn[0,idepth], cmap='gray')
  plt.subplot(133)
  plt.title("seg idepth="+str(idepth))
  plt.yticks([0,5,10,15])
#2D
  plt.imshow(randomize_ids(segs_trn[0,idepth]), cmap='nipy_spectral')
  plt.grid(False)
  plt.show()

for idepth in range(6):
  plt.figure(figsize=(15,15))
  plt.subplot(131)
  plt.title("energy idepth="+str(idepth))
  plt.ylabel("ieta-16")
  plt.yticks([0,5,10,15])
  plt.xlabel("iphi")
#2D
  plt.imshow(imgs_val[0,idepth], cmap='gray')
  plt.subplot(132)
  plt.title("true idepth="+str(idepth))
  plt.yticks([0,5,10,15])
  plt.xlabel("iphi")
#2D
  plt.imshow(bdys_val[0,idepth], cmap='gray')
  plt.subplot(133)
  plt.title("seg idepth="+str(idepth))
  plt.yticks([0,5,10,15])
  plt.xlabel("iphi")
#2D
  plt.imshow(randomize_ids(segs_val[0,idepth]), cmap='nipy_spectral')
  plt.show()



maxima = []


#for c in range(imgs.shape[2]):
 # print("Hi")
  #if (c < 0):
   # continue

kmax = 13
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
          #if e1 and e2 and e3 and e4 and e4 and e5 and e6 and e7 and e8 and not(zero1 and zero2 and zero3 and zero4 and zero5 and zero6 and zero7 and zero8):
            #allmax.append([i,j,k,l])
          if e1 and e2 and e3 and e4 and e4 and e5 and e6 and e7 and e8:
            allmax.append([i,j,k,l])
    etaprime = 0
    phiprime = 0
    totale = 0
    totaleta = 0.0 
    totalphi = 0.0
    for b in range(len(allmax)):
      etaprime = allmax[b][2]
      phiprime = allmax[b][3]
      totale += imgs[i,j,allmax[b][2],allmax[b][3]]
      if etaprime == 0.0:
        etaprime = 1.3485
      if etaprime == 1.0:
        etaprime = 1.4355
      if etaprime == 2.0:
        etaprime = 1.5225
      if etaprime == 3.0:
        etaprime = 1.6095
      if etaprime == 4.0:
        etaprime = 1.6965
      if etaprime == 5.0:
        etaprime = 1.785
      if etaprime == 6.0:
        etaprime = 1.88
      if etaprime == 7.0:
        etaprime = 1.9865
      if etaprime == 8.0:
        etaprime = 2.1075
      if etaprime == 9.0:
        etaprime = 2.247
      if etaprime == 10.0:
        etaprime = 2.411
      if etaprime == 11.0:
        etaprime = 2.575
      if etaprime == 12.0:
        etaprime = 2.825
      if etaprime == 13.0:
        etaprime = 2.9085
      totaleta += imgs[i,j,allmax[b][2],allmax[b][3]]*etaprime
      degree = phiprime * 10.0
      phiprime = (degree * np.pi) / 180
      if phiprime > np.pi:
        phiprime = phiprime - (2*np.pi)
      totalphi += imgs[i,j,allmax[b][2],allmax[b][3]]*phiprime
    if totale != 0:
      totaleta = totaleta / totale
      totalphi = totalphi /totale
    if etaprime != 0 and phiprime != 0:
      maxima.append([i,j,etaprime,phiprime])
      
            
avgmax = []

for a in range(400):
  print("Entry "+str(a)+": "+str(maxima[a]))

for i in range(len(maxima)):
  eta = maxima[i][2]
  if eta >= 1.305 and eta <= 1.392:
    eta = 0
  if eta >= 1.392 and eta <= 1.479:
    eta = 1
  if eta >= 1.479 and eta <= 1.566:
    eta = 2
  if eta >= 1.566 and eta <= 1.653:
    eta = 3
  if eta >= 1.653 and eta <= 1.740:
    eta = 4
  if eta >= 1.740 and eta <= 1.830:
    eta = 5
  if eta >= 1.830 and eta <= 1.930:
    eta = 6
  if eta >= 1.930 and eta <= 2.043:
    eta = 7
  if eta >= 2.043 and eta <= 2.172:
    eta = 8
  if eta >= 2.172 and eta <= 2.322:
    eta = 9
  if eta >= 2.322 and eta <= 2.500:
    eta = 10
  if eta >= 2.500 and eta <= 2.650:
    eta = 11
  if eta >= 2.650 and eta <= 2.853:
    eta = 12
  if eta >= 2.853 and eta <= 3.000:
    eta = 13
  phi = maxima[i][3]
  phi = int(round((phi * 180) / (10 *np.pi),0))
  if phi < 0:
    phi = phi + 36
  avgmax.append([maxima[i][0],maxima[i][1],eta,phi])

for a in range(100):
  print("Entry "+str(a)+": "+str(avgmax[a]))

depth0 = 0
depth1 = 0 
depth2 = 0 
depth3 = 0 
depth4 = 0 
depth5 = 0

i0 = 0 
i1 = 0
i2 = 0
i3 = 0 
i4 = 0 
i5 = 0  

for i in range(len(avgmax)):
  if avgmax[i][1] == 0:
    depth0 += imgs[avgmax[i][0],avgmax[i][1],avgmax[i][2],avgmax[i][3]]
    i0 += 1
  if avgmax[i][1] == 1:
    depth1 += imgs[avgmax[i][0],avgmax[i][1],avgmax[i][2],avgmax[i][3]]
    i1 += 1
  if avgmax[i][1] == 2:
    depth2 += imgs[avgmax[i][0],avgmax[i][1],avgmax[i][2],avgmax[i][3]]
    i2 += 1
  if avgmax[i][1] == 3:
    depth3 += imgs[avgmax[i][0],avgmax[i][1],avgmax[i][2],avgmax[i][3]]
    i3 += 1
  if avgmax[i][1] == 4:
    depth4 += imgs[avgmax[i][0],avgmax[i][1],avgmax[i][2],avgmax[i][3]]
    i4 += 1
  if avgmax[i][1] == 5:
    depth5 += imgs[avgmax[i][0],avgmax[i][1],avgmax[i][2],avgmax[i][3]]
    i5 += 1

edepth0 = depth0/i0
edepth1 = depth1/i1
edepth2 = depth2/i2
edepth3 = depth3/i3
edepth4 = depth4/i4
edepth5 = depth4/i5

fullenergies = [edepth0,edepth1,edepth2,edepth3,edepth4,edepth5]

plt.title("Average Max Energy Per Depth")
plt.ylabel("Energy in depths")
plt.xlabel("Depths (0-5)")
plt.plot(depths,fullenergies)
plt.show()





  
