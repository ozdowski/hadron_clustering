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



