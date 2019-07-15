# Hadronic Clustering

This project uses a Convolutional Neural Network to achieve hadronic clustering. One of the files places two pions (each 50GeV) a set distance apart from each other, while the other produces an analysis on a single pion shower. 

## Get Started

There is preparation required to run this code, including generating a data file by simulating a single 50 GeV pion gun. It is also necessary to make multiple installations in the lxplus workspace. 

### Producing the data file

Create the data files in lxplus using a CMSSW_10_2_4 area. Log into CMSSW2@princeton.edu and look in the directory /tigress/cgtully/cms/CMSSW_10_2_4/src. Use the files below: 
```
Driver: SinglePiPt50ieta23_pythia8_cfi_GEN_SIM.py
First Run: step2_DIGI_L1_HLT_DIGI2RAW.py 
Second Run: step3_RAW2DIGI_L1Reco_RECO_RECOSIM_EI_PAT.py
Make .tif files: pfhcal.py
```
Make sure to edit the names of the output files and the files that are read in for each of these scripts. Use cmsRun to run each script. You should produce multiple png 
