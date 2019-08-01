# Hadronic Clustering

This project uses a Convolutional Neural Network to achieve hadronic clustering. two\_single\_50GeV\_pions.py places two pions (each 50GeV) a set distance apart. Both the single\_shower\_shower\_analysis.py and the wider\_eta\_pions.py files perform an analysis on a single pion, with the wider\_eta\_pions.py file analyzing a wider range of eta that includes the entire HE.   

## Get Started

There is preparation required to run this code, including generating a data file by simulating a single 50 GeV pion gun. It is also necessary to make multiple installations in the lxplus workspace. 

### Producing the Data File

Create the data files in lxplus using a CMSSW_10_2_4 area. Log into CMSSW2@princeton.edu and look in the directory /tigress/cgtully/cms/CMSSW_10_2_4/src. Use the files below: 
```
Driver: SinglePiPt50ieta23_pythia8_cfi_GEN_SIM.py
First Run: step2_DIGI_L1_HLT_DIGI2RAW.py 
Second Run: step3_RAW2DIGI_L1Reco_RECO_RECOSIM_EI_PAT.py
Make .tif files: pfhcal.py
```
Make sure to edit the names of the output files and the files that are read in for each of these scripts. Use cmsRun to run each script. You should produce multiple png files and one .tif file. After producing the .tif file, store it onto a url. Access the url in terminal using the command !wget --no-check-certificate "https://your.url.here.edu/"

### Installations to Make

Execute the following commands to make the necessary installations to run the code. In your work/ directory (/afs/cern.ch/work/) execute the following commands:
```
pip install torch --user
pip install numpy --user --upgrade
pip install torchvision --user --upgrade
pip install --upgrade pip —user
~/.local/bin/pip install --user 'scikit-image<0.15'
pip install tifffile --user
```
It may also be a good idea to make sure you have a large/maximum afs quota, as these installations take up a significant amount of space. 

### Check Whether the Setup Worked 
Try running:
```
python run.py
```
If the file runs successfully with no errors or outputs, all installations were most likely successfull and setup is complete. 

### Outputs

Run correctly, two\_single\_50GeV\_pions.py should output multiple plots of the shower profiles, as well as plots depicting the values of each cell in ieta and iphi for every depth for a selected event. 

