# MIX-TPI
MIX-TPI: A flexible prediction framework for TCR-pMHC interactions based on multimodal representations
# Data description
Datasets used in this work can be downloaded as follow:
* VDJdb-TITAN and Immune-TITAN are downloaded from [TITAN](https://github.com/PaccMann/TITAN)
* VDJdb-ImRex, McPAS-TCRs, and McPAS-peptides are downloaded from [ImRex](https://github.com/pmoris/ImRex)
* IEDB-NetTCR is downloaded from [NetTCR2.0](https://github.com/mnielLab/NetTCR-2.0)  
  
train.csv, train+covid.csv in folder './data/TITAN/ten_fold/tcr_split/fold0' denote the VDJdb-TITAN and Immune-TITAN after splitting, respectively.
# Folder ./physicochemical_properties
This package is downloaded from [ImRex](https://github.com/pmoris/ImRex) and is used to initialize physicochemical features.
# Run
Run main.py to train the model using 10-fold cross-validation. The package 'physicochemical_properties' can be downloaded from [ImRex](https://github.com/pmoris/ImRex) and is used to initialize physicochemical features.
# Requirements
* tqdm 4.63.0
* pandas 1.3.5
* numpy 1.22.0
* transformers 4.17.0
* torch 1.10.0
* scikit-learn 1.0.2
* biopython 1.79
* pyteomics 4.5.3