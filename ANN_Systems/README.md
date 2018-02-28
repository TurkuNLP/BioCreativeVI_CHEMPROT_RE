This code runs the I-ANN (or ST-ANN) ensemble:
- Trains on training set, predicts development and test set
- Trains on training set + development set, predicts test set 

Instruction for running the pipeline: 

1) Install all prerequisits.

2) Download the pre-trained Word2Vec model from: http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin into your hard drive. 
After downloading the file into hard drive, it will have a local address like : /home/user1/PubMed-and-PMC-w2v.bin

3) Open the CONFIG_ChemProt_11Class_1M.json file, and give the local address of the Word2Vec file in the "Model_Address" section. 

Example: 

     "W2V_Model" : {
           "Model_Address"    : "/home/user1/PubMed-and-PMC-w2v.bin" ,
           "MaxWordsInMemory" : 1000000
     },  

4) clone the TEES development branch into your hard drive: 
   git clone -b development https://github.com/jbjorne/TEES.git

5) open chemprot_tools.py file and give the absolute path of the TEES repository in third line of the source code: 
Example: 
   TEES_develBranch_folder  = "/home/user1/Desktop/PROJECTS/GIT/TEES_DEVELBranch/TEES/" 

6) Download CHEMPROT pre-processed data (tokenized, parsed, in TEES Interaction XML format) 
from https://www.dropbox.com/sh/yliclgkrkhyz6um/AABGx7taxC2VYs_I0VGyPm_ia?dl=0 
and unzip the content into the I-ANN/DATA/ folder. After copying files, the folder should have the following files: 
- NEW_CHEMPROT_PREPROCESSED_DATA.pickle
- BLLIP_BIO-SC-CCprocessed/CP17-train.xml
- BLLIP_BIO-SC-CCprocessed/CP17-devel.xml
- BLLIP_BIO-SC-CCprocessed/CP17-test.xml
- BLLIP_BIO-SC-CCprocessed/README_IMPORTANT_NEW.txt

7) - If you want to run the code on CPU: run the run_on_cpu.sh file.
   - If you want to run the code on GPU: run the run_on_gpu.sh file.

You have to run only one of those scripts, not both of them. 
The predictions will be put into the PREDICTIONS folder.
The aggregation script will put the aggregated results into the folder : PREDICTIONS_AGGREGATIONS
Log files will be put into the LOGS folder.

If you have more questions, contact: farmeh@utu.fi
