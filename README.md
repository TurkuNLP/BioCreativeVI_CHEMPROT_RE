# BioCreativeVI_CHEMPROT_RE
This repository contains biomedical relation extraction systems capable of recognizing the statements of relations between chemical compounds/drugs and genes/proteins from biomedical literature. The code is developed for our participation in the BioCreative VI Task 5 (CHEMPROT) challenge. 

Contact: Farrokh Mehryary, farmeh@utu.fi

The code contains three main parts:
1) An SVM-based biomedical relation extraction system which relies on a rich set of features. 
The code/instructions for this system can be found here: https://github.com/jbjorne/TEES
2) Two deep learning-based biomedical relation extraction system (I-ANN and ST-ANN).
3) The script which combines the predictions of the SVM and I-ANN system: https://github.com/jbjorne/TEES/blob/development/Utils/Combine.py

You will need the following prerequisites to run the code in this git repository:
1) Python 2.7
2) Theano 0.9.0 library for python (See: http://deeplearning.net/software/theano/ )
3) Keras 2.0.6  library for python (See: https://keras.io/ )
4) NetworkX 1.11 library for python (See: https://networkx.github.io/documentation/networkx-1.10/overview.html )
5) scikit-learn library for python (See: http://scikit-learn.org/stable/ )
6) wvlib library for python (See: https://github.com/spyysalo/wvlib )
7) Pre-trained Word2Vec Model (Download from: http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin and See: http://bio.nlplab.org/)

The code can be fully executed on CPU. However, we recommend to run this code on GPU for faster execution. 
In this case, you will need CUDA libraries to be installed. 

Please cite our paper if you use (parts of) our code:<BR>
  
@Article{mehryary2018oxfordDB,
author = {Mehryary, Farrokh and Bj{\"{o}}rne, Jari and Salakoski, Tapio and Ginter, Filip},
title = {Potent pairing: ensemble of long short-term memory networks and support vector machine for chemical-protein relation extraction},
journal = {Database},
volume = {2018},
number = {},
pages = {bay120},
year = {2018},
doi = {10.1093/database/bay120},
URL = {http://dx.doi.org/10.1093/database/bay120},
eprint = {/oup/backfile/content_public/journal/database/2018/10.1093_database_bay120/2/bay120.pdf}
}
