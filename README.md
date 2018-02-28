# BioCreativeVI_CHEMPROT_RE
This repository contains biomedical relation extraction systems capable of recognizing the statements of relations between chemical compounds/drugs and genes/proteins from biomedical literature. The code is developed for our participation in the BioCreative VI Task 5 (CHEMPROT) challenge, and released under a noncommercial licence. Please see the LICENSE.md file. 

Contact: Farrokh Mehryary, farmeh@utu.fi

The code contains three main parts:
1) An SVM-based biomedical relation extraction system which relies on a rich set of features. 
The code/instructions for this system can be found here: https://github.com/jbjorne/TEES
2) A deep learning-based biomedical relation extraction system (I-ANN).
3) An script which combines the predictions of the SVM and I-ANN system. 

You will need the following prerequisites to run the code in this git repository:
1) Python 2.7
2) Theano 0.9.0 library for python (See: http://deeplearning.net/software/theano/ )
3) Keras 2.0.6  library for python (See: https://keras.io/ )
4) NetworkX 1.11 library for python (See: https://networkx.github.io/documentation/networkx-1.10/overview.html )
5) scikit-learn library for python (See: http://scikit-learn.org/stable/ )
6) wvlib library for python (See: https://github.com/spyysalo/wvlib )

The code can be fully executed on CPU. However, we recommend to run this code on GPU for faster execution. 
In this case, you will need CUDA libraries to be installed. 










