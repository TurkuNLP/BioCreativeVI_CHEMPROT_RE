# BioCreativeVI_CHEMPROT_RE
A deep learning-based biomedical relation extraction system for recognizing the statements of relations between chemical compounds/drugs and genes/proteins from biomedical literature. The code is developed for our participation in the BioCreative VI CHEMPROT challenge, and released under a noncommercial licence. Please see the LICENSE.md file. 

Contact: Farrokh Mehryary, farmeh@utu.fi

The code contains three main parts:
1) An SVM-based biomedical relation extraction system which relies on a rich set of feature. 
The code/instructions for this system can be found here: https://github.com/jbjorne/TEES

2) A deep learning-based biomedical relation extraction system (I-ANN).
The code/instructions for this system is inside this git repository. 

3) A code that integrates the predictions of the SVM and I-ANN system. 
The code/instructions for this system is inside this git repository. 

You will need the following prerequisites to run the code in this git repository:
1) Python 2.7
2) Theano 0.9.0 library for python (See: http://deeplearning.net/software/theano/ )
3) Keras 2.0.6  library for python (See: https://keras.io/ )
4) scikit-learn library for python 0.18.1 (See: http://scikit-learn.org/stable/ )





