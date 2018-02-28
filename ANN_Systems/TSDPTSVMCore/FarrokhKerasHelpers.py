# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:24:38 2016

@author: farmeh
"""

def GetLayerOutput(model, NI, layername, train_mode = False): 
    #function by farrokh for testing outputs ...
    from keras import backend as K
    LearningPhaseValue = 1 if train_mode else 0 # 0 when testing, 1 when training ...
    INPUTTENSORS = model.input if isinstance (model.input , list) else [model.input] ; 
    RetrieveLayerOutput_theanoFunction = K.function(INPUTTENSORS +[K.learning_phase()], [model.get_layer (layername).output]) 
    if not isinstance (NI , list):
        NI = [NI] ; 
    activations =  RetrieveLayerOutput_theanoFunction (NI + [LearningPhaseValue])
    return activations; 

def GetLayerOutput_idx(model, NI, layername, which_output_idx , train_mode = False): 
    #function by farrokh for testing outputs ...
    from keras import backend as K
    LearningPhaseValue = 1 if train_mode else 0 # 0 when testing, 1 when training ...
    INPUTTENSORS = model.input if isinstance (model.input , list) else [model.input] ; 
    RetrieveLayerOutput_theanoFunction = K.function(INPUTTENSORS +[K.learning_phase()], [model.get_layer (layername).get_output_at (which_output_idx)]) 
    if not isinstance (NI , list):
        NI = [NI] ; 
    activations =  RetrieveLayerOutput_theanoFunction (NI + [LearningPhaseValue])
    return activations; 
