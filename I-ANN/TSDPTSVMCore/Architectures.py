def KERASHelper_Multiply_Weights (x,**arguments):
    """
    Farrokh:
    We have X LSTMs/CNNs (i.e., Processing Layers, or PL), each for a path. 
    EACH path has a weight PER EACH Example. 
    Weights are inside sp_weights for example.
    We would like to weights to be multiplied to the lstm_i correctly. 
    
    Example with 2 LSTM layers, each processing and returning 3 examples. 
    Assume:
    >>> lstm1_output = np.array ([ [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] ])
    >>> lstm2_output = np.array ([ [4,4,4,4,4] , [5,5,5,5,5] , [6,6,6,6,6] ])
    >>> sp_weights = np.array ([[1,2],[3,4],[5,6]])
    >>> lstm_index = 0  --> let's focus on lstm1 
    >>> weights = np.tile(np.expand_dims(sp_weights[:,lstm_index],1),lstm1_output.shape[1])
    array([[1, 1, 1, 1, 1],    ---> weight multipliers for example #1  (for first lstm)
           [3, 3, 3, 3, 3],    ---> weight multipliers for example #2  (for first lstm)
           [5, 5, 5, 5, 5]])   ---> weight multipliers for example #3  (for first lstm)
    >>> weights*lstm1_output
    array([[ 1,  1,  1,  1,  1],
           [ 6,  6,  6,  6,  6],
           [15, 15, 15, 15, 15]])
    """
    from keras import backend as K
    lstm , sp_weights = x
    w1 = sp_weights[:,arguments["PL_index"]]
    w2 = K.expand_dims (w1,1)
    w3 = K.tile(w2, lstm.shape[1])
    return lstm*w3

def KERASHelper_NonZeros_Average(x):
    from keras import backend as K
    lstm_outs , weights = x[0:-1], x[-1]
    if len(lstm_outs)==1:
        return lstm_outs
        
    SUM = K.zeros_like(lstm_outs[0])
    for i in range(len(lstm_outs)):
        SUM+= lstm_outs[i]
    
    VALID_COUNT = K.expand_dims(K.sum(K.sign(weights),axis=1),1)
    return SUM/VALID_COUNT

def KERASHelper_Weighted_average(x):
    from keras import backend as K
    lstm_outs , weights = x[0:-1], x[-1]
    if len(lstm_outs)==1:
        return lstm_outs
        
    SUM = K.zeros_like(lstm_outs[0])
    for i in range(len(lstm_outs)):
        SUM+= lstm_outs[i]
    
    #VALID_COUNT = K.expand_dims(K.sum(K.sign(weights),axis=1),1)
    X = K.tile(K.expand_dims(K.sum(weights,axis=-1),1),lstm_outs[0].shape[1])
    return SUM/X

class ANN_Architecture_Builder:
    def __init__ (self, MaxSPCount, MaxSPLength, MaxFSLength, PRJ, RandomSeed=None):
        self.MaxSPCount  = MaxSPCount
        self.MaxSPLength = MaxSPLength
        self.MaxFSLength = MaxFSLength
        self.PRJ = PRJ 
        
        self.wv = PRJ.wv
        self.FeatureMappingDictionary = PRJ.FeatureMappingDictionary
        self.POSTagsEmbeddings        = PRJ.POSTagsEmbeddings
        self.DPTypesEmbeddings        = PRJ.DPTypesEmbeddings
        
        self.lp = PRJ.lp  
        self.PROGRAM_Halt = PRJ.PROGRAM_Halt
        self.RandomSeed = RandomSeed if RandomSeed <> None else 1337 ; 

        self.WhichFeaturesToUse = {}
        self.WhichOutputsToPredit = {} 
        
        self.WhichFeaturesToUse["entitytypes"] = len(PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"]) > 1
        self.WhichFeaturesToUse["weights"]     = False
        self.WhichFeaturesToUse["words"]       = False
        self.WhichFeaturesToUse["postags"]     = False 
        self.WhichFeaturesToUse["dptypes"]     = False 
        self.WhichFeaturesToUse["fs_words"]    = False
        self.WhichFeaturesToUse["fs_postags"]  = False
        self.WhichFeaturesToUse["fs_chunks"]   = False
        self.WhichFeaturesToUse["fs_p1_dists"] = False
        self.WhichFeaturesToUse["fs_p2_dists"] = False

        self.WhichOutputsToPredit["Y"] = True 
    
    def Arch_TopKP_lstm_w (self,lod,l_actvfunc,dod,d_actvfunc,dr,aggrtp):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"] = True
        self.WhichFeaturesToUse["words"]   = True

        H_lstm_out_dim     = lod 
        H_lstm_actv_func   = l_actvfunc.lower() if len(l_actvfunc)>=3 else None #None:default value will be used
        H_dense_out_dim    = dod 
        H_dence_actv_func  = d_actvfunc.lower() if len(d_actvfunc)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        H_aggregation_type = aggrtp.lower()
        
        if not H_aggregation_type in ('sum','avg'):
            self.PROGRAM_Halt("aggegation of N LSTMs should be either sum or avg!")
        
        H_wordEmbd_vocab_size     = self.wv.shape()[0]
        H_wordEmbd_embedding_size = self.wv.shape()[1]
        H_wordEmbd_weights        = self.wv.get_normalized_weights(deepcopy=True)
        
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #1) BASIC INPUTS: Entity-Types
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            L_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (L_entitytypes)
            
        #2) BASIC INPUTS: SP-Weights
        L_sp_weights = keras.layers.Input (shape=(self.MaxSPCount,),name="inp_sp_weights")
        MODEL_Inputs.append (L_sp_weights)
        
        #3)WORDS EMBEDDINGS and LSTMs
        L_shared_embd_words = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size, output_dim=H_wordEmbd_embedding_size, input_length=self.MaxSPLength , weights=[H_wordEmbd_weights], mask_zero=True , trainable=True, name = "shared_embd_words")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_words = keras.layers.LSTM(units=H_lstm_out_dim,name="shared_lstm_words")
        else:
            L_shared_lstm_words = keras.layers.LSTM(units=H_lstm_out_dim,name="shared_lstm_words", activation=H_lstm_actv_func)
            
        WORDS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_words_"+str(path_index))
            embd   = L_shared_embd_words(inp)
            lstm   = L_shared_lstm_words(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_words_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            WORDS_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        if H_aggregation_type == "avg":
            L_word_lstms_aggr = keras.layers.Lambda(KERASHelper_NonZeros_Average,output_shape=(H_lstm_out_dim,), name="aggr_lstm_words_avg")(WORDS_LSTMS_Outputs+[L_sp_weights])
        else:
            L_word_lstms_aggr = keras.layers.add(WORDS_LSTMS_Outputs, name="aggr_lstm_words_sum")
        
        #4) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_entitytypes])
        else:
            L_all_features = L_word_lstms_aggr
        
        #4) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dence_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit
        
        
    def Arch_TopKP_3lstms_wpd (self,wlod,plod,dlod,l_actvfunc,pembds,dtembds,dod,d_actvfunc,dr,aggrtp):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"] = True
        self.WhichFeaturesToUse["words"]   = True
        self.WhichFeaturesToUse["postags"] = True
        self.WhichFeaturesToUse["dptypes"] = True
        
        #LSTMs Params 
        H_W_lstm_out_dim     = wlod #Words-LSTM Output Dimensionality
        H_P_lstm_out_dim     = plod #POSTags-LSTM Output Dimensionality
        H_D_lstm_out_dim     = dlod #DTypes-LSTM Output Dimensionality
        H_lstm_actv_func     = l_actvfunc.lower() if len(l_actvfunc)>=3 else None #Activation function for all 3 LSTMs. None:default value will be used
        
        #Aggregation Param
        H_aggregation_type = aggrtp.lower()
        if not H_aggregation_type in ('sum','avg'):
            self.PROGRAM_Halt("aggegation of N LSTMs should be either sum or avg!")
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_dptpsEmbd_vocab_size     = self.DPTypesEmbeddings.len()+1
        H_dptpsEmbd_embedding_size = dtembds

        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = d_actvfunc.lower() if len(d_actvfunc)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) BASIC INPUTS: Entity-Types
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            L_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (L_entitytypes)
            
        #2) BASIC INPUTS: SP-Weights
        L_sp_weights = keras.layers.Input (shape=(self.MaxSPCount,),name="inp_sp_weights")
        MODEL_Inputs.append (L_sp_weights)
        
        #3)WORDS EMBEDDINGS and LSTMs
        L_shared_embd_words = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size, output_dim=H_wordEmbd_embedding_size, input_length=self.MaxSPLength , weights=[H_wordEmbd_weights], mask_zero=True , trainable=True, name = "shared_embd_words")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words")
        else:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words", activation=H_lstm_actv_func)
            
        WORDS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_words_"+str(path_index))
            embd   = L_shared_embd_words(inp)
            lstm   = L_shared_lstm_words(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_W_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_words_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            WORDS_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        if H_aggregation_type == "avg":
            L_word_lstms_aggr = keras.layers.Lambda(KERASHelper_NonZeros_Average,output_shape=(H_W_lstm_out_dim,), name="aggr_lstm_words_avg")(WORDS_LSTMS_Outputs+[L_sp_weights])
        else:
            L_word_lstms_aggr = keras.layers.add(WORDS_LSTMS_Outputs, name="aggr_lstm_words_sum")
        
        
        #4)POSTAGS EMBEDDINGS and LSTMs
        L_shared_embd_postags = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size, output_dim=H_postgEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_postags")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags")
        else:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags", activation=H_lstm_actv_func)
            
        POSTAGS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_postags_"+str(path_index))
            embd   = L_shared_embd_postags(inp)
            lstm   = L_shared_lstm_postags(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_P_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_postags_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            POSTAGS_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        if H_aggregation_type == "avg":
            L_postag_lstms_aggr = keras.layers.Lambda(KERASHelper_NonZeros_Average,output_shape=(H_P_lstm_out_dim,), name="aggr_lstm_postags_avg")(POSTAGS_LSTMS_Outputs+[L_sp_weights])
        else:
            L_postag_lstms_aggr = keras.layers.add(POSTAGS_LSTMS_Outputs, name="aggr_lstm_postags_sum")

        
        #5)DPTypes EMBEDDINGS and LSTMs
        L_shared_embd_dptypes = keras.layers.Embedding(input_dim=H_dptpsEmbd_vocab_size, output_dim=H_dptpsEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_dptypes")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes")
        else:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes", activation=H_lstm_actv_func)
            
        DPTYPES_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_dptypes_"+str(path_index))
            embd   = L_shared_embd_dptypes(inp)
            lstm   = L_shared_lstm_dptypes(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_D_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_dptypes_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            DPTYPES_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        if H_aggregation_type == "avg":
            L_dptype_lstms_aggr = keras.layers.Lambda(KERASHelper_NonZeros_Average,output_shape=(H_D_lstm_out_dim,), name="aggr_lstm_dptypes_avg")(DPTYPES_LSTMS_Outputs+[L_sp_weights])
        else:
            L_dptype_lstms_aggr = keras.layers.add(DPTYPES_LSTMS_Outputs, name="aggr_lstm_dptypes_sum")

        
        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr,L_entitytypes])
        else:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr])
        
        #7) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit

    def Arch_TopKP_3lstms_WA_wpd (self,wlod,plod,dlod,l_actvfunc,pembds,dtembds,dod,d_actvfunc,dr):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"] = True
        self.WhichFeaturesToUse["words"]   = True
        self.WhichFeaturesToUse["postags"] = True
        self.WhichFeaturesToUse["dptypes"] = True
        
        #LSTMs Params 
        H_W_lstm_out_dim     = wlod #Words-LSTM Output Dimensionality
        H_P_lstm_out_dim     = plod #POSTags-LSTM Output Dimensionality
        H_D_lstm_out_dim     = dlod #DTypes-LSTM Output Dimensionality
        H_lstm_actv_func     = l_actvfunc.lower() if len(l_actvfunc)>=3 else None #Activation function for all 3 LSTMs. None:default value will be used
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_dptpsEmbd_vocab_size     = self.DPTypesEmbeddings.len()+1
        H_dptpsEmbd_embedding_size = dtembds

        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = d_actvfunc.lower() if len(d_actvfunc)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) BASIC INPUTS: Entity-Types
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            L_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (L_entitytypes)
            
        #2) BASIC INPUTS: SP-Weights
        L_sp_weights = keras.layers.Input (shape=(self.MaxSPCount,),name="inp_sp_weights")
        MODEL_Inputs.append (L_sp_weights)
        
        #3)WORDS EMBEDDINGS and LSTMs
        L_shared_embd_words = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size, output_dim=H_wordEmbd_embedding_size, input_length=self.MaxSPLength , weights=[H_wordEmbd_weights], mask_zero=True , trainable=True, name = "shared_embd_words")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words")
        else:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words", activation=H_lstm_actv_func)
            
        WORDS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_words_"+str(path_index))
            embd   = L_shared_embd_words(inp)
            lstm   = L_shared_lstm_words(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_W_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_words_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            WORDS_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        L_word_lstms_aggr = keras.layers.Lambda(KERASHelper_Weighted_average,output_shape=(H_W_lstm_out_dim,), name="aggr_lstm_words_WA")(WORDS_LSTMS_Outputs+[L_sp_weights])
        
        
        #4)POSTAGS EMBEDDINGS and LSTMs
        L_shared_embd_postags = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size, output_dim=H_postgEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_postags")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags")
        else:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags", activation=H_lstm_actv_func)
            
        POSTAGS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_postags_"+str(path_index))
            embd   = L_shared_embd_postags(inp)
            lstm   = L_shared_lstm_postags(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_P_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_postags_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            POSTAGS_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        L_postag_lstms_aggr = keras.layers.Lambda(KERASHelper_Weighted_average,output_shape=(H_P_lstm_out_dim,), name="aggr_lstm_postags_WA")(POSTAGS_LSTMS_Outputs+[L_sp_weights])

        
        #5)DPTypes EMBEDDINGS and LSTMs
        L_shared_embd_dptypes = keras.layers.Embedding(input_dim=H_dptpsEmbd_vocab_size, output_dim=H_dptpsEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_dptypes")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes")
        else:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes", activation=H_lstm_actv_func)
            
        DPTYPES_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_dptypes_"+str(path_index))
            embd   = L_shared_embd_dptypes(inp)
            lstm   = L_shared_lstm_dptypes(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_D_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_dptypes_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            DPTYPES_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        L_dptype_lstms_aggr = keras.layers.Lambda(KERASHelper_Weighted_average,output_shape=(H_D_lstm_out_dim,), name="aggr_lstm_dptypes_WA")(DPTYPES_LSTMS_Outputs+[L_sp_weights])
        
        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr,L_entitytypes])
        else:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr])
        
        #7) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit


    def Arch_TopKP_3lstms_WA_wpd_DropBeforeDense (self,wlod,plod,dlod,l_actvfunc,pembds,dtembds,dod,d_actvfunc,dr):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"] = True
        self.WhichFeaturesToUse["words"]   = True
        self.WhichFeaturesToUse["postags"] = True
        self.WhichFeaturesToUse["dptypes"] = True
        
        #LSTMs Params 
        H_W_lstm_out_dim     = wlod #Words-LSTM Output Dimensionality
        H_P_lstm_out_dim     = plod #POSTags-LSTM Output Dimensionality
        H_D_lstm_out_dim     = dlod #DTypes-LSTM Output Dimensionality
        H_lstm_actv_func     = l_actvfunc.lower() if len(l_actvfunc)>=3 else None #Activation function for all 3 LSTMs. None:default value will be used
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_dptpsEmbd_vocab_size     = self.DPTypesEmbeddings.len()+1
        H_dptpsEmbd_embedding_size = dtembds

        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = d_actvfunc.lower() if len(d_actvfunc)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) BASIC INPUTS: Entity-Types
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            L_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (L_entitytypes)
            
        #2) BASIC INPUTS: SP-Weights
        L_sp_weights = keras.layers.Input (shape=(self.MaxSPCount,),name="inp_sp_weights")
        MODEL_Inputs.append (L_sp_weights)
        
        #3)WORDS EMBEDDINGS and LSTMs
        L_shared_embd_words = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size, output_dim=H_wordEmbd_embedding_size, input_length=self.MaxSPLength , weights=[H_wordEmbd_weights], mask_zero=True , trainable=True, name = "shared_embd_words")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words")
        else:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words", activation=H_lstm_actv_func)
            
        WORDS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_words_"+str(path_index))
            embd   = L_shared_embd_words(inp)
            lstm   = L_shared_lstm_words(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_W_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_words_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            WORDS_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        L_word_lstms_aggr = keras.layers.Lambda(KERASHelper_Weighted_average,output_shape=(H_W_lstm_out_dim,), name="aggr_lstm_words_WA")(WORDS_LSTMS_Outputs+[L_sp_weights])
        
        
        #4)POSTAGS EMBEDDINGS and LSTMs
        L_shared_embd_postags = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size, output_dim=H_postgEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_postags")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags")
        else:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags", activation=H_lstm_actv_func)
            
        POSTAGS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_postags_"+str(path_index))
            embd   = L_shared_embd_postags(inp)
            lstm   = L_shared_lstm_postags(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_P_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_postags_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            POSTAGS_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        L_postag_lstms_aggr = keras.layers.Lambda(KERASHelper_Weighted_average,output_shape=(H_P_lstm_out_dim,), name="aggr_lstm_postags_WA")(POSTAGS_LSTMS_Outputs+[L_sp_weights])

        
        #5)DPTypes EMBEDDINGS and LSTMs
        L_shared_embd_dptypes = keras.layers.Embedding(input_dim=H_dptpsEmbd_vocab_size, output_dim=H_dptpsEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_dptypes")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes")
        else:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes", activation=H_lstm_actv_func)
            
        DPTYPES_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_dptypes_"+str(path_index))
            embd   = L_shared_embd_dptypes(inp)
            lstm   = L_shared_lstm_dptypes(embd)
            X = keras.layers.Lambda(KERASHelper_Multiply_Weights,output_shape=(H_D_lstm_out_dim,), arguments={'PL_index':path_index}, name="lstm_dptypes_weights_mul_"+str(path_index))([lstm,L_sp_weights])
            DPTYPES_LSTMS_Outputs.append (X)
            MODEL_Inputs.append (inp)
        
        L_dptype_lstms_aggr = keras.layers.Lambda(KERASHelper_Weighted_average,output_shape=(H_D_lstm_out_dim,), name="aggr_lstm_dptypes_WA")(DPTYPES_LSTMS_Outputs+[L_sp_weights])
        
        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr,L_entitytypes])
        else:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr])
        
        #7) Dropout if needed , BEFORE DENSE .... 
        if H_dropout_Rate > 0:
            L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
        else:
            L_drop = L_all_features
            
        #8) Dense and Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_drop)
            L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit


    def Arch_TopKP_3lstms_wpd_maxp (self,wlod,plod,dlod,l_actvfunc,pembds,dtembds,dod,d_actvfunc,dr):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"] = False
        self.WhichFeaturesToUse["words"]   = True
        self.WhichFeaturesToUse["postags"] = True
        self.WhichFeaturesToUse["dptypes"] = True
        
        #LSTMs Params 
        H_W_lstm_out_dim     = wlod #Words-LSTM Output Dimensionality
        H_P_lstm_out_dim     = plod #POSTags-LSTM Output Dimensionality
        H_D_lstm_out_dim     = dlod #DTypes-LSTM Output Dimensionality
        H_lstm_actv_func     = l_actvfunc.lower() if len(l_actvfunc)>=3 else None #Activation function for all 3 LSTMs. None:default value will be used
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_dptpsEmbd_vocab_size     = self.DPTypesEmbeddings.len()+1
        H_dptpsEmbd_embedding_size = dtembds

        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = d_actvfunc.lower() if len(d_actvfunc)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) BASIC INPUTS: Entity-Types
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            L_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (L_entitytypes)
            
        #2) BASIC INPUTS: SP-Weights
        #L_sp_weights = keras.layers.Input (shape=(self.MaxSPCount,),name="inp_sp_weights")
        #MODEL_Inputs.append (L_sp_weights)
        
        #3)WORDS EMBEDDINGS and LSTMs
        L_shared_embd_words = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size, output_dim=H_wordEmbd_embedding_size, input_length=self.MaxSPLength , weights=[H_wordEmbd_weights], mask_zero=True , trainable=True, name = "shared_embd_words")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words")
        else:
            L_shared_lstm_words = keras.layers.LSTM(units=H_W_lstm_out_dim,name="shared_lstm_words", activation=H_lstm_actv_func)
            
        WORDS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_words_"+str(path_index))
            embd   = L_shared_embd_words(inp)
            lstm   = L_shared_lstm_words(embd)
            WORDS_LSTMS_Outputs.append (lstm)
            MODEL_Inputs.append (inp)
        
        L_word_lstms_aggr = keras.layers.Maximum(name="aggr_lstm_words_sum")(WORDS_LSTMS_Outputs)
        
        #4)POSTAGS EMBEDDINGS and LSTMs
        L_shared_embd_postags = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size, output_dim=H_postgEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_postags")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags")
        else:
            L_shared_lstm_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="shared_lstm_postags", activation=H_lstm_actv_func)
            
        POSTAGS_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_postags_"+str(path_index))
            embd   = L_shared_embd_postags(inp)
            lstm   = L_shared_lstm_postags(embd)
            POSTAGS_LSTMS_Outputs.append (lstm)
            MODEL_Inputs.append (inp)
        
        L_postag_lstms_aggr = keras.layers.Maximum(name="aggr_lstm_postags_sum")(POSTAGS_LSTMS_Outputs)
        
        #5)DPTypes EMBEDDINGS and LSTMs
        L_shared_embd_dptypes = keras.layers.Embedding(input_dim=H_dptpsEmbd_vocab_size, output_dim=H_dptpsEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "shared_embd_dptypes")
        
        if H_lstm_actv_func == None:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes")
        else:
            L_shared_lstm_dptypes = keras.layers.LSTM(units=H_D_lstm_out_dim,name="shared_lstm_dptypes", activation=H_lstm_actv_func)
            
        DPTYPES_LSTMS_Outputs = [] 
        for path_index in range(self.MaxSPCount):
            inp    = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_dptypes_"+str(path_index))
            embd   = L_shared_embd_dptypes(inp)
            lstm   = L_shared_lstm_dptypes(embd)
            DPTYPES_LSTMS_Outputs.append (lstm)
            MODEL_Inputs.append (inp)
        
        L_dptype_lstms_aggr = keras.layers.Maximum(name="aggr_lstm_dptypes_sum")(DPTYPES_LSTMS_Outputs)
        
        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr,L_entitytypes])
        else:
            L_all_features = keras.layers.concatenate([L_word_lstms_aggr,L_postag_lstms_aggr,L_dptype_lstms_aggr])
        
        #7) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit
        
    def Arch_FS_1Bilstm (self,lod,pembds,cnkembs,dod,actfuncs,dr,maxPool):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["fs_words"]    = True
        self.WhichFeaturesToUse["fs_postags"]  = True
        self.WhichFeaturesToUse["fs_chunks"]   = True
        self.WhichFeaturesToUse["fs_p1_dists"] = True
        self.WhichFeaturesToUse["fs_p2_dists"] = True
        
        # Params 
        H_lstm_out_dim = lod #Words-LSTM Output Dimensionality
        H_lstm_actv_func  = actfuncs.lower() if len(actfuncs)>=3 else None #None:default value will be used
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_chunksEmbd_vocab_size    = 3 + len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"].keys()) + 1 # before,middle,after,e1_tp,e2_tp,...,en_tp,MASK
        H_cunksEmbd_embedding_size = cnkembs
        
        H_posiEmbd_vocab_size      = self.PRJ.PositionEmbeddings.shape()[0]
        H_posiEmbd_embedding_size  = self.PRJ.PositionEmbeddings.shape()[1]
        H_posiEmbd_weights         = np.array(self.PRJ.PositionEmbeddings.get_position_embedding_matrix(), dtype = np.float32)
        
        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = actfuncs.lower() if len(actfuncs)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        can_mask = not maxPool
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) INPUTS: Entity-Types
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            INP_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (INP_entitytypes)
            
        #2) INPUTS
        INP_words  = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_words")
        INP_postgs = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_postags")
        INP_chunks = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_chunks")
        INP_p1dist = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_p1_dists")
        INP_p2dist = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_p2_dists")
        
        MODEL_Inputs.extend ([INP_words,INP_postgs,INP_chunks,INP_p1dist,INP_p2dist])
        
        #3) Embeddings 
        EMBD_words  = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size  , output_dim=H_wordEmbd_embedding_size , input_length=self.MaxFSLength,weights=[H_wordEmbd_weights], mask_zero=can_mask , trainable=True, name = "shared_embd_words")(INP_words)
        EMBD_postgs = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size , output_dim=H_postgEmbd_embedding_size, input_length=self.MaxFSLength,                              mask_zero=can_mask , trainable=True, name = "shared_embd_postags")(INP_postgs)
        EMBD_chunks = keras.layers.Embedding(input_dim=H_chunksEmbd_vocab_size, output_dim=H_cunksEmbd_embedding_size, input_length=self.MaxFSLength,                              mask_zero=can_mask, trainable=True, name = "shared_embd_chunks")(INP_chunks)
        EMBD_p1dist = keras.layers.Embedding(input_dim=H_posiEmbd_vocab_size  , output_dim=H_posiEmbd_embedding_size , input_length=self.MaxFSLength,weights=[H_posiEmbd_weights], mask_zero=can_mask , trainable=True, name = "shared_embd_p1dists")(INP_p1dist)
        EMBD_p2dist = keras.layers.Embedding(input_dim=H_posiEmbd_vocab_size  , output_dim=H_posiEmbd_embedding_size , input_length=self.MaxFSLength,weights=[H_posiEmbd_weights], mask_zero=can_mask , trainable=True, name = "shared_embd_p2dists")(INP_p2dist)
        
        #4) concat embeddings 
        EMBD_ALL = keras.layers.concatenate([EMBD_words,EMBD_postgs,EMBD_chunks,EMBD_p1dist,EMBD_p2dist])
        
        #5) lstm 
        if H_lstm_actv_func == None:
            LSTM = keras.layers.Bidirectional(keras.layers.LSTM(units=H_lstm_out_dim,name="LSTM_all", return_sequences=maxPool))(EMBD_ALL)
        else:
            LSTM = keras.layers.Bidirectional(keras.layers.LSTM(units=H_lstm_out_dim,name="LSTM_all", activation=H_lstm_actv_func, return_sequences=maxPool))(EMBD_ALL)
        
        if maxPool:
            LSTM = keras.layers.GlobalMaxPool1D()(LSTM)
            
        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([LSTM,INP_entitytypes])
        else:
            L_all_features = LSTM
        
        #7) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit


    def Arch_TopKP_3Bilstms_WA_wpd_MaxPoolSup (self,wlod,plod,dlod,l_actvfunc,pembds,dtembds,dod,d_actvfunc,dr,max_pool):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"] = False
        self.WhichFeaturesToUse["words"]   = True
        self.WhichFeaturesToUse["postags"] = True
        self.WhichFeaturesToUse["dptypes"] = True
        
        #LSTMs Params 
        H_W_lstm_out_dim     = wlod #Words-LSTM Output Dimensionality
        H_P_lstm_out_dim     = plod #POSTags-LSTM Output Dimensionality
        H_D_lstm_out_dim     = dlod #DTypes-LSTM Output Dimensionality
        H_lstm_actv_func     = l_actvfunc.lower() if len(l_actvfunc)>=3 else None #Activation function for all 3 LSTMs. None:default value will be used
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_dptpsEmbd_vocab_size     = self.DPTypesEmbeddings.len()+1
        H_dptpsEmbd_embedding_size = dtembds

        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = d_actvfunc.lower() if len(d_actvfunc)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        
        #can mask
        can_mask = not max_pool
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) INPUTS: 
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            INP_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (INP_entitytypes)
            
        #4) Inputs 
        INP_words   = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_words")
        INP_postags = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_postags")
        INP_deptyps = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_dptypes")
        MODEL_Inputs.extend ([INP_words,INP_postags,INP_deptyps])
        
        #2) Embeddings 
        EMBD_words   = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size , output_dim=H_wordEmbd_embedding_size , input_length=self.MaxSPLength , weights=[H_wordEmbd_weights], mask_zero=can_mask , trainable=True, name = "Embd_words")(INP_words)
        EMBD_postags = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size, output_dim=H_postgEmbd_embedding_size, input_length=self.MaxSPLength ,                               mask_zero=can_mask , trainable=True, name = "Embd_postags")(INP_postags)
        EMBD_deptyps = keras.layers.Embedding(input_dim=H_dptpsEmbd_vocab_size, output_dim=H_dptpsEmbd_embedding_size, input_length=self.MaxSPLength ,                               mask_zero=can_mask , trainable=True, name = "Embd_dptypes")(INP_deptyps)
        
        #3) LSTMS: 
        if H_lstm_actv_func == None:
            LSTM_words   = keras.layers.Bidirectional(keras.layers.LSTM(units=H_W_lstm_out_dim,name="lstm_words"   , return_sequences = max_pool))(EMBD_words)
            LSTM_postags = keras.layers.Bidirectional(keras.layers.LSTM(units=H_P_lstm_out_dim,name="lstm_postags" , return_sequences = max_pool))(EMBD_postags)
            LSTM_deptyps = keras.layers.Bidirectional(keras.layers.LSTM(units=H_D_lstm_out_dim,name="lstm_dptypes" , return_sequences = max_pool))(EMBD_deptyps)
        else:
            LSTM_words   = keras.layers.Bidirectional(keras.layers.LSTM(units=H_W_lstm_out_dim,name="lstm_words"   , return_sequences = max_pool, activation=H_lstm_actv_func))(EMBD_words)
            LSTM_postags = keras.layers.Bidirectional(keras.layers.LSTM(units=H_P_lstm_out_dim,name="lstm_postags" , return_sequences = max_pool, activation=H_lstm_actv_func))(EMBD_postags)
            LSTM_deptyps = keras.layers.Bidirectional(keras.layers.LSTM(units=H_D_lstm_out_dim,name="lstm_dptypes" , return_sequences = max_pool, activation=H_lstm_actv_func))(EMBD_deptyps)

        if max_pool:
            LSTM_words   = keras.layers.GlobalMaxPool1D()(LSTM_words)
            LSTM_postags = keras.layers.GlobalMaxPool1D()(LSTM_postags)
            LSTM_deptyps = keras.layers.GlobalMaxPool1D()(LSTM_deptyps)
       
        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([LSTM_words,LSTM_postags,LSTM_deptyps,INP_entitytypes])
        else:
            L_all_features = keras.layers.concatenate([LSTM_words,LSTM_postags,LSTM_deptyps])
        
        #7) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit

    def Arch_SDP3lstms_FS1lstm_MaxPool (self,wlod,plod,dlod,fslod,pembds,dtembds,cnkembs,dod,dr,actfuncs):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"]     = False 
        self.WhichFeaturesToUse["words"]       = True
        self.WhichFeaturesToUse["postags"]     = True
        self.WhichFeaturesToUse["dptypes"]     = True
        self.WhichFeaturesToUse["fs_words"]    = True
        self.WhichFeaturesToUse["fs_postags"]  = True
        self.WhichFeaturesToUse["fs_chunks"]   = True
        self.WhichFeaturesToUse["fs_p1_dists"] = True
        self.WhichFeaturesToUse["fs_p2_dists"] = True
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_dptpsEmbd_vocab_size     = self.DPTypesEmbeddings.len()+1
        H_dptpsEmbd_embedding_size = dtembds

        H_chunksEmbd_vocab_size    = 3 + len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"].keys()) + 1 # before,middle,after,e1_tp,e2_tp,...,en_tp,MASK
        H_cunksEmbd_embedding_size = cnkembs
        
        H_posiEmbd_vocab_size      = self.PRJ.PositionEmbeddings.shape()[0]
        H_posiEmbd_embedding_size  = self.PRJ.PositionEmbeddings.shape()[1]
        H_posiEmbd_weights         = np.array(self.PRJ.PositionEmbeddings.get_position_embedding_matrix(), dtype = np.float32)
        
        #LSTMs Params 
        H_W_lstm_out_dim     = wlod  #SDP
        H_P_lstm_out_dim     = plod  #SDP
        H_D_lstm_out_dim     = dlod  #DTypes
        H_FS_lstm_out_dim    = fslod #Full Sentence 
        H_lstm_actv_func     = actfuncs.lower() if len(actfuncs)>=3 else None #Activation function for all 3 LSTMs. None:default value will be used
        
        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = actfuncs.lower() if len(actfuncs)>=3 else None #None:default value will be used
        H_dropout_Rate     = dr
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) INPUTS: 
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            INP_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (INP_entitytypes)
            
        INP_SDP_words   = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_SDP_words")
        INP_SDP_postags = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_SDP_postags")
        INP_SDP_deptyps = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_SDP_dptypes")
        INP_FS_words    = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_words")
        INP_FS_postgs   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_postags")
        INP_FS_chunks   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_chunks")
        INP_FS_p1dist   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_p1_dists")
        INP_FS_p2dist   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_p2_dists")
        
        MODEL_Inputs.extend ([INP_SDP_words,INP_SDP_postags,INP_SDP_deptyps,INP_FS_words,INP_FS_postgs,INP_FS_chunks,INP_FS_p1dist,INP_FS_p2dist])

        #2) Embeddings 
        #input_length=self.MaxSPLength ,                               
        SHARED_EMBD_words = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size , output_dim=H_wordEmbd_embedding_size , weights=[H_wordEmbd_weights], mask_zero=False , trainable=True, name = "embd_shared_words")
        EMBD_SDP_words    = SHARED_EMBD_words(INP_SDP_words)
        EMBD_FS_words     = SHARED_EMBD_words(INP_FS_words)
        
        #input_length=self.MaxSPLength ,                               
        SHARED_EMBD_postags = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size, output_dim=H_postgEmbd_embedding_size, mask_zero=False , trainable=True, name = "embd_shared_postags")
        EMBD_SDP_postags    = SHARED_EMBD_postags(INP_SDP_postags)
        EMBD_FS_postags     = SHARED_EMBD_postags(INP_FS_postgs)
        
        EMBD_SDP_deptyps = keras.layers.Embedding(input_dim=H_dptpsEmbd_vocab_size, output_dim=H_dptpsEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=True , trainable=True, name = "embd_sdp_dptypes")(INP_SDP_deptyps)
        
        EMBD_FS_chunks = keras.layers.Embedding(input_dim=H_chunksEmbd_vocab_size, output_dim=H_cunksEmbd_embedding_size, input_length=self.MaxFSLength,                              mask_zero=False,  trainable=True, name = "embd_fs_chunks")(INP_FS_chunks)
        EMBD_FS_p1dist = keras.layers.Embedding(input_dim=H_posiEmbd_vocab_size  , output_dim=H_posiEmbd_embedding_size , input_length=self.MaxFSLength,weights=[H_posiEmbd_weights], mask_zero=False, trainable=True, name = "embd_fs_p1dists")(INP_FS_p1dist)
        EMBD_FS_p2dist = keras.layers.Embedding(input_dim=H_posiEmbd_vocab_size  , output_dim=H_posiEmbd_embedding_size , input_length=self.MaxFSLength,weights=[H_posiEmbd_weights], mask_zero=False, trainable=True, name = "embd_fs_p2dists")(INP_FS_p2dist)

        EMBD_ALL_FS = keras.layers.concatenate([EMBD_FS_words,EMBD_FS_postags,EMBD_FS_chunks,EMBD_FS_p1dist,EMBD_FS_p2dist])

        #3) LSTMS: 
        if H_lstm_actv_func == None:
            SDP_LSTM_words   = keras.layers.LSTM(units=H_W_lstm_out_dim, name="lstm_SDP_words"  )(EMBD_SDP_words)
            SDP_LSTM_postags = keras.layers.LSTM(units=H_P_lstm_out_dim, name="lstm_SDP_postags")(EMBD_SDP_postags)
            SDP_LSTM_deptyps = keras.layers.LSTM(units=H_D_lstm_out_dim, name="lstm_SDP_dptypes")(EMBD_SDP_deptyps)
            FS_LSTM          = keras.layers.Bidirectional(keras.layers.LSTM(units=H_FS_lstm_out_dim,name="LSTM_FS", return_sequences=True))(EMBD_ALL_FS)
        else:
            SDP_LSTM_words   = keras.layers.LSTM(units=H_W_lstm_out_dim,name="lstm_SDP_words"   , activation=H_lstm_actv_func)(EMBD_SDP_words)
            SDP_LSTM_postags = keras.layers.LSTM(units=H_P_lstm_out_dim,name="lstm_SDP_postags" , activation=H_lstm_actv_func)(EMBD_SDP_postags)
            SDP_LSTM_deptyps = keras.layers.LSTM(units=H_D_lstm_out_dim,name="lstm_SDP_dptypes" , activation=H_lstm_actv_func)(EMBD_SDP_deptyps)
            FS_LSTM          = keras.layers.Bidirectional(keras.layers.LSTM(units=H_FS_lstm_out_dim,name="LSTM_FS", activation=H_lstm_actv_func, return_sequences=True))(EMBD_ALL_FS)
       
        FS_LSTM = keras.layers.GlobalMaxPool1D()(FS_LSTM) 

        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate([SDP_LSTM_words,SDP_LSTM_postags,SDP_LSTM_deptyps,FS_LSTM,INP_entitytypes])
        else:
            L_all_features = keras.layers.concatenate([SDP_LSTM_words,SDP_LSTM_postags,SDP_LSTM_deptyps,FS_LSTM])

        #7) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit

    def Arch_SDPCNN_FSCNN (self,SDPfilters,SDPWS,FSfilters,FSWS,pembds,dtembds,cnkembs,dod,dr,dactfunc):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) 
        from keras.models import Model 
        import keras.layers
        
        self.WhichFeaturesToUse["weights"]     = False 
        self.WhichFeaturesToUse["words"]       = True
        self.WhichFeaturesToUse["postags"]     = True
        self.WhichFeaturesToUse["dptypes"]     = True
        self.WhichFeaturesToUse["fs_words"]    = True
        self.WhichFeaturesToUse["fs_postags"]  = True
        self.WhichFeaturesToUse["fs_chunks"]   = True
        self.WhichFeaturesToUse["fs_p1_dists"] = True
        self.WhichFeaturesToUse["fs_p2_dists"] = True
        
        #Embeddings Params
        H_wordEmbd_vocab_size      = self.wv.shape()[0]
        H_wordEmbd_embedding_size  = self.wv.shape()[1]
        H_wordEmbd_weights         = self.wv.get_normalized_weights(deepcopy=True)
        
        H_postgEmbd_vocab_size     = self.POSTagsEmbeddings.len()+1
        H_postgEmbd_embedding_size = pembds
        
        H_dptpsEmbd_vocab_size     = self.DPTypesEmbeddings.len()+1
        H_dptpsEmbd_embedding_size = dtembds

        H_chunksEmbd_vocab_size    = 3 + len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"].keys()) + 1 # before,middle,after,e1_tp,e2_tp,...,en_tp,MASK
        H_cunksEmbd_embedding_size = cnkembs
        
        H_posiEmbd_vocab_size      = self.PRJ.PositionEmbeddings.shape()[0]
        H_posiEmbd_embedding_size  = self.PRJ.PositionEmbeddings.shape()[1]
        H_posiEmbd_weights         = np.array(self.PRJ.PositionEmbeddings.get_position_embedding_matrix(), dtype = np.float32)
        
        #Dence and Dropout Params 
        H_dense_out_dim    = dod 
        H_dense_actv_func  = dactfunc.lower() if len(dactfunc)>=3 else "tanh" #None:default value will be used
        H_dropout_Rate     = dr
        
        #Model inputs / outputs with order 
        MODEL_Inputs  = []
        MODEL_Outputs = []
        
        #---------------------------------------- BUILDING LAYERS -------------------------------------------------
        #1) INPUTS: 
        if self.WhichFeaturesToUse["entitytypes"] == True:
            entitytype_length = 2*len(self.PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])
            INP_entitytypes = keras.layers.Input (shape=(entitytype_length,),name="inp_entity_types")
            MODEL_Inputs.append (INP_entitytypes)
            
        INP_SDP_words   = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_SDP_words")
        INP_SDP_postags = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_SDP_postags")
        INP_SDP_deptyps = keras.layers.Input (shape=(self.MaxSPLength,),name="inp_SDP_dptypes")
        INP_FS_words    = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_words")
        INP_FS_postgs   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_postags")
        INP_FS_chunks   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_chunks")
        INP_FS_p1dist   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_p1_dists")
        INP_FS_p2dist   = keras.layers.Input (shape=(self.MaxFSLength,),name="inp_FS_p2_dists")
        
        MODEL_Inputs.extend ([INP_SDP_words,INP_SDP_postags,INP_SDP_deptyps,INP_FS_words,INP_FS_postgs,INP_FS_chunks,INP_FS_p1dist,INP_FS_p2dist])

        #2) Embeddings 
        #input_length=self.MaxSPLength ,                               
        SHARED_EMBD_words = keras.layers.Embedding(input_dim=H_wordEmbd_vocab_size , output_dim=H_wordEmbd_embedding_size , weights=[H_wordEmbd_weights], mask_zero=False , trainable=True, name = "embd_shared_words")
        EMBD_SDP_words    = SHARED_EMBD_words(INP_SDP_words)
        EMBD_FS_words     = SHARED_EMBD_words(INP_FS_words)
        
        #input_length=self.MaxSPLength ,                               
        SHARED_EMBD_postags = keras.layers.Embedding(input_dim=H_postgEmbd_vocab_size, output_dim=H_postgEmbd_embedding_size, mask_zero=False , trainable=True, name = "embd_shared_postags")
        EMBD_SDP_postags    = SHARED_EMBD_postags(INP_SDP_postags)
        EMBD_FS_postags     = SHARED_EMBD_postags(INP_FS_postgs)
        
        EMBD_SDP_deptyps = keras.layers.Embedding(input_dim=H_dptpsEmbd_vocab_size, output_dim=H_dptpsEmbd_embedding_size, input_length=self.MaxSPLength , mask_zero=False , trainable=True, name = "embd_sdp_dptypes")(INP_SDP_deptyps)
        
        EMBD_FS_chunks = keras.layers.Embedding(input_dim=H_chunksEmbd_vocab_size, output_dim=H_cunksEmbd_embedding_size, input_length=self.MaxFSLength,                              mask_zero=False,  trainable=True, name = "embd_fs_chunks")(INP_FS_chunks)
        EMBD_FS_p1dist = keras.layers.Embedding(input_dim=H_posiEmbd_vocab_size  , output_dim=H_posiEmbd_embedding_size , input_length=self.MaxFSLength,weights=[H_posiEmbd_weights], mask_zero=False, trainable=True, name = "embd_fs_p1dists")(INP_FS_p1dist)
        EMBD_FS_p2dist = keras.layers.Embedding(input_dim=H_posiEmbd_vocab_size  , output_dim=H_posiEmbd_embedding_size , input_length=self.MaxFSLength,weights=[H_posiEmbd_weights], mask_zero=False, trainable=True, name = "embd_fs_p2dists")(INP_FS_p2dist)

        EMBD_ALL_FS  = keras.layers.concatenate([EMBD_FS_words,EMBD_FS_postags,EMBD_FS_chunks,EMBD_FS_p1dist,EMBD_FS_p2dist])
        EMDB_ALL_SDP = keras.layers.concatenate([EMBD_SDP_words,EMBD_SDP_postags,EMBD_SDP_deptyps])
        
        CNNs_OUTS = [] 
        #3) CNNs
        for window_size in SDPWS:
            cnn = keras.layers.Conv1D(SDPfilters,window_size, name ="CNN_SDP_ws_"+str(window_size) , activation = "relu")(EMDB_ALL_SDP)
            cnn = keras.layers.GlobalMaxPool1D(name="Maxpool_CNN_SDP_ws_"+str(window_size))(cnn)
            CNNs_OUTS.append (cnn)
        
        for window_size in FSWS:
            cnn = keras.layers.Conv1D(FSfilters,window_size, name ="CNN_FS_ws_"+str(window_size) ,  activation = "relu")(EMBD_ALL_FS)
            cnn = keras.layers.GlobalMaxPool1D(name="Maxpool_CNN_FS_ws_"+str(window_size))(cnn)
            CNNs_OUTS.append (cnn)
            
        #FS_LSTM = keras.layers.Bidirectional(keras.layers.LSTM(units=H_FS_lstm_out_dim,name="LSTM_FS", return_sequences=True))(EMBD_ALL_FS)
        #FS_LSTM = keras.layers.GlobalMaxPool1D()(FS_LSTM) 

        #6) CONCATANATE ALL FEATURES
        if self.WhichFeaturesToUse["entitytypes"] == True:
            L_all_features = keras.layers.concatenate(CNNs_OUTS + [INP_entitytypes])
        else:
            L_all_features = keras.layers.concatenate(CNNs_OUTS)

        #7) Dense/Dropout/Decision layers
        Y_dim = len (self.PRJ.Configs["OneHotEncodingForMultiClass"])
        if H_dense_out_dim > 0:         
            L_dense = keras.layers.Dense(units=H_dense_out_dim,activation=H_dense_actv_func)(L_all_features)
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_dense)
                L_decision      = keras.layers.Dense(units=Y_dim ,activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_dense)
        else:
            if H_dropout_Rate > 0:
                L_drop = keras.layers.Dropout(H_dropout_Rate)(L_all_features)
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_drop)
            else:
                L_decision      = keras.layers.Dense(units=Y_dim, activation="softmax",name="decision_Y")(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model , self.WhichFeaturesToUse , self.WhichOutputsToPredit       