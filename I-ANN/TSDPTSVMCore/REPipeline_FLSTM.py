import TEESDocumentHelper;
import ProjectRunner; 
import GeneralFunctions as GF ; 
import sklearn.metrics as METRICS ; 

class FLSTM_RE_PARAMS:
    def __init__(self):
        self.TrainingSet_Files_List    = []
        self.DevelopmentSet_Files_List = []
        self.TestSet_Files_Lists       = []
        self.TrainingSet_DuplicationRemovalPolicy = None    #None: select according to the given config file, no set-specific policy
        self.DevelopmentSet_DuplicationRemovalPolicy = None #None: select according to the given config file, no set-specific policy
        self.TestSet_DuplicationRemovalPolicy = None        #None: select according to the given config file, no set-specific policy
        self.HowManyClassifiers = 15 
        self.KERAS_validation_split_IfDevel__DATANotGiven = 0.15 
        self.KERAS_fit_verbose = 2 
        self.ShuffleTrainingSetSentencesBeforeNIGeneration = False  #<<<CRITICAL>>> default value is FALSE for replicablity purpose, when using a particular random seed.
        self.ArchitectureBuilderMethodName = ["BuildArchitecture_BioNLPST2016_Paper"] 

        self.NN_MaxNumberOfEpochs  = 10 
        self.NN_Keras_batch_size   = 10  
        self.NN_Keras_class_weight = None 
        self.NN_Keras_optimizer    = "adam"  

        self.PredictDevelSet  = True
        self.EvaluateDevelSet = True
        self.WriteBackDevelSetPredictions  = False 
        self.DevelSetPredictionOutputFolder= None

        self.PredictTestSet  = False 
        self.EvaluateTestSet = False 
        self.WriteBackTestSetPredictions   = False 
        self.TestSetPredictionOutputFolder = None

        self.ProcessDevelSetAfterEpochNo =  0 ; #After epoch no X, DevelSet is predicted and evaluated. [Hint: set to -1 to NEVER predict develset]
        self.ProcessTestSetAfterEpochNo  = -1 ; #After epoch no X, IF needed, Test set is going to be predicted and/or evaluated and/or written-back. [Hint: -1 to Never process EXCEPT the last epoch!]

        self.SkipWholeSentences_TrainingSet    = [] #Completely ignore sentences with given ids  
        self.SkipWholeSentences_DevelopmentSet = [] #Completely ignore sentences with given ids  
        self.SkipWholeSentences_TestSet        = [] #Completely ignore sentences with given ids  

        self.WriteBackProcessedTrainingSet    = False 
        self.WriteBackProcessedDevelopmentSet = False 
        self.WriteBackProcessedTestSet        = False
        self.WriteBackProcessedTrainingSet_OutputFileAddress    = None
        self.WriteBackProcessedDevelopmentSet_OutputFileAddress = None
        self.WriteBackProcessedTestSet_OutputFileAddress        = None
        
    def __str__(self):
        S = "TrainingSet_Files_List         :" + str(self.TrainingSet_Files_List)          + "\n" + \
            "DevelopmentSet_Files_List      :" + str(self.DevelopmentSet_Files_List)       + "\n" + \
            "TestSet_Files_Lists            :" + str(self.TestSet_Files_Lists)             + "\n" + \
            "HowManyClassifiers             :" + str(self.HowManyClassifiers)              + "\n" + \
            "ArchitectureBuilderMethodName  :" + str(self.ArchitectureBuilderMethodName)   + "\n" + \
            "NN_MaxNumberOfEpochs           :" + str(self.NN_MaxNumberOfEpochs)            + "\n" + \
            "NN_Keras_batch_size            :" + str(self.NN_Keras_batch_size)             + "\n" + \
            "NN_Keras_class_weight          :" + str(self.NN_Keras_class_weight)           + "\n" + \
            "NN_Keras_optimizer             :" + str(self.NN_Keras_optimizer)              + "\n" + \
            \
            "PredictDevelSet                :" + str(self.PredictDevelSet)                 + "\n" + \
            "EvaluateDevelSet               :" + str(self.EvaluateDevelSet)                + "\n" + \
            "WriteBackDevelSetPredictions   :" + str(self.WriteBackDevelSetPredictions)    + "\n" + \
            "DevelSetPredictionOutputFolder :" + str(self.DevelSetPredictionOutputFolder)  + "\n" + \
            \
            "PredictTestSet                 :" + str(self.PredictTestSet)                  + "\n" + \
            "EvaluateTestSet                :" + str(self.EvaluateTestSet)                 + "\n" + \
            "WriteBackTestSetPredictions    :" + str(self.WriteBackTestSetPredictions)     + "\n" + \
            "TestSetPredictionOutputFolder  :" + str(self.TestSetPredictionOutputFolder)   + "\n" + \
            \
            "ProcessDevelSetAfterEpochNo    :" + str(self.ProcessDevelSetAfterEpochNo)     + "\n" + \
            "ProcessTestSetAfterEpochNo     :" + str(self.ProcessTestSetAfterEpochNo)      + "\n" + \
            "KERAS_validation_split_IfDevel__DATANotGiven :" + str(self.KERAS_validation_split_IfDevel__DATANotGiven) + "\n" + \
            "KERAS_fit_verbose            :" + str(self.KERAS_fit_verbose) + "\n" + \
            "ShuffleTrainingSetSentencesBeforeNIGeneration:" + str(self.ShuffleTrainingSetSentencesBeforeNIGeneration) + "\n" + \
            "SkipWholeSentences_TrainingSet    :" + str(self.SkipWholeSentences_TrainingSet)    + "\n" + \
            "SkipWholeSentences_DevelopmentSet :" + str(self.SkipWholeSentences_DevelopmentSet) + "\n" + \
            "SkipWholeSentences_TestSet        :" + str(self.SkipWholeSentences_TestSet)        + "\n" + \
            "WriteBackProcessedTrainingSet                :" + str(self.WriteBackProcessedTrainingSet) + "\n" + \
            "WriteBackProcessedDevelopmentSet             :" + str(self.WriteBackProcessedDevelopmentSet) + "\n" + \
            "WriteBackProcessedTestSet                    :" + str(self.WriteBackProcessedTestSet) + "\n" + \
            "WriteBackProcessedTrainingSet_OutputFileAddress    :" + str(self.WriteBackProcessedTrainingSet_OutputFileAddress) + "\n" + \
            "WriteBackProcessedDevelopmentSet_OutputFileAddress :" + str(self.WriteBackProcessedDevelopmentSet_OutputFileAddress) + "\n" + \
            "WriteBackProcessedTestSet_OutputFileAddress        :" + str(self.WriteBackProcessedTestSet_OutputFileAddress) + "\n" + \
            "TrainingSet_DuplicationRemovalPolicy     :" + str(self.TrainingSet_DuplicationRemovalPolicy) + "\n" + \
            "DevelopmentSet_DuplicationRemovalPolicy  :" + str(self.DevelopmentSet_DuplicationRemovalPolicy) + "\n" + \
            "TestSet_DuplicationRemovalPolicy         :" + str(self.TestSet_DuplicationRemovalPolicy) ; 

        return S ; 
        
       
class FLSTM_RE_Pipeline:
    def __init__(self, ConfigFileAddress, LogFileAddress, RE_PARAMS):
        self.__PRJ   = ProjectRunner.FLSTM_ProjectRunner (ConfigFileAddress, LogFileAddress) 
        self.__DATA  = {"TrainingSet":{}, "DevelopmentSet":{}, "TestSet": {}}
        self.__model = None 
        self.__WhichFeaturesToUse   = None 
        self.__WhichOutputsToPredit = None 
        self.__PerformanceResults = [] ; 
        self.__ResetTraining ()
        self.__DecimalPoints = self.__PRJ.Configs["EvaluationParameters"]["DecimalPoints"]; 
        if not isinstance(RE_PARAMS , FLSTM_RE_PARAMS):
            self.__PRJ.PROGRAM_Halt ("RE_PARAMS should be an instance of FLSTM_RE_PARAMS"); 
        else:
            self.__RE_PARAMS = RE_PARAMS ; 
            self.__Verify_RE_PARAMS();
            self.__PRJ.lp (["-"*40, "RE_PARAMS:", "----------", str(self.__RE_PARAMS) , "-"*40]); 
            
    def __exit__(self):
        self.__PRJ.__exit__(); 
            
    def __ResetTraining (self):
        self.__RuntimeStatistics = {"SelectedArchitecture": None, "ExecutionRound":0 , "RandomSeed": None, "NoEpochs": 0}; 
        if self.__model <> None:        
            del self.__model;
        self.__model = None; 
        self.__WhichFeaturesToUse = None 
        self.__WhichOutputsToPredit = None 
        
        
    def __Verify_RE_PARAMS (self):
        if self.__RE_PARAMS.HowManyClassifiers < 1:
            self.__PRJ.PROGRAM_Halt ("RE_PARAMS. HowManyClassifiers should be >=1.")            

        if (self.__RE_PARAMS.ArchitectureBuilderMethodName == None) or (not isinstance (self.__RE_PARAMS.ArchitectureBuilderMethodName, list)) or (len(self.__RE_PARAMS.ArchitectureBuilderMethodName)<1):
            self.__PRJ.PROGRAM_Halt ("ArchitectureBuilderMethodName should be a list with at least one valid NN architecture builder method."); 
        
        #Evaluate Training Set Params 
        if (self.__RE_PARAMS.TrainingSet_Files_List == None) or (not isinstance (self.__RE_PARAMS.TrainingSet_Files_List, list)) or (len(self.__RE_PARAMS.TrainingSet_Files_List)<1):
            self.__PRJ.PROGRAM_Halt ("TrainingSet_Files_List should be a list with a address of at least one TEES XML file."); 
        
        #Evaluate Development Set Params 
        if (self.__RE_PARAMS.PredictDevelSet == True) or (self.__RE_PARAMS.EvaluateDevelSet == True) or (self.__RE_PARAMS.WriteBackDevelSetPredictions == True):
            if (self.__RE_PARAMS.DevelopmentSet_Files_List == None) or (not isinstance (self.__RE_PARAMS.DevelopmentSet_Files_List, list)) or (len(self.__RE_PARAMS.DevelopmentSet_Files_List)<1):
                self.__PRJ.PROGRAM_Halt ("DevelopmentSet_Files_List should be a list with a address of at least one TEES XML file."); 

        if (self.__RE_PARAMS.EvaluateDevelSet == True) and (self.__RE_PARAMS.PredictDevelSet == False):
            self.__PRJ.PROGRAM_Halt ("Evaluation of DevelSet predictions is requested, But PredictDevelSet is set to False.")

        if (self.__RE_PARAMS.WriteBackDevelSetPredictions == True):
            if self.__RE_PARAMS.PredictDevelSet == False:
                self.__PRJ.PROGRAM_Halt ("WriteBack of DevelSet predictions is requested, But PredictDevelSet is set to False.")
                
            if len(self.__RE_PARAMS.DevelopmentSet_Files_List)<>1: 
                self.__PRJ.PROGRAM_Halt ("WriteBack of DevelSet predictions is requested, hence only 1 GOLD DevelSet xml file should be given to the DevelopmentSet_Files_List param.")
            
            import os ; 
            if (self.__RE_PARAMS.DevelSetPredictionOutputFolder == None) or (not isinstance (self.__RE_PARAMS.DevelSetPredictionOutputFolder, str)) or \
               (not os.path.isdir (self.__RE_PARAMS.DevelSetPredictionOutputFolder)): 
                self.__PRJ.PROGRAM_Halt ("WriteBack for DevelSet predictions is requested, but DevelSetPredictionOutputFolder is not a valid folder address."); 
            else:
                self.__RE_PARAMS.DevelSetPredictionOutputFolder = self.__RE_PARAMS.DevelSetPredictionOutputFolder.replace ("\\" , "/"); 
                if self.__RE_PARAMS.DevelSetPredictionOutputFolder[-1] <> "/":
                    self.__RE_PARAMS.DevelSetPredictionOutputFolder+= "/"; 

        #Evaluate Test Set Params 
        if (self.__RE_PARAMS.PredictTestSet == True) or (self.__RE_PARAMS.EvaluateTestSet == True) or (self.__RE_PARAMS.WriteBackTestSetPredictions == True):
            if (self.__RE_PARAMS.TestSet_Files_Lists == None) or (not isinstance (self.__RE_PARAMS.TestSet_Files_Lists, list)) or (len(self.__RE_PARAMS.TestSet_Files_Lists)<1):
                self.__PRJ.PROGRAM_Halt ("TestSet_Files_Lists should be a list with a address of at least one TEES XML file."); 
        
        if (self.__RE_PARAMS.EvaluateTestSet == True) and (self.__RE_PARAMS.PredictTestSet == False):
            self.__PRJ.PROGRAM_Halt ("Evaluation of TestSet predictions is requested, But PredictTestSet is set to False.")
            
        if (self.__RE_PARAMS.WriteBackTestSetPredictions == True):
            if self.__RE_PARAMS.PredictTestSet == False:
                self.__PRJ.PROGRAM_Halt ("WriteBack of TestSet predictions is requested, But PredictTestSet is set to False.")
                
            if len(self.__RE_PARAMS.TestSet_Files_Lists)<>1: 
                self.__PRJ.PROGRAM_Halt ("WriteBack of TestSet predictions is requested, hence only 1 GOLD TestSet xml file should be given to the TestSet_Files_Lists param.")
            
            import os ; 
            if (self.__RE_PARAMS.TestSetPredictionOutputFolder == None) or (not isinstance (self.__RE_PARAMS.TestSetPredictionOutputFolder, str)) or \
               (not os.path.isdir (self.__RE_PARAMS.TestSetPredictionOutputFolder)): 
                self.__PRJ.PROGRAM_Halt ("WriteBack for TestSet predictions is requested, but TestSetPredictionOutputFolder is not a valid folder address."); 
            else:
                self.__RE_PARAMS.TestSetPredictionOutputFolder = self.__RE_PARAMS.TestSetPredictionOutputFolder.replace ("\\" , "/"); 
                if self.__RE_PARAMS.TestSetPredictionOutputFolder[-1] <> "/":
                    self.__RE_PARAMS.TestSetPredictionOutputFolder+= "/"; 
                
        #When to process devel/test set
        if (self.__RE_PARAMS.ProcessDevelSetAfterEpochNo >=0):
            if not self.__RE_PARAMS.ProcessDevelSetAfterEpochNo in range(self.__RE_PARAMS.NN_MaxNumberOfEpochs+1):
                self.__PRJ.PROGRAM_Halt ("ProcessDevelSetAfterEpochNo should be either -1 or between 0 and NN_MaxNumberOfEpoch !"); 
                
        if (self.__RE_PARAMS.ProcessTestSetAfterEpochNo >=0):
            if not self.__RE_PARAMS.ProcessTestSetAfterEpochNo in range(self.__RE_PARAMS.NN_MaxNumberOfEpochs+1):
                self.__PRJ.PROGRAM_Halt ("ProcessTestSetAfterEpochNo should be either -1 or between 0 and NN_MaxNumberOfEpoch !"); 
                
        #WriteBack Processed Training/Devel/Test GOLDs 
        if (self.__RE_PARAMS.WriteBackProcessedTrainingSet == True):
            if (self.__RE_PARAMS.WriteBackProcessedTrainingSet_OutputFileAddress == None):
                self.__PRJ.PROGRAM_Halt ("Writing back the processed training set is requested, but output file address is not given!"); 

            if len (self.__RE_PARAMS.TrainingSet_Files_List) <> 1:
                self.__PRJ.PROGRAM_Halt ("Writing back the processed training set is requested, but len(TrainingSet_Files_List) <> 1"); 
            
        if (self.__RE_PARAMS.WriteBackProcessedDevelopmentSet == True):
            if (self.__RE_PARAMS.WriteBackProcessedDevelopmentSet_OutputFileAddress == None):
                self.__PRJ.PROGRAM_Halt ("Writing back the processed development set is requested, but output file address is not given!"); 
                
            if len (self.__RE_PARAMS.DevelopmentSet_Files_List) <> 1:
                self.__PRJ.PROGRAM_Halt ("Writing back the processed development set is requested, but len(DevelopmentSet_Files_List) <> 1"); 
                
        if (self.__RE_PARAMS.WriteBackProcessedTestSet == True): 
            if (self.__RE_PARAMS.WriteBackProcessedTestSet_OutputFileAddress == None):
                self.__PRJ.PROGRAM_Halt ("Writing back the processed test set is requested, but output file address is not given!"); 
            
            if len(self.__RE_PARAMS.TestSet_Files_Lists) <> 1:
                self.__PRJ.PROGRAM_Halt ("Writing back the processed test set is requested, but len(TestSet_Files_Lists) <>1"); 
                
                
    def __PrepareNetworkInput__DATA (self):
        self.__PRJ.lp (["-"*80, "Preparing Network Input DATA for Training/Devel/Test Sets ... PLEASE WAIT ..." , "-"*80]) ; 
        TrainingSet_Files_List    = self.__RE_PARAMS.TrainingSet_Files_List ; 
        DevelopmentSet_Files_List = self.__RE_PARAMS.DevelopmentSet_Files_List ;
        TestSet_Files_Lists       = self.__RE_PARAMS.TestSet_Files_Lists ; 

        for idx , L in enumerate([TrainingSet_Files_List, DevelopmentSet_Files_List, TestSet_Files_Lists]):
            #Print what is going on ... 
            if idx == 0:
                Set = "TrainingSet" ; 
                DuplicationRemovalPolicy = self.__RE_PARAMS.TrainingSet_DuplicationRemovalPolicy 
                SkipSentences            = self.__RE_PARAMS.SkipWholeSentences_TrainingSet 
                
            elif idx ==1:
                Set = "DevelopmentSet" ;
                DuplicationRemovalPolicy = self.__RE_PARAMS.DevelopmentSet_DuplicationRemovalPolicy 
                SkipSentences            = self.__RE_PARAMS.SkipWholeSentences_DevelopmentSet 

            else:
                Set = "TestSet" ; 
                DuplicationRemovalPolicy = self.__RE_PARAMS.TestSet_DuplicationRemovalPolicy 
                SkipSentences            = self.__RE_PARAMS.SkipWholeSentences_TestSet 

            MSG = ["Preparing " + Set + " :"]; 
            for FileName in L:
               MSG.append ("    - " + FileName); 
            MSG.append ("-"*80); 
            self.__PRJ.lp (MSG);
            
            RES = [] ; 
            ALL_Sentences = [] ; 
            
            for FileAddress in L:
                Sentences , Root , LOCAL_TotalCorpusRootAnalysisResult , docs, docs_info  = self.__PRJ.Preprocessor.OnTheFlyPipeline (FileAddress,DuplicationRemovalPolicy,SkipSentences); 
                RES.append ({"Sentences":Sentences, "Root":Root, "LOCAL_TotalCorpusRootAnalysisResult":LOCAL_TotalCorpusRootAnalysisResult , "docs":docs, "docs_info":docs_info}); 
                ALL_Sentences.extend (Sentences); 
            
            if len(ALL_Sentences)>0:
                if (Set =="TrainingSet") and (self.__RE_PARAMS.ShuffleTrainingSetSentencesBeforeNIGeneration==True):
                    self.__PRJ.lp ("Note: Shuffling TrainingSet Sentences as requested..."); 
                    import numpy as np ; 
                    np.random.shuffle (ALL_Sentences); 
                    
                self.__PRJ.lp (["-"*40, "-"*40, "-"*40, "Now, Making network input from sentences:" , "Number of total sentences:" + str(len(ALL_Sentences))]); 
                NI = self.__PRJ.NetworkInputGenerator.Get_Data_ALL_FROM_Sentences (ALL_Sentences) ; 
                self.__DATA[Set]["ALL_RESULTS"]  = RES ; 
                self.__DATA[Set]["NETWORK_INPUT"]= NI  ; 

        for key in self.__DATA.keys():
            if self.__DATA[key]<>{}:
                self.__DATA[key]["NI"]={};
                self.__DATA[key]["NI"]["Bags_WordEmbeddings_B1"]          = self.__DATA[key]["NETWORK_INPUT"]["DS"]["Word_Matrices"][0]
                self.__DATA[key]["NI"]["Bags_WordEmbeddings_B2"]          = self.__DATA[key]["NETWORK_INPUT"]["DS"]["Word_Matrices"][1] 
                self.__DATA[key]["NI"]["Bags_WordEmbeddings_B3"]          = self.__DATA[key]["NETWORK_INPUT"]["DS"]["Word_Matrices"][2] 
                self.__DATA[key]["NI"]["Bags_PosTagEmbeddings_B1"]        = self.__DATA[key]["NETWORK_INPUT"]["DS"]["PosTag_Matrices"][0]
                self.__DATA[key]["NI"]["Bags_PosTagEmbeddings_B2"]        = self.__DATA[key]["NETWORK_INPUT"]["DS"]["PosTag_Matrices"][1]
                self.__DATA[key]["NI"]["Bags_PosTagEmbeddings_B3"]        = self.__DATA[key]["NETWORK_INPUT"]["DS"]["PosTag_Matrices"][2]
                self.__DATA[key]["NI"]["Bags_WordShapeFeatures_B1"]       = self.__DATA[key]["NETWORK_INPUT"]["DS"]["WordFeature_Matrices"][0]
                self.__DATA[key]["NI"]["Bags_WordShapeFeatures_B2"]       = self.__DATA[key]["NETWORK_INPUT"]["DS"]["WordFeature_Matrices"][1]
                self.__DATA[key]["NI"]["Bags_WordShapeFeatures_B3"]       = self.__DATA[key]["NETWORK_INPUT"]["DS"]["WordFeature_Matrices"][2]
                self.__DATA[key]["NI"]["PairEntityFeatures"]              = self.__DATA[key]["NETWORK_INPUT"]["DS"]["example_pair_entity_feature"]

                self.__DATA[key]["NI"]["Forward_SDP_WordEmbeddings"]      = self.__DATA[key]["NETWORK_INPUT"]["DS"]["SP_FEATURES"][0]
                self.__DATA[key]["NI"]["Forward_SDP_PosTagEmbeddings"]    = self.__DATA[key]["NETWORK_INPUT"]["DS"]["SP_FEATURES"][1]
                self.__DATA[key]["NI"]["Forward_SDP_DTEmbeddings"]        = self.__DATA[key]["NETWORK_INPUT"]["DS"]["SP_FEATURES"][2]
                
                self.__DATA[key]["NI"]["Backward_SDP_WordEmbeddings"]     = self.__DATA[key]["NETWORK_INPUT"]["DS"]["SP_FEATURES_REV"][0]
                self.__DATA[key]["NI"]["Backward_SDP_PosTagEmbeddings"]   = self.__DATA[key]["NETWORK_INPUT"]["DS"]["SP_FEATURES_REV"][1]
                self.__DATA[key]["NI"]["Backward_SDP_DTEmbeddings"]       = self.__DATA[key]["NETWORK_INPUT"]["DS"]["SP_FEATURES_REV"][2]

                self.__DATA[key]["NI"]["y_vector"]                        = self.__DATA[key]["NETWORK_INPUT"]["DS"]["y_vector"]
                self.__DATA[key]["NI"]["y_vector_reverse"]                = self.__DATA[key]["NETWORK_INPUT"]["DS"]["y_vector_reverse"]
                self.__DATA[key]["NI"]["y_vector_coarse"]                 = self.__DATA[key]["NETWORK_INPUT"]["DS"]["y_vector_coarse"]
                self.__DATA[key]["NI"]["y_vector_binary"]                 = self.__DATA[key]["NETWORK_INPUT"]["DS"]["y_vector_binary"]
                
                self.__DATA[key]["NI"]["PAIR_TRACKING"]                   = self.__DATA[key]["NETWORK_INPUT"]["PAIR_TRACKING"] 
    
    def __Assing_NetworkInput (self, key , Assign_OutPut=False):
        if not key in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("Invalid key for setting Neural Network Inputs:" + key); 

        if self.__WhichFeaturesToUse == None:
            self.__PRJ.PROGRAM_Halt ("self.__WhichFeaturesToUse Is None!"); 
        
        if self.__WhichOutputsToPredit == None:
            self.__PRJ.PROGRAM_Halt ("self.__WhichOutputsToPredit Us None!"); 
        
        touts = self.__WhichOutputsToPredit ; 
        if (touts["y_vector"] or touts["y_vector_reverse"] or touts["y_vector_coarse"] or touts["y_vector_binary"])==False:
            self.__PRJ.PROGRAM_Halt ("At least one of the labels in self.__WhichOutputsToPredit must be requested for prediction!"); 
            
        ANN_INPUT = {}; 

        if self.__WhichFeaturesToUse["Bags_WordEmbeddings_B1"]:
            ANN_INPUT["Bags_WordEmbeddings_B1"] = self.__DATA[key]["NI"]["Bags_WordEmbeddings_B1"]

        if self.__WhichFeaturesToUse["Bags_WordEmbeddings_B2"]:
            ANN_INPUT["Bags_WordEmbeddings_B2"] = self.__DATA[key]["NI"]["Bags_WordEmbeddings_B2"]

        if self.__WhichFeaturesToUse["Bags_WordEmbeddings_B3"]:
            ANN_INPUT["Bags_WordEmbeddings_B3"] = self.__DATA[key]["NI"]["Bags_WordEmbeddings_B3"]

        if self.__WhichFeaturesToUse["Bags_PosTagEmbeddings_B1"]: 
            ANN_INPUT["Bags_PosTagEmbeddings_B1"] = self.__DATA[key]["NI"]["Bags_PosTagEmbeddings_B1"]

        if self.__WhichFeaturesToUse["Bags_PosTagEmbeddings_B2"]: 
            ANN_INPUT["Bags_PosTagEmbeddings_B2"] = self.__DATA[key]["NI"]["Bags_PosTagEmbeddings_B2"]

        if self.__WhichFeaturesToUse["Bags_PosTagEmbeddings_B3"]: 
            ANN_INPUT["Bags_PosTagEmbeddings_B3"] = self.__DATA[key]["NI"]["Bags_PosTagEmbeddings_B3"]

        if self.__WhichFeaturesToUse["Bags_WordShapeFeatures_B1"]:
            ANN_INPUT["Bags_WordShapeFeatures_B1"] = self.__DATA[key]["NI"]["Bags_WordShapeFeatures_B1"] 

        if self.__WhichFeaturesToUse["Bags_WordShapeFeatures_B2"]:
            ANN_INPUT["Bags_WordShapeFeatures_B2"] = self.__DATA[key]["NI"]["Bags_WordShapeFeatures_B2"] 

        if self.__WhichFeaturesToUse["Bags_WordShapeFeatures_B3"]:
            ANN_INPUT["Bags_WordShapeFeatures_B3"] = self.__DATA[key]["NI"]["Bags_WordShapeFeatures_B3"] 
            
        if self.__WhichFeaturesToUse["PairEntityFeatures"]:
            ANN_INPUT["PairEntityFeatures"] = self.__DATA[key]["NI"]["PairEntityFeatures"]
        
        if self.__WhichFeaturesToUse["Forward_SDP_WordEmbeddings"]:
            ANN_INPUT["Forward_SDP_WordEmbeddings"]   = self.__DATA[key]["NI"]["Forward_SDP_WordEmbeddings"] 
            
        if self.__WhichFeaturesToUse["Forward_SDP_PosTagEmbeddings"]:
            ANN_INPUT["Forward_SDP_PosTagEmbeddings"] = self.__DATA[key]["NI"]["Forward_SDP_PosTagEmbeddings"] 
            
        if self.__WhichFeaturesToUse["Forward_SDP_DTEmbeddings"]:
            ANN_INPUT["Forward_SDP_DTEmbeddings"]     = self.__DATA[key]["NI"]["Forward_SDP_DTEmbeddings"] 

        if self.__WhichFeaturesToUse["Backward_SDP_WordEmbeddings"]:
            ANN_INPUT["Backward_SDP_WordEmbeddings"] = self.__DATA[key]["NI"]["Backward_SDP_WordEmbeddings"]
        
        if self.__WhichFeaturesToUse["Backward_SDP_PosTagEmbeddings"]:
            ANN_INPUT["Backward_SDP_PosTagEmbeddin"] = self.__DATA[key]["NI"]["Backward_SDP_PosTagEmbeddin"]

        if self.__WhichFeaturesToUse["Backward_SDP_DTEmbeddings"]:
            ANN_INPUT["Backward_SDP_DTEmbeddings"]   = self.__DATA[key]["NI"]["Backward_SDP_DTEmbeddings"] 
            
        if Assign_OutPut==True:
            ANN_OUTPUT = {} ; 
            if self.__WhichOutputsToPredit["y_vector"]:
                ANN_OUTPUT["y_vector"] = self.__DATA[key]["NI"]["y_vector"];

            if self.__WhichOutputsToPredit["y_vector_reverse"]:
                ANN_OUTPUT["y_vector_reverse"] = self.__DATA[key]["NI"]["y_vector_reverse"];

            if self.__WhichOutputsToPredit["y_vector_coarse"]:
                ANN_OUTPUT["y_vector_coarse"] = self.__DATA[key]["NI"]["y_vector_coarse"];

            if self.__WhichOutputsToPredit["y_vector_binary"]:
                ANN_OUTPUT["y_vector_binary"] = self.__DATA[key]["NI"]["y_vector_binary"];
            
        else:
            ANN_OUTPUT = None ; 
        
        return ANN_INPUT , ANN_OUTPUT ; 
    
    def __Compile_model(self):
        self.__PRJ.lp (["---------------- COMPILING MODEL ----------------", "OPTIMIZER:"+str(self.__RE_PARAMS.NN_Keras_optimizer)]); 

        #1) Handle Loss functions:
        temp_loss = {} ;
        if self.__WhichOutputsToPredit["y_vector"]:
            if self.__PRJ.Configs["ClassificationType"]=="binary":
                temp_loss["y_vector"]="binary_crossentropy" ;
            else:
                temp_loss["y_vector"]="categorical_crossentropy"
        
        if self.__WhichOutputsToPredit["y_vector_reverse"]:
            temp_loss["y_vector_reverse"]="categorical_crossentropy"
        

        if self.__WhichOutputsToPredit["y_vector_coarse"]:
            temp_loss["y_vector_coarse"]="categorical_crossentropy"
        
        if self.__WhichOutputsToPredit["y_vector_binary"]:
            temp_loss["y_vector_binary"]="binary_crossentropy"
        
        self.__model.compile (loss=temp_loss, optimizer=self.__RE_PARAMS.NN_Keras_optimizer) ; 
        self.__PRJ.lp ("--------------- DONE MODEL BUILDING ----------------"); 
        
    def __Train__model (self):
        if self.__DATA["TrainingSet"]=={}:
            self.__PRJ.PROGRAM_Halt ("Train function is called, however, TrainingSet is {}"); 
        
        if self.__model == None:
            self.__PRJ.PROGRAM_Halt ("Model should have been compiled before calling training."); 
            
        self.__RuntimeStatistics["NoEpochs"]+=1 ;
        self.__PRJ.lp ("---- TRAINING, ROUND:"+ str(self.__RuntimeStatistics["ExecutionRound"]) + " , EPOCH:" + str(self.__RuntimeStatistics["NoEpochs"])); 
        ANN_INPUT , ANN_OUTPUT = self.__Assing_NetworkInput ("TrainingSet" , Assign_OutPut=True); 
        self.H = self.__model.fit(ANN_INPUT,ANN_OUTPUT, batch_size= self.__RE_PARAMS.NN_Keras_batch_size, nb_epoch=1, verbose= self.__RE_PARAMS.KERAS_fit_verbose, class_weight= self.__RE_PARAMS.NN_Keras_class_weight) ; 
        self.__PRJ.lp ("Training loss: " + str(self.H.history)); 
        
    def __Predict (self, key):
        import numpy as np ; 

        if self.__model == None:
            self.__PRJ.PROGRAM_Halt ("self.__model is None!"); 

        if not key in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("Invalid key for setting Neural Network Inputs:" + key); 
        
        ANN_INPUT , ANN_OUTPUT = self.__Assing_NetworkInput (key, True)

        R = self.__model.predict (ANN_INPUT)
        if not isinstance(R, list):
            R = [R] ; 
        
        y_true , y_pred , confsc  = {} , {} , {} 
        
        #<<<CRITICAL>>> VERY VERY VERY CRITICAL ... 
        for index , output_name in enumerate(self.__model.output_names):
            if (output_name == "y_vector_binary") or (output_name == "y_vector" and self.__PRJ.Configs["ClassificationType"]=="binary"):
                y_true[output_name] = ANN_OUTPUT[output_name];
                confsc[output_name] = R[index].reshape(-1);
                y_pred[output_name] = np.around (confsc[output_name]); 
            else:
                y_true[output_name] = np.argmax (ANN_OUTPUT[output_name] , axis = -1); 
                confsc[output_name] = R[index] ;
                y_pred[output_name] = np.argmax (confsc[output_name], axis = -1) ; 
        
        if ("y_vector" in self.__model.output_names) and ("y_vector_reverse" in self.__model.output_names):
            self.__Predict_LinearCombination_Forward_Reverse (y_true, y_pred, confsc) 
            
        return y_true, y_pred, confsc;
    
    def __Predict_LinearCombination_Forward_Reverse (self, y_true, y_pred, confsc):
        import numpy as np ; 
        if (not "y_vector" in self.__model.output_names) or (not "y_vector_reverse" in self.__model.output_names):
            self.__PRJ.PROGRAM_Halt ("Error in calling function.") 
        
        if (not "y_vector" in y_true) or (not "y_vector_reverse" in y_true) or \
           (not "y_vector" in y_pred) or (not "y_vector_reverse" in y_pred) or \
           (not "y_vector" in confsc) or (not "y_vector_reverse" in confsc):
            self.__PRJ.PROGRAM_Halt ("Error in calling function.") 
        
        def Z(reverse_pred_confsc):
            classes = self.__PRJ.Configs["OneHotEncodingForMultiClass"] 
            converted_reverse_pred_confsc = np.zeros_like (reverse_pred_confsc)
            for class_lbl in classes.keys(): 
                if class_lbl.endswith ("(e1,e2)"):
                    convert_class_lbl = class_lbl.split("(e1,e2)")[0]+ "(e2,e1)"
                elif class_lbl.endswith ("(e2,e1)"):
                    convert_class_lbl = class_lbl.split("(e2,e1)")[0]+ "(e1,e2)"
                else:
                    convert_class_lbl = class_lbl ; 
                converted_reverse_pred_confsc[:,classes[convert_class_lbl]] = reverse_pred_confsc[:,classes[class_lbl]]
            return converted_reverse_pred_confsc ; 
        
        f_confsc = confsc["y_vector"]
        b_confsc = confsc["y_vector_reverse"] 
        alpha = 0.65 
        lincomb_confsc = (alpha * f_confsc) + ((1-alpha)*Z(b_confsc))
        lincomb_y_pred = np.argmax (lincomb_confsc, axis = -1) ; 
        y_true["y_vector_lincomb"]= y_true["y_vector"] # <<<CRITICAL>>> linear combination will be evaluated against the forward (normal) path 
        y_pred["y_vector_lincomb"]= lincomb_y_pred
        confsc["y_vector_lincomb"]= lincomb_confsc 
        
    def __Evaluate_Binary (self, y_true, y_pred, confs, MSG):
        import numpy as np ; 
        #Binary Classification Evaluation
        Precision, Recall, FScore, Support = METRICS.precision_recall_fscore_support (y_true, y_pred);
        COUNTS = METRICS.confusion_matrix (y_true , y_pred); 
        TP = COUNTS[1,1];
        TN = COUNTS[0,0];
        FP = COUNTS[0,1]; 
        FN = COUNTS[1,0];
        RES = { 
          "positive" : {"precision": float(GF.FDecimalPoints(Precision[1], self.__DecimalPoints)),
                        "recall"   : float(GF.FDecimalPoints(Recall[1]   , self.__DecimalPoints)), 
                        "f1_score" : float(GF.FDecimalPoints(FScore[1]   , self.__DecimalPoints)),
                        "support"  : Support[1]
                        },
          "negative" : {"precision": float(GF.FDecimalPoints(Precision[0], self.__DecimalPoints)),
                        "recall"   : float(GF.FDecimalPoints(Recall[0]   , self.__DecimalPoints)), 
                        "f1_score" : float(GF.FDecimalPoints (FScore[0]  , self.__DecimalPoints)), 
                        "support": Support[0]
                        },
          "total"    : {"f1_macro"   : float(GF.FDecimalPoints(np.mean (FScore), self.__DecimalPoints)) , 
                        "f1_weighted": float(GF.FDecimalPoints( (((FScore[0]*Support[0])+(FScore[1]*Support[1]))/float(Support[0]+Support[1])) , self.__DecimalPoints))
                       } ,
          "TP" : TP , "TN" : TN , "FP" : FP , "FN" : FN ,
          "y_pred" : y_pred, 
          "confs"  : confs ,
        } 
        
        MSG.append ("Negative:" +  \
                    "    f1_score: "  + str(RES["negative"]["f1_score"])  + \
                    "    precision: " + str(RES["negative"]["precision"]) + \
                    "    recall: "    + str(RES["negative"]["recall"])    + \
                    "    support: "    + str(RES["negative"]["support"])); 
        MSG.append ("Positive:" +  \
                    "    f1_score: "  + str(RES["positive"]["f1_score"])  + \
                    "    precision: " + str(RES["positive"]["precision"]) + \
                    "    recall: "    + str(RES["positive"]["recall"])    + \
                    "    support: "    + str(RES["positive"]["support"])); 
        MSG.append ("total:" +  \
                    "    f1_macro: "   + str(RES["total"]["f1_macro"])  + \
                    "    f1_weighted: " + str(RES["total"]["f1_weighted"]));
        MSG.append ("TP, FP, TN, FN : " + str(RES["TP"]) + " , " + str(RES["FP"]) + " , " + str(RES["TN"]) + " , " + str(RES["FN"])) ; 
        
    def __Evaluate_Multiclass (self, y_true, y_pred, confs, CLASSES, MSG):
        
        IGNORE_CLASS_LABELS_FOR_EVALUATION = self.__PRJ.Configs["EvaluationParameters"]["ExcludeClassLabelsList"]
        INV_CLASSES = {} 
        for key in CLASSES:
            INV_CLASSES[CLASSES[key]] = key ; 
        
        PerClass_Precision, PerClass_Recall, PerClass_FScore, PerClass_Support = METRICS.precision_recall_fscore_support (y_true, y_pred, labels = range(len(CLASSES))) ;

        for i in range(len(CLASSES)):
            precision = GF.FDecimalPoints (PerClass_Precision[i] , self.__DecimalPoints) 
            recall    = GF.FDecimalPoints (PerClass_Recall[i]    , self.__DecimalPoints) 
            f1score   = GF.FDecimalPoints (PerClass_FScore[i]    , self.__DecimalPoints) 
            support   = str ( PerClass_Support[i]) ; 
            MSG.append (GF.NVLR(INV_CLASSES[i] + ":", 30) + "    f1-score: " + f1score + "    precision: " + precision + "    " + "recall: " + recall + "    " + "    " + "support: " + support )
            #RES[INV_CLASSES[i]] = {"precision":float(precision), "recall":float(recall) , "f1-score":float(f1score), "support":float(support)}
        
        f1_macro     = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred , labels = range(len(CLASSES)) , average = "macro")    , self.__DecimalPoints)
        f1_micro     = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred , labels = range(len(CLASSES)) , average = "micro")    , self.__DecimalPoints)
        f1_weighted  = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred , labels = range(len(CLASSES)) , average = "weighted") , self.__DecimalPoints)
        
        MSG.extend (["total:","f1_macro:" + f1_macro + "    f1_micro:" + f1_micro + "    f1-weighted:" + f1_weighted]); 
        
        if len(IGNORE_CLASS_LABELS_FOR_EVALUATION)>0:
            # Nice, but does not work for negative classes  ... 
            #     IG_IDX = [CLASSES[i.lower()] for i in IGNORE_CLASS_LABELS_FOR_EVALUATION]
            #     IG_LBL = "total_MINUS_" + "_".join ([INV_CLASSES[i] for i in IG_IDX]) ; 

            # <<<TODO>>>
            # <<<CRITICAL>>> 
            IG_IDX = [] ;                 
            for igc in IGNORE_CLASS_LABELS_FOR_EVALUATION:
                igc = igc.lower(); 
                if igc in CLASSES:
                    IG_IDX.append (CLASSES[igc]); 
                else:
                    if igc in self.__PRJ.Configs["CLASSES"]["Negative"]:
                        if not 0 in IG_IDX:
                            IG_IDX.append(0); 
                    
            IG_LBL = "total_MINUS_" + "_".join ([INV_CLASSES[i] for i in IG_IDX]) ; 
            REMAIN_IDX = [i for i in range(len(CLASSES)) if not i in IG_IDX]
            f1_macro     = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred , labels = REMAIN_IDX , average = "macro")    , self.__DecimalPoints)
            f1_micro     = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred , labels = REMAIN_IDX , average = "micro")    , self.__DecimalPoints)
            f1_weighted  = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred , labels = REMAIN_IDX , average = "weighted") , self.__DecimalPoints)
            #RES[IG_LBL] = {"f1_macro":float(f1_macro), "f1_micro":float(f1_micro), "f1_weighted":float(f1_weighted)}
            MSG.extend ([IG_LBL +":", "f1_macro:" + f1_macro + "    f1_micro:" + f1_micro + "    f1-weighted:" + f1_weighted]); 
        
    def __Evaluate (self, y_true, y_pred, confs, SetName = None, FalseNegatives = None):
        if SetName == None:
            SetName = "Unknown input" ; 
        
        if not (len(y_true)==len(y_pred)==len(confs)):
            self.__PRJ.PROGRAM_Halt ("Length of y_true, y_pred, confs should be the same!");
        
        if not (set(y_true.keys())==set(y_pred.keys())==set(confs.keys())):
            self.__PRJ.PROGRAM_Halt ("y_true, y_pred, confs should have same class labels!");
        
        if self.__model == None:
            self.__PRJ.PROGRAM_Halt ("model is None!") ; 

        MSG = ["Evaluating on " + SetName + ":"] 

        for output_name in ["y_vector", "y_vector_reverse" , "y_vector_coarse", "y_vector_binary" , "y_vector_lincomb"]:
            if not output_name in y_true.keys():
                continue ;

            MSG.extend(["", "="*110, "OUTPUT:" + output_name , "-"*10]); 
            if (output_name == "y_vector_binary") or (output_name == "y_vector" and self.__PRJ.Configs["ClassificationType"]=="binary"):
                self.__Evaluate_Binary(y_true[output_name], y_pred[output_name], confs[output_name], MSG); 
                
            else: #Multiclass
                if output_name in ["y_vector", "y_vector_reverse", "y_vector_lincomb"]:
                    CLASSES = self.__PRJ.Configs["OneHotEncodingForMultiClass"] 
                elif output_name == "y_vector_coarse":
                    CLASSES = self.__PRJ.Configs["OneHotEncodingForCoarseMultiClass"]
                self.__Evaluate_Multiclass (y_true[output_name], y_pred[output_name], confs[output_name], CLASSES, MSG);
        
        self.__PRJ.lp (MSG)
                    
    def __WriteBackSetPredictionResults (self, y_pred, confs, PAIR_TRACKING, GOLD_XML_FileAddress, OutputFileAddress):
        if len(y_pred) <> len(confs):
            self.__PRJ.PROGRAM_Halt ("len(y_pred) is not equal to len(confs).")
        
        if len(y_pred) <> len(PAIR_TRACKING):
            self.__PRJ.PROGRAM_Halt ("Inconsistency between y_pred and PAIR_TRACKING!")
        
        TEESXMLHelper = TEESDocumentHelper.FLSTM_TEESXMLHelper (self.__PRJ.Configs , self.__PRJ.lp , self.__PRJ.PROGRAM_Halt) ; 
        if self.__PRJ.Configs["ClassificationType"]=="binary":
            NEGATIVE_LABEL = list (self.__PRJ.Configs["WRITEBACK_CLASSES"]["Negative"])[0]; 
            POSITIVE_LABEL = list (self.__PRJ.Configs["WRITEBACK_CLASSES"]["Positive"])[0]; 
            labels = [NEGATIVE_LABEL if i==0 else POSITIVE_LABEL for i in y_pred] ; 

        else:
            #<<<TODO>>> PLEASE PLEASE TEST .... 
            #<<<CRITICAL>>> NOT TESTED .... 
            MULTICLASS_LABELS_DICT = {}; 
            for LABEL in self.__PRJ.Configs["WRITEBACK_OneHotEncodingForMultiClass"]:
                INDEX = self.__PRJ.Configs["WRITEBACK_OneHotEncodingForMultiClass"][LABEL]
                MULTICLASS_LABELS_DICT[INDEX] = LABEL ; 
            labels = [MULTICLASS_LABELS_DICT[i] for i in y_pred] ;
        
        WRITE_BACK_RES = [] ; 
        for idx , [S_ID, T_ID, E1_ID, E2_ID, E1_TP, E2_TP] in enumerate (PAIR_TRACKING):
            TYPE = labels[idx]; 

            if self.__PRJ.Configs["ClassificationType"]=="binary":
                """
                    if confs >= 0.5 --> class = positive ==> positive_confs = confs , negative_confs = 1 - positive_confs
                    if confs <  0.5 --> class = negative ==> positive_confs = confs , negative_confs = 1 - positive_confs
                """
                positive_conf = confs[idx]
                negative_conf = 1 - positive_conf 
                CONFS_STR = POSITIVE_LABEL + ":" + str(positive_conf) + "," + NEGATIVE_LABEL + ":"+str(negative_conf)
            
            else:
                CONFS = confs[idx,:];
                CONFS_STR = "" ; 
                for c in sorted(MULTICLASS_LABELS_DICT.keys()):
                    CONFS_STR += MULTICLASS_LABELS_DICT[c]+":"+str(CONFS[c])+",";
                CONFS_STR = CONFS_STR[:-1]; 
            #print (S_ID,T_ID,TYPE,CONF, E1_TP, E2_TP) ;
            WRITE_BACK_RES.append ((S_ID,T_ID,TYPE,CONFS_STR, E1_ID, E2_ID, E1_TP,E2_TP)) ;
        
        GOLD_XML_TREE = TEESXMLHelper.LoadTEESFile_GetTreeRoot (GOLD_XML_FileAddress, ReturnWholeTree=True); 
        #TEESXMLHelper.TREEOPERATION_FILL_Version1 (GOLD_XML_TREE , WRITE_BACK_RES) ; 
        TEESXMLHelper.TREEOPERATION_FILL_Version2 (GOLD_XML_TREE , WRITE_BACK_RES, RemoveAllParseInfo=True ) ; #RemoveAllParseInfo, so that file size decrease dramatically ... no need to have parse info in pred files.
        self.__PRJ.lp ("WRITING BACK PREDICTION RESULTS TO FILE:" + OutputFileAddress) ; 
        TEESXMLHelper.Save_TEES_File (GOLD_XML_TREE , OutputFileAddress);
        return GOLD_XML_TREE ; 


    def __HandleDeveltSet (self):
        if self.__RE_PARAMS.PredictDevelSet:
            y_true, y_pred, confs  = self.__Predict ("DevelopmentSet");
            
            if self.__RE_PARAMS.EvaluateDevelSet:
                self.__Evaluate (y_true, y_pred, confs, SetName = "DevelopmentSet", FalseNegatives = None);
                        
            if self.__RE_PARAMS.WriteBackDevelSetPredictions: 
                DevelSetPredictionOutputFolder = self.__RE_PARAMS.DevelSetPredictionOutputFolder + self.__RuntimeStatistics["SelectedArchitecture"] + "/" ;
                Round                          = self.__RuntimeStatistics["ExecutionRound"];  
                RandomSeed                     = self.__RuntimeStatistics["RandomSeed"] ; 
                EpochNo                        = self.__RuntimeStatistics["NoEpochs"] ;
                OutputFileAddressName = DevelSetPredictionOutputFolder + "Pred_Round" + str(Round) + "_Seed" + str(RandomSeed) + "_EpochNo" + str(EpochNo) + ".xml" ; 
                #<<<CRITICAL>>> We only support Writing Back for 1 devel test FILE at a time ... 
                PAIR_TRACKING         = self.__DATA["DevelopmentSet"]["NI"]["PAIR_TRACKING"]
                GOLD_XML_FileAddress  = self.__RE_PARAMS.DevelopmentSet_Files_List[0] ; 
                #{END}<<<CRITICAL>>>
                self.__WriteBackSetPredictionResults (y_pred["y_vector"], confs["y_vector"], PAIR_TRACKING, GOLD_XML_FileAddress, OutputFileAddressName);

                if ("y_vector_lincomb" in y_pred):
                    OutputFileAddressName = DevelSetPredictionOutputFolder + "LinComb_Round" + str(Round) + "_Seed" + str(RandomSeed) + "_EpochNo" + str(EpochNo) + ".xml" ;                     
                    self.__WriteBackSetPredictionResults (y_pred["y_vector_lincomb"], confs["y_vector_lincomb"], PAIR_TRACKING, GOLD_XML_FileAddress, OutputFileAddressName);

    def __HandleTestSet (self):
        if self.__RE_PARAMS.PredictTestSet:
            y_true, y_pred, confs  = self.__Predict ("TestSet");
            
            if self.__RE_PARAMS.EvaluateTestSet:
                self.__Evaluate (y_true, y_pred, confs, SetName = "TestSet", FalseNegatives = None);
                        
            if self.__RE_PARAMS.WriteBackTestSetPredictions: 
                TestSetPredictionOutputFolder = self.__RE_PARAMS.TestSetPredictionOutputFolder + self.__RuntimeStatistics["SelectedArchitecture"] + "/" ;
                Round                         = self.__RuntimeStatistics["ExecutionRound"];  
                RandomSeed                    = self.__RuntimeStatistics["RandomSeed"] ; 
                EpochNo                       = self.__RuntimeStatistics["NoEpochs"] ;
                OutputFileAddressName = TestSetPredictionOutputFolder + "Pred_Round" + str(Round) + "_Seed" + str(RandomSeed) + "_EpochNo" + str(EpochNo) + ".xml" ; 
                #<<<CRITICAL>>> We only support Writing Back for 1 test test FILE at a time ... 
                PAIR_TRACKING         = self.__DATA["TestSet"]["NI"]["PAIR_TRACKING"]
                GOLD_XML_FileAddress  = self.__RE_PARAMS.TestSet_Files_Lists[0] ; 
                #{END}<<<CRITICAL>>>
                self.__WriteBackSetPredictionResults (y_pred["y_vector"], confs["y_vector"], PAIR_TRACKING, GOLD_XML_FileAddress, OutputFileAddressName);

                if ("y_vector_lincomb" in y_pred):
                    OutputFileAddressName = TestSetPredictionOutputFolder + "LinComb_Round" + str(Round) + "_Seed" + str(RandomSeed) + "_EpochNo" + str(EpochNo) + ".xml" ;                     
                    self.__WriteBackSetPredictionResults (y_pred["y_vector_lincomb"], confs["y_vector_lincomb"], PAIR_TRACKING, GOLD_XML_FileAddress, OutputFileAddressName);
                    
                
    def __WriteBackProcessedGOLD (self, Set):
        import numpy as np ; 
        if not Set in ["TrainingSet", "DevelopmentSet" , "TestSet"]:
            self.__PRJ.PROGRAM_Halt ("Unknown Set is requested: " + Set) ;
        
        y_true        = self.__DATA[Set]["NI"]["y_vector"]
        PAIR_TRACKING = self.__DATA[Set]["NI"]["PAIR_TRACKING"]

        if Set == "TrainingSet":
            GOLD_XML_FileAddress = self.__RE_PARAMS.TrainingSet_Files_List[0]
            OutputFileAddress    = self.__RE_PARAMS.WriteBackProcessedTrainingSet_OutputFileAddress
        
        elif Set == "DevelopmentSet":
            GOLD_XML_FileAddress = self.__RE_PARAMS.DevelopmentSet_Files_List[0]
            OutputFileAddress    = self.__RE_PARAMS.WriteBackProcessedDevelopmentSet_OutputFileAddress

        else:            
            GOLD_XML_FileAddress = self.__RE_PARAMS.TestSet_Files_Lists[0]
            OutputFileAddress    = self.__RE_PARAMS.WriteBackProcessedTestSet_OutputFileAddress

        self.__PRJ.lp ("WRITING BACK PROCESSED GOLD " + Set + " TO FILE:" + OutputFileAddress) ; 

        TEESXMLHelper = TEESDocumentHelper.FLSTM_TEESXMLHelper (self.__PRJ.Configs , self.__PRJ.lp , self.__PRJ.PROGRAM_Halt) ; 
        if self.__PRJ.Configs["ClassificationType"]=="binary":
            NEGATIVE_LABEL = list (self.__PRJ.Configs["WRITEBACK_CLASSES"]["Negative"])[0]; 
            POSITIVE_LABEL = list (self.__PRJ.Configs["WRITEBACK_CLASSES"]["Positive"])[0]; 
            labels = [NEGATIVE_LABEL if i==0 else POSITIVE_LABEL for i in y_true] ; 

        else:
            #<<<TODO>>> PLEASE PLEASE TEST .... 
            #<<<CRITICAL>>> NOT TESTED .... 
            MULTICLASS_LABELS_DICT = {}; 
            for LABEL in self.__PRJ.Configs["WRITEBACK_OneHotEncodingForMultiClass"]:
                INDEX = self.__PRJ.Configs["WRITEBACK_OneHotEncodingForMultiClass"][LABEL]
                MULTICLASS_LABELS_DICT[INDEX] = LABEL ; 
            labels = [MULTICLASS_LABELS_DICT[np.argmax(i)] for i in y_true] ;
        
        WRITE_BACK_RES = [] ; 
        for idx , [S_ID, T_ID, E1_ID, E2_ID, E1_TP, E2_TP] in enumerate (PAIR_TRACKING):
            TYPE = labels[idx]; 
            CONFS_STR = None
            WRITE_BACK_RES.append ((S_ID,T_ID,TYPE,CONFS_STR, E1_ID, E2_ID, E1_TP,E2_TP)) ;
        
        GOLD_XML_TREE = TEESXMLHelper.LoadTEESFile_GetTreeRoot (GOLD_XML_FileAddress, ReturnWholeTree=True); 
        #TEESXMLHelper.TREEOPERATION_FILL_Version1 (GOLD_XML_TREE , WRITE_BACK_RES) ; 
        TEESXMLHelper.TREEOPERATION_FILL_Version2 (GOLD_XML_TREE , WRITE_BACK_RES) ; 
        TEESXMLHelper.Save_TEES_File (GOLD_XML_TREE , OutputFileAddress);

    def Run(self):
        self.__PRJ.lp ("Running Training/Evaluation Pipeline..."); 
        self.__PRJ.lp ("ANN Architectures:" + str(self.__RE_PARAMS.ArchitectureBuilderMethodName)); 
        self.__PrepareNetworkInput__DATA(); 
        
        if self.__RE_PARAMS.WriteBackProcessedTrainingSet:
            self.__WriteBackProcessedGOLD ("TrainingSet") 

        if self.__RE_PARAMS.WriteBackProcessedDevelopmentSet:
            self.__WriteBackProcessedGOLD ("DevelopmentSet") 
        
        if self.__RE_PARAMS.WriteBackProcessedTestSet:
            self.__WriteBackProcessedGOLD ("TestSet") 
            
        for ARCHITECTURE in self.__RE_PARAMS.ArchitectureBuilderMethodName:

            #create DEVELOPMENT-SET predictin output SUB-folder (Architecture based)
            if (self.__RE_PARAMS.WriteBackDevelSetPredictions == True):
                    DevelSetPredictionOutputFolder = self.__RE_PARAMS.DevelSetPredictionOutputFolder + ARCHITECTURE + "/" ; 
                    import os; 
                    if not os.path.exists(DevelSetPredictionOutputFolder):
                        try:
                            self.__PRJ.lp (["Creating DevelSet predictions output folder:" , "   - " + DevelSetPredictionOutputFolder]); 
                            os.makedirs(DevelSetPredictionOutputFolder);
                        except Exception as E:
                            self.__PRJ.PROGRAM_Halt ("Error creating DevelSet predictions output folder. Error:" + E.message); 

            #create TEST-SET predictin output SUB-folder (Architecture based)
            if (self.__RE_PARAMS.WriteBackTestSetPredictions == True):
                    TestSetPredictionOutputFolder = self.__RE_PARAMS.TestSetPredictionOutputFolder + ARCHITECTURE + "/" ; 
                    import os; 
                    if not os.path.exists(TestSetPredictionOutputFolder):
                        try:
                            self.__PRJ.lp (["Creating TestSet predictions output folder:" , "   - " + TestSetPredictionOutputFolder]); 
                            os.makedirs(TestSetPredictionOutputFolder);
                        except Exception as E:
                            self.__PRJ.PROGRAM_Halt ("Error creating TestSet predictions output folder. Error:" + E.message); 
                    
            for COUNTER in range (self.__RE_PARAMS.HowManyClassifiers):
                self.__ResetTraining();

                #<<<CRITICAL>>> Importing should be done everytime I guess. 
                from Architectures import FLSTM_RE_Architectures ;  
                if COUNTER == 0:
                    Temp__Model_Builder_Class = FLSTM_RE_Architectures(self.__DATA["TrainingSet"]["NI"] , self.__PRJ.wv , self.__PRJ.lp , self.__PRJ.PROGRAM_Halt, self.__PRJ.Configs , RandomSeed=110); 
                else:
                    Temp__Model_Builder_Class = FLSTM_RE_Architectures(self.__DATA["TrainingSet"]["NI"] , self.__PRJ.wv , self.__PRJ.lp , self.__PRJ.PROGRAM_Halt, self.__PRJ.Configs); 

                self.__RuntimeStatistics["SelectedArchitecture"] = ARCHITECTURE ; 
                self.__RuntimeStatistics["ExecutionRound"]       = COUNTER+1 ; 
                self.__RuntimeStatistics["RandomSeed"]           = str(Temp__Model_Builder_Class.RandomSeed) ; 
                self.__PRJ.lp (["-"*40,"-","-" , "ARCHITECTURE:" + ARCHITECTURE + " ,  ROUND NO:" + str(COUNTER+1) , "-","-","-"*40]);
                
                if ARCHITECTURE[-1] <> ")":
                    self.__model , self.__WhichFeaturesToUse , self.__WhichOutputsToPredit = eval ("Temp__Model_Builder_Class." + ARCHITECTURE + "()");
                else:
                    self.__model , self.__WhichFeaturesToUse , self.__WhichOutputsToPredit = eval ("Temp__Model_Builder_Class." + ARCHITECTURE );
                    
                self.__Compile_model(); 
                
                for EPOCH_COUNTER in range (self.__RE_PARAMS.NN_MaxNumberOfEpochs):
                    #Train model for 1 epoch 
                    self.__Train__model (); 

                    #Process DevelopmentSet if requested ...
                    if (self.__RE_PARAMS.ProcessDevelSetAfterEpochNo >=0) and ((EPOCH_COUNTER+1) >= self.__RE_PARAMS.ProcessDevelSetAfterEpochNo):
                        self.__HandleDeveltSet ()
                        
                    #Process TestSet if requested ...
                    if (self.__RE_PARAMS.ProcessTestSetAfterEpochNo >=0) and ((EPOCH_COUNTER+1) >= self.__RE_PARAMS.ProcessTestSetAfterEpochNo):
                        self.__HandleTestSet ()

                self.__PRJ.lp (["-"*25 + "   STOP TRAINING   " + "-"*25 , "-"*80]); 
                
                #If self.ProcessTestSetAfterEpochNo>=0, then TestSet is already processed in the last iteration of the loop, so no need to redo it. 
                #If self.ProcessTestSetAfterEpochNo<0 , then it means it has never been processesed in the loop, so, after all training epochs are done, procees it.
                if (self.__RE_PARAMS.ProcessTestSetAfterEpochNo <0):
                    self.__HandleTestSet () ; 

    def Run_ProcessAndWriteBackGOLDS(self):
        #Call this only when you want to prepare gold-files (after duplicate removal/negative generation) without running training/prediction pipeline
        #<<<CRITICAL>>>: Only first file of the list of files will be processed ...
        # e.g., Trainingset = [f1,f2,f3,f5][0]
        self.__PRJ.lp ("Processing And Writing Back GOLDS ..."); 
        self.__PrepareNetworkInput__DATA(); 

        if self.__RE_PARAMS.WriteBackProcessedTrainingSet:
            self.__WriteBackProcessedGOLD ("TrainingSet") 

        if self.__RE_PARAMS.WriteBackProcessedDevelopmentSet:
            self.__WriteBackProcessedGOLD ("DevelopmentSet") 
        
        if self.__RE_PARAMS.WriteBackProcessedTestSet:
            self.__WriteBackProcessedGOLD ("TestSet") 
                