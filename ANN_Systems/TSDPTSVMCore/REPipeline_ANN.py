import os
import numpy as np 
import sklearn.metrics as METRICS 
from collections import OrderedDict

try:
    import cPickle as pickle 
except:
    import pickle 

import TEESDocumentHelper
import ProjectRunner 
import FeatureGeneration as FeGen
import GeneralFunctions as GF
import SharedFunctions as SF 

class TSP_TANN_RE_PARAMS:
    def __init__(self):
        self.TrainingSet_Files_List    = []
        self.DevelopmentSet_Files_List = []
        self.TestSet_Files_Lists       = []

        self.TrainingSet_DuplicationRemovalPolicy    = None #None: select according to the given config file, no set-specific policy
        self.DevelopmentSet_DuplicationRemovalPolicy = None #None: select according to the given config file, no set-specific policy
        self.TestSet_DuplicationRemovalPolicy        = None #None: select according to the given config file, no set-specific policy

        self.KERAS_shuffle_training_examples = True  #<<<CRITICAL>>> Is set to true, hence not to optimize on a particular order of sentences in the Training set. Set to False if you really really want to be able to replicate results later. 
        self.KERAS_minibatch_size = 32 
        self.KERAS_class_weight   = None 
        self.KERAS_optimizer      = "nadam" 
        self.KERAS_optimizer_lr   = 0.0005
        self.KERAS_fit_verbose    = 2 

        self.Classification_OptimizationMetric = "positive_f1_score" 
        self.Classification_OptimizationOutput = "Y" 
        self.Classification_ExternalEvaluator  = None 
        
        self.PredictDevelSet  = True
        self.EvaluateDevelSet = True
        self.WriteBackDevelSetPredictions  = False 
        self.DevelSetPredictionOutputFolder= None
        self.ProcessDevelSetAfterEpochNo =  0 #After epoch no X, DevelSet is predicted and evaluated. [Hint: set to -1 to NEVER predict develset]

        self.PredictTestSet  = False 
        self.EvaluateTestSet = False 
        self.WriteBackTestSetPredictions   = False 
        self.TestSetPredictionOutputFolder = None
        self.ProcessTestSetAfterEpochNo  = -1  #After epoch no X, IF needed, Test set is going to be predicted and/or evaluated and/or written-back. [Hint: -1 to Never process EXCEPT the last epoch!]

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
            "TestSet_Files_Lists            :" + str(self.TestSet_Files_Lists)             + "\n\n" + \
            \
            "TrainingSet_DuplicationRemovalPolicy     :" + str(self.TrainingSet_DuplicationRemovalPolicy)    + "\n" + \
            "DevelopmentSet_DuplicationRemovalPolicy  :" + str(self.DevelopmentSet_DuplicationRemovalPolicy) + "\n" + \
            "TestSet_DuplicationRemovalPolicy         :" + str(self.TestSet_DuplicationRemovalPolicy)        + "\n\n" + \
            \
            "KERAS_shuffle_training_examples   :" + str(self.KERAS_shuffle_training_examples) + "\n" + \
            "KERAS_minibatch_size              :" + str(self.KERAS_minibatch_size) + "\n" + \
            "KERAS_class_weight                :" + str(self.KERAS_class_weight)   + "\n" + \
            "KERAS_optimizer                   :" + str(self.KERAS_optimizer)      + "\n" + \
            "KERAS_optimizer_lr                :" + str(self.KERAS_optimizer_lr)   + "\n" + \
            "KERAS_fit_verbose                 :" + str(self.KERAS_fit_verbose)    + "\n\n" + \
            \
            "Classification_OptimizationMetric :" + str(self.Classification_OptimizationMetric) + "\n" + \
            "Classification_OptimizationOutput :" + str(self.Classification_OptimizationOutput) + "\n" + \
            "Classification_ExternalEvaluator  :" + str(self.Classification_ExternalEvaluator)  + "\n\n" + \
            \
            "PredictDevelSet                :" + str(self.PredictDevelSet)                 + "\n" + \
            "EvaluateDevelSet               :" + str(self.EvaluateDevelSet)                + "\n" + \
            "WriteBackDevelSetPredictions   :" + str(self.WriteBackDevelSetPredictions)    + "\n" + \
            "DevelSetPredictionOutputFolder :" + str(self.DevelSetPredictionOutputFolder)  + "\n" + \
            "ProcessDevelSetAfterEpochNo    :" + str(self.ProcessDevelSetAfterEpochNo)     + "\n\n" + \
            \
            "PredictTestSet                 :" + str(self.PredictTestSet)                  + "\n" + \
            "EvaluateTestSet                :" + str(self.EvaluateTestSet)                 + "\n" + \
            "WriteBackTestSetPredictions    :" + str(self.WriteBackTestSetPredictions)     + "\n" + \
            "TestSetPredictionOutputFolder  :" + str(self.TestSetPredictionOutputFolder)   + "\n" + \
            "ProcessTestSetAfterEpochNo     :" + str(self.ProcessTestSetAfterEpochNo)      + "\n\n" + \
            \
            "SkipWholeSentences_TrainingSet    :" + str(self.SkipWholeSentences_TrainingSet)    + "\n" + \
            "SkipWholeSentences_DevelopmentSet :" + str(self.SkipWholeSentences_DevelopmentSet) + "\n" + \
            "SkipWholeSentences_TestSet        :" + str(self.SkipWholeSentences_TestSet)        + "\n\n" + \
            \
            "WriteBackProcessedTrainingSet                :" + str(self.WriteBackProcessedTrainingSet) + "\n" + \
            "WriteBackProcessedDevelopmentSet             :" + str(self.WriteBackProcessedDevelopmentSet) + "\n" + \
            "WriteBackProcessedTestSet                    :" + str(self.WriteBackProcessedTestSet) + "\n" + \
            "WriteBackProcessedTrainingSet_OutputFileAddress    :" + str(self.WriteBackProcessedTrainingSet_OutputFileAddress) + "\n" + \
            "WriteBackProcessedDevelopmentSet_OutputFileAddress :" + str(self.WriteBackProcessedDevelopmentSet_OutputFileAddress) + "\n" + \
            "WriteBackProcessedTestSet_OutputFileAddress        :" + str(self.WriteBackProcessedTestSet_OutputFileAddress) + "\n" ;
        return S ; 
        
       
class TSP_TANN_RE_Pipeline:
    def __init__(self, ConfigFileAddress, LogFileAddress, RE_PARAMS):
        self.__PRJ   = ProjectRunner.TSDP_TSVM_ProjectRunner (ConfigFileAddress, LogFileAddress) 
        self.__DATA  = {"TrainingSet":{}, "DevelopmentSet":{}, "TestSet": {}}
        self.__GOLD  = {"TrainingSet":{"positives":{}, "negatives":{}}, "DevelopmentSet":{"positives":{}, "negatives":{}}, "TestSet": {"positives":{}, "negatives":{}}}
        self.__model = None 
        self.__WhichFeaturesToUse   = None 
        self.__WhichOutputsToPredit = None 
        self.__ResetTraining (None)
        self.__ResetMatrices ()
        self.__DecimalPoints = self.__PRJ.Configs["EvaluationParameters"]["DecimalPoints"]
        if not isinstance(RE_PARAMS , TSP_TANN_RE_PARAMS):
            self.__PRJ.PROGRAM_Halt ("RE_PARAMS should be an instance of TSP_TANN_RE_PARAMS")
        else:
            self.__RE_PARAMS = RE_PARAMS 
            self.__Verify_RE_PARAMS()
            self.__PRJ.lp (["-"*40, "RE_PARAMS:", "-" * 10, str(self.__RE_PARAMS) , "-"*40])
            
    def __exit__(self):
        self.__PRJ.__exit__(); 
    
    def __load_save_GOLD_DATA (self, action, fileaddress):
        if not action in ["load","save"]:
            self.__PRJ.PROGRAM_Halt ("action not understood: " + str(action)) 

        if action == "load":
            try:
                self.__PRJ.lp ("[INFO]: LOADING GOLD/DATA FROM FILE: " + fileaddress)
                with open(fileaddress, "rb") as f:
                    self.__GOLD , self.__DATA = pickle.load(f)
                
            except Exception as E:
                self.__PRJ.PROGRAM_Halt ("Could not load pickled GOLD and DATA from file.\nError:"+E.message)                
        
        elif action == "save":
            try:
                self.__PRJ.lp ("[INFO]: SAVING GOLD/DATA INTO FILE: " + fileaddress)
                with open(fileaddress, "wb") as f:
                    pickle.dump([self.__GOLD , self.__DATA], f)
            except Exception as E:
                self.__PRJ.PROGRAM_Halt ("Could not save pickled GOLD and DATA from file.\nError:"+E.message)                
            
    def __ResetTraining (self, useOnlyXTopPathsForFeatureGeneration):
        self.__RuntimeStatistics = {"SelectedArchitecture": None, "ExecutionRound":0 , "RandomSeed": None, "NoEpochs": 0 , "TopKP":str(useOnlyXTopPathsForFeatureGeneration)}; 
        if self.__model <> None:        
            del self.__model;
        self.__model = None; 
        self.__WhichFeaturesToUse = None 
        self.__WhichOutputsToPredit = None 
    
    def __ResetMatrices (self):
        self.__Matrices = {"TrainingSet":{}, "DevelopmentSet":{}, "TestSet": {}}
        
    def __Verify_RE_PARAMS (self):
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
            
            if (self.__RE_PARAMS.TestSetPredictionOutputFolder == None) or (not isinstance (self.__RE_PARAMS.TestSetPredictionOutputFolder, str)) or \
               (not os.path.isdir (self.__RE_PARAMS.TestSetPredictionOutputFolder)): 
                self.__PRJ.PROGRAM_Halt ("WriteBack for TestSet predictions is requested, but TestSetPredictionOutputFolder is not a valid folder address."); 
            else:
                self.__RE_PARAMS.TestSetPredictionOutputFolder = self.__RE_PARAMS.TestSetPredictionOutputFolder.replace ("\\" , "/"); 
                if self.__RE_PARAMS.TestSetPredictionOutputFolder[-1] <> "/":
                    self.__RE_PARAMS.TestSetPredictionOutputFolder+= "/"; 
                
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
                
    def __PreprocessAllFiles (self):
        self.__PRJ.lp (["-"*80, "Preprocessing all files for Training, Devel, and Test Sets ..." , "-"*80]) 
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
            
            allsentences = [] ; 
            
            for FileAddress in L:
                Sentences , Root , LOCAL_TotalCorpusRootAnalysisResult = self.__PRJ.Preprocessor.ProcessCorpus (FileAddress,DuplicationRemovalPolicy,SkipSentences); 
                allsentences.extend (Sentences); 
            
            self.__DATA[Set]["allsentences"] = allsentences
            
            self.__GOLD[Set] = {"positives":{}, "negatives":{}}
            
            for sentence in self.__DATA[Set]["allsentences"]:
                for pair in sentence["PAIRS"]:
                    if pair["POSITIVE"]:
                        self.__GOLD[Set]["positives"][pair["ID"]] = pair
                    else:
                        self.__GOLD[Set]["negatives"][pair["ID"]] = pair
        
        MSG = ["-"*30+"CORPUS STATISTICS"+"-"*30]
        for Set in ["TrainingSet","DevelopmentSet","TestSet"]:
            total_positives = len (self.__GOLD[Set]["positives"])
            total_negatives = len (self.__GOLD[Set]["negatives"])
            
            positive_types = {}
            for p_id in self.__GOLD[Set]["positives"]:
                tp = self.__GOLD[Set]["positives"][p_id]["CLASS_TP"]
                if not tp in positive_types:
                    positive_types[tp]=1
                else:
                    positive_types[tp]+=1
            
            MSG.append ("Set :" + Set + "\n" + "-"*20)
            MSG.append ("\t positives: " + str(total_positives))
            MSG.append ("\t negatives: " + str(total_negatives))
            MSG.append ("\t total    : " + str(total_positives+total_negatives))
            MSG.append ("\t ----------------")
            MSG.append ("\t positives:")
            for tp in positive_types:
                MSG.append ("\t\t- " + tp + ": " + str(positive_types[tp]))
        self.__PRJ.lp (MSG)

    def __FindTopKShortestPaths (self, key, FeatureGenerationParams):
        if not key in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("Invalid key. Key should be in " + str(self.__DATA.keys()))
        
        import datetime
        start = datetime.datetime.now()
        self.__DATA[key]["allsentences"] = self.__PRJ.FeatureGenerator.CalculateAndAddTopKSDPs_Parallel (self.__DATA[key]["allsentences"], FeatureGenerationParams,key)
        end = datetime.datetime.now()
        self.__PRJ.lp ("Finding Top-K Shortest Path Finished for " + key + ". Processing time: " + str(end - start).split(".")[0])

    def __GenerateFeatures (self, key, FeatureGenerationParams, AddFeatureToDictIfNotExists,useOnlyXTopPathsForFeatureGeneration=None):
        if not key in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("Invalid key. Key should be in " + str(self.__DATA.keys()))
        
        self.__PRJ.lp (["Running Feature Extraction for :" + key, "Allowed to add to Feature Mapping Dictionary:"+str(AddFeatureToDictIfNotExists)])
        self.__PRJ.FeatureGenerator.ExtractFeatures (self.__DATA[key]["allsentences"], FeatureGenerationParams, AddFeatureToDictIfNotExists,useOnlyXTopPathsForFeatureGeneration) 


    def __RestShortestPathsAndFeatures (self):
        self.__PRJ.FeatureMappingDictionary.reset_dictionary()
        self.__PRJ.lp ("Reseting all top-k-paths and all extracted features in sentences.")
        for key in self.__DATA:
            if self.__DATA[key].has_key("allsentences"):
                for sentence in self.__DATA[key]["allsentences"]:
                    for pair in sentence["PAIRS"]:
                        if pair.has_key("TOPKP"):
                            del pair["TOPKP"]
                        if pair.has_key("TOPKP_Features"):
                            del pair["TOPKP_Features"]
                        if pair.has_key("SP_Features"):
                            del pair["SP_Features"]
                        if pair.has_key("FS_Features"):
                            del pair["FS_Features"]
        
    def __ResetFeatures (self):
        self.__PRJ.FeatureMappingDictionary.reset_dictionary()
        self.__PRJ.POSTagsEmbeddings.reset_dictionary()
        self.__PRJ.DPTypesEmbeddings.reset_dictionary()
        self.__PRJ.lp ("Reseting all extracted features in sentences.")
        for key in self.__DATA:
            if self.__DATA[key].has_key("allsentences"):
                for sentence in self.__DATA[key]["allsentences"]:
                    for pair in sentence["PAIRS"]:
                        if pair.has_key("TOPKP_Features"):
                            del pair["TOPKP_Features"]
                        if pair.has_key("SP_Features"):
                            del pair["SP_Features"]
                        if pair.has_key("FS_Features"):
                            del pair["FS_Features"]
                            
                            
    def __GetSPMaxLength (self, useOnlyXTopPathsForFeatureGeneration, sets=["TrainingSet","DevelopmentSet"]):
        MaxSPLength = 0 
        for key in sets:
            for sentence in self.__DATA[key]["allsentences"]:
                for pair in sentence["PAIRS"]:
                    if (pair.has_key("TOPKP")) and (pair.has_key("TOPKP_Features")) and (pair.has_key("SP_Features")):
                        if (pair["TOPKP"] <> None) and (pair["TOPKP_Features"] <> None) and (pair["SP_Features"] <> None):#an example might have pair["TOPKP_Features"] = [] in a very rare and very very unlikely condition
                            #each path in pair["SP_Features"] is a tuple like (path_weight,normalized_path_weight,words,postags,dptypes)
                            #hence, path[2] returns the words 
                            if (useOnlyXTopPathsForFeatureGeneration == None) or (useOnlyXTopPathsForFeatureGeneration >= len(pair["SP_Features"])):
                                tempMaxSPLength = max([len(path[2]) for path in pair["SP_Features"]])
                                if tempMaxSPLength > MaxSPLength:
                                    MaxSPLength = tempMaxSPLength
                            
                            else:
                                tempSelectedPaths = pair["SP_Features"][0:useOnlyXTopPathsForFeatureGeneration] #for example, select 10 paths out of 30 existing paths
                                tempMaxSPLength = max([len(path[2]) for path in tempSelectedPaths])
                                if tempMaxSPLength > MaxSPLength:
                                    MaxSPLength = tempMaxSPLength
        return MaxSPLength                             
                
    def __GetSPMaxCount (self, sets=["TrainingSet","DevelopmentSet"]):
        MaxSPCount = 0 
        for key in sets:
            for sentence in self.__DATA[key]["allsentences"]:
                for pair in sentence["PAIRS"]:
                    if (pair.has_key("TOPKP")) and (pair.has_key("TOPKP_Features")) and (pair.has_key("SP_Features")):
                        if (pair["TOPKP"] <> None) and (pair["TOPKP_Features"] <> None) and (pair["SP_Features"] <> None):#an example might have pair["TOPKP_Features"] = [] in a very rare and very very unlikely condition
                            tempMaxSPCount = len(pair["SP_Features"])
                            if tempMaxSPCount > MaxSPCount:
                                MaxSPCount = tempMaxSPCount
        return MaxSPCount                             

    def __GetFSMaxLength (self, useOnlyXTopPathsForFeatureGeneration, sets=["TrainingSet","DevelopmentSet"]):
        MaxFSLength = 0 
        for key in sets:
            for sentence in self.__DATA[key]["allsentences"]:
                for pair in sentence["PAIRS"]:
                    if (pair.has_key("TOPKP")) and (pair.has_key("TOPKP_Features")) and (pair.has_key("SP_Features")) and (pair.has_key("FS_Features")):
                        if (pair["TOPKP"] <> None) and (pair["TOPKP_Features"] <> None) and (pair["SP_Features"] <> None) and (pair["FS_Features"] <> None):#an example might have pair["TOPKP_Features"] = [] in a very rare and very very unlikely condition
                            tempMaxFSLength = len(pair["FS_Features"][0])
                            if tempMaxFSLength > MaxFSLength:
                                MaxFSLength = tempMaxFSLength
                            
        return MaxFSLength                             

    def __GenerateANNMatrices (self,key, FeatureGenerationParams, MaxSPCount, MaxSPLength, MaxFSLength):
        #We will have MaxSPCount lstms/cnns, and input_length for each will be MaxSPLength
        if not key in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("Invalid key. Key should be in " + str(self.__DATA.keys()))

        import SharedFunctions as SF
        
        all_sentences = self.__DATA[key]["allsentences"] 
        pos_neg_dict, class_tp_dict, total_example_count = SF.CalculateHowManyRelationsWithShortestPathInDataset (all_sentences)
        
        if total_example_count < 1:
            self.__PRJ.PROGRAM_Halt (key + " has no examples!")
            
        SENTENCE_INDEX = np.zeros (total_example_count, dtype=np.int32) #for pairs .. so, if sentence one has 3 examples and sentence two has four --> 1112222 
        PAIR_TRACKING  = [] 

        HowManyColumnsForOneHotEncoding = len (self.__PRJ.Configs["OneHotEncodingForMultiClass"]); 
        Y = np.zeros ((total_example_count, HowManyColumnsForOneHotEncoding),dtype=np.int16); 

        seen_pair_count = 0 
        feature_weights = np.zeros((total_example_count,MaxSPCount), dtype=np.float32) #normalized_path_weight for each lstm/cnn 
        feature_words   = [np.zeros((total_example_count,MaxSPLength), dtype=np.int32) for i in range(MaxSPCount)]
        feature_postags = [np.zeros((total_example_count,MaxSPLength), dtype=np.int16) for i in range(MaxSPCount)]
        feature_dptypes = [np.zeros((total_example_count,MaxSPLength), dtype=np.int16) for i in range(MaxSPCount)]
        
        #full-sentence-features
        fs_feature_words    = np.zeros((total_example_count,MaxFSLength), dtype=np.int32)
        fs_feature_postags  = np.zeros((total_example_count,MaxFSLength), dtype=np.int16)
        fs_feature_chunks   = np.zeros((total_example_count,MaxFSLength), dtype=np.int8)
        fs_feature_p1_dists = np.zeros((total_example_count,MaxFSLength), dtype=np.int8)
        fs_feature_p2_dists = np.zeros((total_example_count,MaxFSLength), dtype=np.int8)
        
        if len(self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"]) > 1: #there are at least two entity types, so it worths to make clear which is which for each example 
            feature_entity_type = np.zeros ((total_example_count,2*len(self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])), dtype=np.int8) 
        else:
            feature_entity_type = None 
            
        for Sentence_Index , S in enumerate (all_sentences):
            for pair in S["PAIRS"]:
                if (not pair.has_key("TOPKP")) or (pair["TOPKP"] == None):
                    continue 
                if (not pair.has_key("SP_Features")) or (pair["SP_Features"] == None): #an example might have pair["TOPKP_Features"] = [] in a very rare and very very unlikely condition
                    continue 
                if (not pair.has_key("FS_Features")) or (pair["FS_Features"] == None): #an example might have pair["TOPKP_Features"] = [] in a very rare and very very unlikely condition
                    continue 
                
                #1-Track: for later writing-back the prediction results into XML
                e1_type = S["ENTITIES"][pair["E1"]]["TYPE"]; 
                e2_type = S["ENTITIES"][pair["E2"]]["TYPE"]; 
                _e1tp = e1_type.capitalize()
                _e2tp = e2_type.capitalize()
                PAIR_TRACKING.append ( (S["ID"] , pair["ID"] , pair["E1"] , pair["E2"], _e1tp , _e2tp) )
                    
                #2-entity type and their order features ...
                if len(self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"]) > 1: #there are at least two entity types
                    #What Policy has been used to generate SDP.
                    SDP_DIRECTION = FeatureGenerationParams.SDPDirection
                        
                    if SDP_DIRECTION == FeGen.TSDP_TSVM_SDPFeatureGenerationDirection.from_1stOccurring_to_2nd:
                        #first column_index, always type of first occuring entity
                        e1_bgn = S["ENTITIES"][pair["E1"]]["HEADOFFSET"][0]
                        e2_bgn = S["ENTITIES"][pair["E2"]]["HEADOFFSET"][0]

                        if e1_bgn == e2_bgn:
                            e1_bgn = S["ENTITIES"][pair["E1"]]["ORIGOFFSETS"][0][0]
                            e2_bgn = S["ENTITIES"][pair["E2"]]["ORIGOFFSETS"][0][0]
                            
                        if e1_bgn < e2_bgn: #CRITICAL
                            e1_tp = S["ENTITIES"][pair["E1"]]["TYPE"].lower()
                            e2_tp = S["ENTITIES"][pair["E2"]]["TYPE"].lower()
                        else:
                            e1_tp = S["ENTITIES"][pair["E2"]]["TYPE"].lower()
                            e2_tp = S["ENTITIES"][pair["E1"]]["TYPE"].lower()
                            
                    elif SDP_DIRECTION == FeGen.TSDP_TSVM_SDPFeatureGenerationDirection.from_2ndOccurring_to_1st:
                        #first column_index, always type of second occuring entity
                        e1_bgn = S["ENTITIES"][pair["E1"]]["HEADOFFSET"][0]
                        e2_bgn = S["ENTITIES"][pair["E2"]]["HEADOFFSET"][0]

                        if e1_bgn == e2_bgn:
                            e1_bgn = S["ENTITIES"][pair["E1"]]["ORIGOFFSETS"][0][0]
                            e2_bgn = S["ENTITIES"][pair["E2"]]["ORIGOFFSETS"][0][0]
                            
                        if e1_bgn > e2_bgn: #CRITICAL
                            e1_tp = S["ENTITIES"][pair["E1"]]["TYPE"].lower()
                            e2_tp = S["ENTITIES"][pair["E2"]]["TYPE"].lower()
                        else:
                            e1_tp = S["ENTITIES"][pair["E2"]]["TYPE"].lower()
                            e2_tp = S["ENTITIES"][pair["E1"]]["TYPE"].lower()
                            
                        
                    elif SDP_DIRECTION == FeGen.TSDP_TSVM_SDPFeatureGenerationDirection.from_e1value_to_e2value:
                        e1_tp = S["ENTITIES"][pair["E1"]]["TYPE"].lower()
                        e2_tp = S["ENTITIES"][pair["E2"]]["TYPE"].lower()

                    elif SDP_DIRECTION == FeGen.TSDP_TSVM_SDPFeatureGenerationDirection.from_e2value_to_e1value:
                        e1_tp = S["ENTITIES"][pair["E2"]]["TYPE"].lower()
                        e2_tp = S["ENTITIES"][pair["E1"]]["TYPE"].lower()
                    
                    else:
                        self.__PRJ.PROGRAM_Halt ("SDP_DIRECTION METHOD : " + SDP_DIRECTION + " IS NOT IMPLEMENTED YET!");
                    
                    ETP1_COLUMN_IDX = self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"][e1_tp] 
                    ETP2_COLUMN_IDX = len(self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"]) + self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"][e2_tp] 
                    feature_entity_type[seen_pair_count, ETP1_COLUMN_IDX] = 1 ;
                    feature_entity_type[seen_pair_count, ETP2_COLUMN_IDX] = 1 ; 
                            
                #3-weights, words, postags, dptypes in TOP-K-SDPs 
                if len(pair["SP_Features"]) <= MaxSPCount:
                    selectedPaths = pair["SP_Features"]
                else:
                    selectedPaths = pair["SP_Features"][0:MaxSPCount]
                
                for path_idx, path_info in enumerate(selectedPaths):
                    path_weight,normalized_path_weight,path_words,path_postags,path_dptypes = path_info
                    feature_weights[seen_pair_count,path_idx] = normalized_path_weight
                    feature_words  [path_idx][seen_pair_count,0:len(path_words)]   = path_words
                    feature_postags[path_idx][seen_pair_count,0:len(path_postags)] = path_postags
                    feature_dptypes[path_idx][seen_pair_count,0:len(path_dptypes)] = path_dptypes
                
                #4-Full sentence coding
                _fs_words , _fs_postags , _fs_chunks , _fs_p1 , _fs_p2 = pair["FS_Features"]
                fs_feature_words    [seen_pair_count,0:len(_fs_words)]   = _fs_words
                fs_feature_postags  [seen_pair_count,0:len(_fs_postags)] = _fs_postags
                fs_feature_chunks   [seen_pair_count,0:len(_fs_chunks)]  = _fs_chunks
                fs_feature_p1_dists [seen_pair_count,0:len(_fs_p1)]      = _fs_p1
                fs_feature_p2_dists [seen_pair_count,0:len(_fs_p2)]      = _fs_p2
                
                #4-Label:
                if pair["POSITIVE"]==False:
                    Y [seen_pair_count,0]=1;#index zero is always for negative class(es)
                else:
                    class_label = pair["CLASS_TP"] 
                    OneHotIndex = self.__PRJ.Configs["OneHotEncodingForMultiClass"][class_label]
                    Y[seen_pair_count, OneHotIndex] = 1 
                            
                #5-Increament index ...
                SENTENCE_INDEX[seen_pair_count] = Sentence_Index;
                seen_pair_count+=1 ; 
        
        #<<<CRITICAL>>> giving shape is critical, because devel/test might have lower features ... 
        self.__Matrices[key]["weights"]     = feature_weights
        self.__Matrices[key]["words"]       = feature_words
        self.__Matrices[key]["postags"]     = feature_postags
        self.__Matrices[key]["dptypes"]     = feature_dptypes
        self.__Matrices[key]["entitytypes"] = feature_entity_type
        self.__Matrices[key]["fs_words"]    = fs_feature_words
        self.__Matrices[key]["fs_postags"]  = fs_feature_postags
        self.__Matrices[key]["fs_chunks"]   = fs_feature_chunks
        self.__Matrices[key]["fs_p1_dists"] = fs_feature_p1_dists
        self.__Matrices[key]["fs_p2_dists"] = fs_feature_p2_dists
        self.__Matrices[key]["Y"] = Y
        self.__Matrices[key]["SENTENCE_INDEX"] = SENTENCE_INDEX
        self.__Matrices[key]["PAIR_TRACKING"]  = PAIR_TRACKING
    
    def __BuildANNModel (self, ARCHITECTURE, _MaxSPCount, _MaxSPLength, _MaxFSLength, useOnlyXTopPathsForFeatureGeneration, RandomSeed):
        MSG = ["-"*30+" Building ANN "+"-"*30]        
        MSG+= ["Architecture   : " + ARCHITECTURE]
        MSG+= ["Optimizer      : " + str(self.__RE_PARAMS.KERAS_optimizer)]
        MSG+= ["Optimizer lr   : " + str(self.__RE_PARAMS.KERAS_optimizer_lr)]
        MSG+= ["TopKPaths      : " + str(useOnlyXTopPathsForFeatureGeneration)]
        MSG+= ["Max SP Count   : " + str(_MaxSPCount)]
        MSG+= ["Max SP Length  : " + str(_MaxSPLength)]
        MSG+= ["Max FS Length  : " + str(_MaxFSLength)]
        MSG+= ["-"*80]
        self.__PRJ.lp (MSG)
                                                           
        self.__ResetTraining(useOnlyXTopPathsForFeatureGeneration)
        from Architectures import ANN_Architecture_Builder
        TempArchBuilderObject = ANN_Architecture_Builder(_MaxSPCount,_MaxSPLength, _MaxFSLength, self.__PRJ, RandomSeed)

        self.__RuntimeStatistics["SelectedArchitecture"] = ARCHITECTURE 
        self.__RuntimeStatistics["RandomSeed"]           = str(TempArchBuilderObject.RandomSeed) 

        if ARCHITECTURE[-1] <> ")":
            self.__model , self.__WhichFeaturesToUse , self.__WhichOutputsToPredit = eval ("TempArchBuilderObject." + ARCHITECTURE + "()")
        else:
            self.__model , self.__WhichFeaturesToUse , self.__WhichOutputsToPredit = eval ("TempArchBuilderObject." + ARCHITECTURE )
        
        from keras.optimizers import Adam, Nadam, SGD 
        if self.__RE_PARAMS.KERAS_optimizer.lower() == "adam":
            myOptimizer = Adam(lr=self.__RE_PARAMS.KERAS_optimizer_lr)
        elif self.__RE_PARAMS.KERAS_optimizer.lower() == "nadam":
            myOptimizer = Nadam(lr=self.__RE_PARAMS.KERAS_optimizer_lr)
        elif self.__RE_PARAMS.KERAS_optimizer.lower() == "sgd":
            myOptimizer = SGD(lr=self.__RE_PARAMS.KERAS_optimizer_lr)
            
        self.__model.compile (loss="categorical_crossentropy", optimizer=myOptimizer) ; 
        self.__model.summary()

    def __AssingANNInputOutput (self, SetName,ReturnOutput=False): 
        if not SetName in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("Invalid SetName for setting Neural Network Inputs:" + SetName)

        if self.__WhichFeaturesToUse == None:
            self.__PRJ.PROGRAM_Halt ("self.__WhichFeaturesToUse Is None!")
        
        if self.__WhichOutputsToPredit == None:
            self.__PRJ.PROGRAM_Halt ("self.__WhichOutputsToPredit Us None!")
        
        if self.__Matrices[SetName] == {}:
            self.__PRJ.PROGRAM_Halt (SetName + " data is not ready!")
        
        ANN_INPUT  = []
        ANN_OUTPUT = [] if ReturnOutput else None 
        
        #<<<CRITICAL>>> the order is so important! Also, in the Architetrure.py functions, order should be the same as this!         
        if self.__WhichFeaturesToUse["entitytypes"] == True:
            ANN_INPUT.append(self.__Matrices[SetName]["entitytypes"])
        
        if self.__WhichFeaturesToUse["weights"]     == True: 
            ANN_INPUT.append(self.__Matrices[SetName]["weights"])

        if self.__WhichFeaturesToUse["words"]       == True: 
            ANN_INPUT.extend(self.__Matrices[SetName]["words"]) #<<<CRITICAL>>> use extend because it is a list

        if self.__WhichFeaturesToUse["postags"]     == True:  
            ANN_INPUT.extend(self.__Matrices[SetName]["postags"])  #<<<CRITICAL>>> use extend because it is a list

        if self.__WhichFeaturesToUse["dptypes"]     == True: 
            ANN_INPUT.extend(self.__Matrices[SetName]["dptypes"])  #<<<CRITICAL>>> use extend because it is a list

        if self.__WhichFeaturesToUse["fs_words"]    == True: 
            ANN_INPUT.append(self.__Matrices[SetName]["fs_words"])
            
        if self.__WhichFeaturesToUse["fs_postags"]  == True: 
            ANN_INPUT.append(self.__Matrices[SetName]["fs_postags"])

        if self.__WhichFeaturesToUse["fs_chunks"]   == True: 
            ANN_INPUT.append(self.__Matrices[SetName]["fs_chunks"])

        if self.__WhichFeaturesToUse["fs_p1_dists"] == True: 
            ANN_INPUT.append(self.__Matrices[SetName]["fs_p1_dists"])

        if self.__WhichFeaturesToUse["fs_p2_dists"] == True: 
            ANN_INPUT.append(self.__Matrices[SetName]["fs_p2_dists"])

        #<<<CRITICAL>>> the order is so important! Also, in the Architetrure.py functions, order should be the same as this!         
        if ReturnOutput:
            if self.__WhichOutputsToPredit["Y"] == True:
                ANN_OUTPUT.append(self.__Matrices[SetName]["Y"])
        
        if ANN_INPUT == []:
            self.__PRJ.PROGRAM_Halt ("Cannot assign any network input: No selected input features for " + SetName)
        
        if ANN_OUTPUT == []:
            self.__PRJ.PROGRAM_Halt ("Cannot assign any network output: No selected output features for " + SetName)
            
        return ANN_INPUT , ANN_OUTPUT
    
    def __Train (self, SetName):
        if self.__model == None:
            self.__PRJ.PROGRAM_Halt ("Model should have been compiled before calling training.")
        
        if isinstance (SetName, basestring):
            if not SetName in self.__DATA.keys():
                self.__PRJ.PROGRAM_Halt ("Undefined SetName for training the network: " + SetName)
                
            ANN_INPUT , ANN_OUTPUT = self.__AssingANNInputOutput(SetName, ReturnOutput=True) 
        
        elif isinstance (SetName, list):
            ALL_INPUTS , ALL_OUTPUTS = OrderedDict() , OrderedDict()
            FIN_INPUTS , FIN_OUTPUTS = [] , []
            
            for s in SetName:
                if not s in self.__DATA.keys():
                    self.__PRJ.PROGRAM_Halt ("Undefined SetName for training the network: " + s)
            
            for s in SetName:
                ANN_INPUT , ANN_OUTPUT = self.__AssingANNInputOutput(s, ReturnOutput=True) 
                ALL_INPUTS[s]  = ANN_INPUT
                ALL_OUTPUTS[s] = ANN_OUTPUT
            
            HowManyMatrices = range(len(ALL_INPUTS[s]))
            for i in HowManyMatrices:
                FIN_INPUTS.append(np.vstack ([ALL_INPUTS[key][i] for key in ALL_INPUTS]))
            
            FIN_OUTPUTS.append(np.vstack ([ALL_OUTPUTS[key][0] for key in ALL_OUTPUTS]))
            
            ANN_INPUT  = FIN_INPUTS
            ANN_OUTPUT = FIN_OUTPUTS
            
        else:
            self.__PRJ.PROGRAM_Halt ("Unknown SetName for training: " + str(SetName))
        
        self.__RuntimeStatistics["NoEpochs"]+=1 
        H = self.__model.fit (ANN_INPUT,
                              ANN_OUTPUT, 
                              epochs       = 1, 
                              batch_size   = self.__RE_PARAMS.KERAS_minibatch_size, 
                              class_weight = self.__RE_PARAMS.KERAS_class_weight, 
                              shuffle      = self.__RE_PARAMS.KERAS_shuffle_training_examples, 
                              verbose      = self.__RE_PARAMS.KERAS_fit_verbose) 

        MSG = ["Training:","-"*30]    
        MSG+= ["\tTopKP            : " + str(self.__RuntimeStatistics["TopKP"])]
        MSG+= ["\tArchitecture     : " + str(self.__RuntimeStatistics["SelectedArchitecture"])]
        MSG+= ["\tRandom Seed      : " + str(self.__RuntimeStatistics["RandomSeed"])]
        MSG+= ["\t---------------------------------------------------------------"]
        MSG+= ["\tTraining Set     : " + str(SetName)] 
        MSG+= ["\tTraining Inputs  : " + str([x.shape for x in ANN_INPUT])]
        MSG+= ["\tTraining Outputs : " + str([x.shape for x in ANN_OUTPUT])]
        MSG+= ["\tBatch Size       : " + str(self.__RE_PARAMS.KERAS_minibatch_size)]
        MSG+= ["\tClass Weights    : " + str(self.__RE_PARAMS.KERAS_class_weight)]
        MSG+= ["\tShuffle examples : " + str(self.__RE_PARAMS.KERAS_shuffle_training_examples)]
        MSG+= ["\t---------------------------------------------------------------"]
        MSG+= ["\tRound            : " + str(self.__RuntimeStatistics["ExecutionRound"])]
        MSG+= ["\tEpoch            : " + str(self.__RuntimeStatistics["NoEpochs"])]
        MSG+= ["\tTraining Loss    : " + str(H.history)]        
        MSG+= ["\t---------------------------------------------------------------"]
        self.__PRJ.lp (MSG)
        
    def __PredictEvaluateWriteback (self, SetName):
        ALL_POSSIBLE_OUTPUT_NAMES = ["Y"] #<<<CRITICAL>>> Order is very important !!!
        if self.__model == None:
            self.__PRJ.PROGRAM_Halt ("Model should have been compiled before calling training.")

        if SetName == "DevelopmentSet":
            param_ProcessAfterXEpoch = self.__RE_PARAMS.ProcessDevelSetAfterEpochNo
            param_ShouldPredict      = self.__RE_PARAMS.PredictDevelSet
            param_ShouldEvaluate     = self.__RE_PARAMS.EvaluateDevelSet
            param_SholdWriteBack     = self.__RE_PARAMS.WriteBackDevelSetPredictions
            param_WriteBackFolder    = self.__RE_PARAMS.DevelSetPredictionOutputFolder
            if len(self.__RE_PARAMS.DevelopmentSet_Files_List)>0:
                param_GoldXMLFileAddress = self.__RE_PARAMS.DevelopmentSet_Files_List[0] #<<<CRITICAL>>> We only support Writing Back for first given devel-set file

        elif SetName == "TestSet":
            param_ProcessAfterXEpoch = self.__RE_PARAMS.ProcessTestSetAfterEpochNo
            param_ShouldPredict      = self.__RE_PARAMS.PredictTestSet
            param_ShouldEvaluate     = self.__RE_PARAMS.EvaluateTestSet
            param_SholdWriteBack     = self.__RE_PARAMS.WriteBackTestSetPredictions
            param_WriteBackFolder    = self.__RE_PARAMS.TestSetPredictionOutputFolder
            if len(self.__RE_PARAMS.TestSet_Files_Lists)>0:
                param_GoldXMLFileAddress = self.__RE_PARAMS.TestSet_Files_Lists[0] #<<<CRITICAL>>> We only support Writing Back for first given test-set file
        
        if (param_ProcessAfterXEpoch < 0) or (self.__RuntimeStatistics["NoEpochs"] < param_ProcessAfterXEpoch) or (not param_ShouldPredict):
            return 


        param_PairTracking = self.__Matrices[SetName]["PAIR_TRACKING"] 
        
        #1) Prediction:
        ANN_INPUT , ANN_OUTPUT = self.__AssingANNInputOutput(SetName,ReturnOutput=True)
        R = self.__model.predict (ANN_INPUT)
        if not isinstance(R, list):
            R = [R]  

        ANN_OUTPUT_Shapes = str([x.shape for x in R])
        
        y_true_labels , y_pred_labels , y_pred_confds , eval_results = OrderedDict() , {} , {} , {}
        eval_metric = None
        
        for output_name in ALL_POSSIBLE_OUTPUT_NAMES:
            if self.__WhichOutputsToPredit[output_name]==True:
                y_pred_confds[output_name] = R[0] 
                y_pred_labels[output_name] = np.argmax (y_pred_confds[output_name], axis = -1)  
                y_true_labels[output_name] = np.argmax (ANN_OUTPUT[0], axis = -1)
                R = R[1:] #Go to next
                ANN_OUTPUT = ANN_OUTPUT[1:] #Go to next
                
        #2) Evaluation:
        EVAL_MSG = ["="*80, "EVALUATION RESULTS : " + SetName + "\tPredicted Array(s) Shape:" + ANN_OUTPUT_Shapes]
        if param_ShouldEvaluate:
            for output_name in y_true_labels.keys():
                y_true = y_true_labels[output_name]
                y_pred = y_pred_labels[output_name]
                if output_name == "Y":                
                    eval_results[output_name] = self.__lowelevel_Evaluate_WithGold (y_true, y_pred, param_PairTracking, SetName)
                else:
                    eval_results[output_name] = self.__lowelevel_Evaluate (y_true, y_pred)
                
                EVAL_MSG+= ["\nOUTPUT: " + output_name , "-"*30]
                for key in eval_results[output_name]:
                    s = "\t"+GF.NVLR(key,65)+": " 
                    v = eval_results[output_name][key]
                    s+= GF.FDecimalPoints(v, self.__DecimalPoints) if isinstance(v,float) else str(v)
                    EVAL_MSG+=[s]
            s = "-"*50+"\n"
            s+= "Optimization Metric Value ("+ self.__RE_PARAMS.Classification_OptimizationOutput + " " + self.__RE_PARAMS.Classification_OptimizationMetric + ") : " 
            s+= GF.FDecimalPoints(eval_results[self.__RE_PARAMS.Classification_OptimizationOutput][self.__RE_PARAMS.Classification_OptimizationMetric], self.__DecimalPoints)
            EVAL_MSG+= [s]
            eval_metric = eval_results[self.__RE_PARAMS.Classification_OptimizationOutput][self.__RE_PARAMS.Classification_OptimizationMetric]
                 
        #3) Writing Back Prediction Results or if external evaluation needed, writeback eventhough not explicitly requested.
        if (param_SholdWriteBack) or (self.__RE_PARAMS.Classification_ExternalEvaluator <> None):
            
            #This is used when external evaluation requested, but NOT writeback. 
            if param_SholdWriteBack:
                PredictionOutputFolder = param_WriteBackFolder + self.__RuntimeStatistics["SelectedArchitecture"]+"_"+ self.__RuntimeStatistics["TopKP"]+"/" 
            else:
                PredictionOutputFolder = "" 
            Round                          = self.__RuntimeStatistics["ExecutionRound"] 
            RandomSeed                     = self.__RuntimeStatistics["RandomSeed"] 
            EpochNo                        = self.__RuntimeStatistics["NoEpochs"] 
            WriteBackOutputFileAddressName = PredictionOutputFolder + "Pred_Round" + str(Round) + "_Seed" + str(RandomSeed) + "_EpochNo" + str(EpochNo) + ".xml" 
            self.__lowelevel_WriteBackSetPredictionResults (y_pred_labels["Y"], y_pred_confds["Y"], param_PairTracking, param_GoldXMLFileAddress, WriteBackOutputFileAddressName)
        
        #4) External Evaluation 
        if self.__RE_PARAMS.Classification_ExternalEvaluator <> None:
            if SetName == "DevelopmentSet":
                GoldFiles = self.__RE_PARAMS.DevelopmentSet_Files_List
            elif SetName == "TestSet": 
                GoldFiles = self.__RE_PARAMS.TestSet_Files_Lists
            else:
                self.__PRJ.PROGRAM_Halt ("Cannot call external evaluation function. Set should be either DevelopmentSet or TestSet")                
                    
            input_params = {
                "lp"            : self.__PRJ.lp,  
                "PROGRAM_Halt"  : self.__PRJ.PROGRAM_Halt, 
                "Set"           : SetName,
                "PredictedFile" : WriteBackOutputFileAddressName,
                "GoldFilesList" : GoldFiles, 
            }
            
            External_Evaluation_Result = self.__RE_PARAMS.Classification_ExternalEvaluator(input_params)
            EVAL_MSG+= ["-"*50,"External Evaluation Result:" + str(External_Evaluation_Result)]
            eval_metric = External_Evaluation_Result 
            
            if not param_SholdWriteBack: #then delete the created file ...
                os.remove (WriteBackOutputFileAddressName)
                
        if param_ShouldEvaluate:
            EVAL_MSG+= ["="*80]
            self.__PRJ.lp (EVAL_MSG)
                
        return eval_metric
    
    def RunPreProcessing(self, DataPreparationSteps, FeatureGenerationParams):
        self.__PRJ.lp (["*"*80,"Preprocessing:","-"*17,"DataPreparationSteps:"+str(DataPreparationSteps),"MachineLearningTasks:"+str("MachineLearningTasks"),"*"*80]);

        if "load_gold_and_data" in DataPreparationSteps and DataPreparationSteps["load_gold_and_data"] <> None:
            self.__load_save_GOLD_DATA ("load", fileaddress=DataPreparationSteps["load_gold_and_data"])
            
        if "preprocess" in DataPreparationSteps and DataPreparationSteps["preprocess"]==True:
            self.__PreprocessAllFiles(); 
            if self.__RE_PARAMS.WriteBackProcessedTrainingSet:
                self.__WriteBackProcessedGOLD ("TrainingSet") 
    
            if self.__RE_PARAMS.WriteBackProcessedDevelopmentSet:
                self.__WriteBackProcessedGOLD ("DevelopmentSet") 
            
            if self.__RE_PARAMS.WriteBackProcessedTestSet:
                self.__WriteBackProcessedGOLD ("TestSet") 
        
        #self.__DATA["TrainingSet"]["allsentences"]    = self.__DATA["TrainingSet"]["allsentences"]
        #self.__DATA["DevelopmentSet"]["allsentences"] = self.__DATA["DevelopmentSet"]["allsentences"]
        #self.__DATA["TestSet"]["allsentences"]        = self.__DATA["TestSet"]["allsentences"]
        
        if "findshortestpaths" in DataPreparationSteps and DataPreparationSteps["findshortestpaths"]==True:
            self.__RestShortestPathsAndFeatures ()
            for key in self.__DATA:
                self.__FindTopKShortestPaths (key, FeatureGenerationParams)

        if "save_gold_and_data" in DataPreparationSteps and DataPreparationSteps["save_gold_and_data"] <> None:
            self.__load_save_GOLD_DATA ("save", fileaddress=DataPreparationSteps["save_gold_and_data"])

        MSG = ["-"*30+"CORPUS STATISTICS"+"-"*30]
        for Set in ["TrainingSet","DevelopmentSet","TestSet"]:
            total_positives = len (self.__GOLD[Set]["positives"])
            total_negatives = len (self.__GOLD[Set]["negatives"])
            
            positive_types = {}
            for p_id in self.__GOLD[Set]["positives"]:
                tp = self.__GOLD[Set]["positives"][p_id]["CLASS_TP"]
                if not tp in positive_types:
                    positive_types[tp]=1
                else:
                    positive_types[tp]+=1
            
            MSG.append ("Set :" + Set + "\n" + "-"*20)
            MSG.append ("\t positives: " + str(total_positives))
            MSG.append ("\t negatives: " + str(total_negatives))
            MSG.append ("\t total    : " + str(total_positives+total_negatives))
            MSG.append ("\t ----------------")
            MSG.append ("\t positives:")
            for tp in positive_types:
                MSG.append ("\t\t- " + tp + ": " + str(positive_types[tp]))

        MSG+= ["","",""]
        MSG+= ["MACHINE LEARNING EXAMPLES STATISTICS:", "-"*130]
        MSG.append (" "*20+SF.NVLL("Positives",30)+SF.NVLL("Negatives",30)+SF.NVLL("Total",30))
        for key in self.__DATA:
            all_sentences = self.__DATA[key]["allsentences"] 
            pos_neg_dict, class_tp_dict, total_example_count = SF.CalculateHowManyRelationsWithTOPKPaths (all_sentences)
            MSG.append ( SF.NVLR (key+" GOLD", 20) + SF.NVLL (str(len(self.__GOLD[key]["positives"])),30) + SF.NVLL (str(len(self.__GOLD[key]["negatives"])),30) + SF.NVLL (str(len(self.__GOLD[key]["positives"])+len(self.__GOLD[key]["negatives"])),30) )
            MSG.append ( SF.NVLR (key+" MLEx", 20) + SF.NVLL (str(pos_neg_dict["Positives"]),30) + SF.NVLL (str(pos_neg_dict["Negatives"]),30) +  SF.NVLL (str(total_example_count),30))
            MSG.append ("-"*130)
        self.__PRJ.lp (MSG)
        
    def RunSimplePipeline(self,DataPreparationSteps, FeatureGenerationParams, Archictectures_MaxEpoch, useOnlyXTopPathsForFeatureGeneration=None, RandomSeed=None):
        #train on train, optimize on devel-set, get best C_value
        self.__PRJ.lp (["-"*80,"FEATURE GENERATION PARAMS:" , "-"*30 , str(FeatureGenerationParams) , "-"*80])

        self.RunPreProcessing(DataPreparationSteps, FeatureGenerationParams)

        #<<<CRITICAL>>> Train can add to feature dictionary, devel/test cannot
        self.__ResetFeatures()
        self.__GenerateFeatures ("TrainingSet", FeatureGenerationParams, True, useOnlyXTopPathsForFeatureGeneration)

        if self.__RE_PARAMS.PredictDevelSet:
            self.__GenerateFeatures ("DevelopmentSet", FeatureGenerationParams, False , useOnlyXTopPathsForFeatureGeneration)

        if self.__RE_PARAMS.PredictTestSet:
            self.__GenerateFeatures ("TestSet", FeatureGenerationParams, False , useOnlyXTopPathsForFeatureGeneration)

        self.__PRJ.lp ("Total Extracted Features:" + str(self.__PRJ.FeatureMappingDictionary.return_dictionary_length()))
        
        selected_sets = ["TrainingSet"]
        if self.__RE_PARAMS.PredictDevelSet:
            selected_sets.append ("DevelopmentSet")
        
        if self.__RE_PARAMS.PredictTestSet:
             selected_sets.append ("TestSet")
             
        if useOnlyXTopPathsForFeatureGeneration == None:
            _MaxSPCount = self.__GetSPMaxCount (sets=selected_sets)
        else:
            _MaxSPCount = useOnlyXTopPathsForFeatureGeneration

        _MaxSPLength = self.__GetSPMaxLength (useOnlyXTopPathsForFeatureGeneration, sets=selected_sets)
        _MaxFSLength = self.__GetFSMaxLength (useOnlyXTopPathsForFeatureGeneration, sets=selected_sets)
        
        self.__ResetMatrices () 
        self.__GenerateANNMatrices("TrainingSet", FeatureGenerationParams, _MaxSPCount, _MaxSPLength, _MaxFSLength) 
        if self.__RE_PARAMS.PredictDevelSet:
            self.__GenerateANNMatrices("DevelopmentSet", FeatureGenerationParams, _MaxSPCount, _MaxSPLength, _MaxFSLength) 
        
        if self.__RE_PARAMS.PredictTestSet:
            self.__GenerateANNMatrices("TestSet", FeatureGenerationParams, _MaxSPCount, _MaxSPLength, _MaxFSLength) 
        
        EvalDevel_AllResults , EvalDevel_ArchsBestResult = [] , []
        EvalTest_AllResults  , EvalTest_ArchsBestResult  = [] , []
        
        for ARCHITECTURE,MAX_EPOCH in Archictectures_MaxEpoch:
            EvalDevel_ThisArchResults , EvalTest_ThisArchResults = [] , [] 
            
            #create DEVELOPMENT-SET predictin output SUB-folder (Architecture based)
            if (self.__RE_PARAMS.WriteBackDevelSetPredictions == True):
                    DevelSetPredictionOutputFolder = self.__RE_PARAMS.DevelSetPredictionOutputFolder + ARCHITECTURE + "_" + str(useOnlyXTopPathsForFeatureGeneration) + "/"
                    if not os.path.exists(DevelSetPredictionOutputFolder):
                        try:
                            self.__PRJ.lp (["Creating DevelSet predictions output folder:" , "   - " + DevelSetPredictionOutputFolder])
                            os.makedirs(DevelSetPredictionOutputFolder);
                        except Exception as E:
                            self.__PRJ.PROGRAM_Halt ("Error creating DevelSet predictions output folder. Error:" + E.message)
    
            #create TEST-SET predictin output SUB-folder (Architecture based)
            if (self.__RE_PARAMS.WriteBackTestSetPredictions == True):
                    TestSetPredictionOutputFolder = self.__RE_PARAMS.TestSetPredictionOutputFolder + ARCHITECTURE + "_" + str(useOnlyXTopPathsForFeatureGeneration) + "/" 
                    if not os.path.exists(TestSetPredictionOutputFolder):
                        try:
                            self.__PRJ.lp (["Creating TestSet predictions output folder:" , "   - " + TestSetPredictionOutputFolder])
                            os.makedirs(TestSetPredictionOutputFolder);
                        except Exception as E:
                            self.__PRJ.PROGRAM_Halt ("Error creating TestSet predictions output folder. Error:" + E.message)
            
            self.__BuildANNModel (ARCHITECTURE, _MaxSPCount, _MaxSPLength, _MaxFSLength, useOnlyXTopPathsForFeatureGeneration, RandomSeed) 


            for EpochCounter in range(MAX_EPOCH):
                self.__Train("TrainingSet")
                Eval_Devel_Metric = self.__PredictEvaluateWriteback("DevelopmentSet")
                Eval_Test_Metric  = self.__PredictEvaluateWriteback("TestSet")

                param_arch  = self.__RuntimeStatistics["SelectedArchitecture"]
                param_topkp = self.__RuntimeStatistics["TopKP"]
                param_round = self.__RuntimeStatistics["ExecutionRound"] 
                param_epoch = self.__RuntimeStatistics["NoEpochs"] 
                param_res   = [param_arch , param_topkp, param_round, param_epoch]

                if Eval_Devel_Metric <> None:
                    this_arch_eval = param_res + [Eval_Devel_Metric]
                    EvalDevel_AllResults.append (this_arch_eval)
                    EvalDevel_ThisArchResults.append (this_arch_eval)
                    
                if Eval_Test_Metric <> None:
                    this_arch_eval = param_res + [Eval_Test_Metric]
                    EvalTest_AllResults.append (this_arch_eval)
                    EvalTest_ThisArchResults.append (this_arch_eval)
            
            if len(EvalDevel_ThisArchResults) > 0:
                EvalDevel_ArchsBestResult.append ( sorted(EvalDevel_ThisArchResults , key=lambda x:x[-1] , reverse=True)[0] )
                EvalDevel_ArchsBestResult = sorted(EvalDevel_ArchsBestResult , key=lambda x:x[-1] , reverse=True)
            
            if len(EvalTest_ThisArchResults) > 0:
                EvalTest_ArchsBestResult.append ( sorted(EvalTest_ThisArchResults , key=lambda x:x[-1] , reverse=True)[0] )
                EvalTest_ArchsBestResult = sorted(EvalTest_ArchsBestResult , key=lambda x:x[-1] , reverse=True)

            #PRINT BEST RESULTS SO FAR ... 
            if len(EvalDevel_ArchsBestResult) > 0:
                MSG = ["*"*30+"\tBEST RESULTS SO FAR ON: DEVELOPMENT SET\t"+"*"*30]
                for i in EvalDevel_ArchsBestResult:
                    MSG += [i]
                self.__PRJ.lp (MSG) 
            
            if len(EvalTest_ArchsBestResult) > 0: 
                MSG = ["*"*30+"\tBEST RESULTS SO FAR ON: TEST SET\t"+"*"*30]
                for i in EvalTest_ArchsBestResult:
                    MSG+= [i]
                self.__PRJ.lp (MSG)
            
        return EvalDevel_AllResults , EvalDevel_ArchsBestResult , EvalTest_AllResults  , EvalTest_ArchsBestResult  
            
        """
        In simple pipeline:
            - get statstics for best results on develset : all_epochs_all_archs ---> best_epoch_per_arch and print
            - get statstics for best results on testset  : all_epochs_all_archs ---> best_epoch_per_arch and print 
            
        In optimization pipeline: (useful for (1) BioCreative, (2) KTop project 
            - optimize on devel set while NOT predicting test: all_epochs_all_archs ---> best_epoch_per_arch
            - Final test predictions: train on train+devel for found best_epoch epochs for each arch, predic test 
        """
        """        
        if self.__RE_PARAMS.PredictTestSet:
            #train on train+devel (using Best_C_value), predict test...
            #<<<CRITICAL>>> Train and Devel can add to feature dictionary, Test cannot
            self.__ResetFeatures()
            self.__GenerateFeatures ("TrainingSet"   , FeatureGenerationParams, True  ,useOnlyXTopPathsForFeatureGeneration)
            self.__GenerateFeatures ("DevelopmentSet", FeatureGenerationParams, True  ,useOnlyXTopPathsForFeatureGeneration)
            self.__GenerateFeatures ("TestSet"       , FeatureGenerationParams, False ,useOnlyXTopPathsForFeatureGeneration)
            self.__PRJ.lp ("Total Extracted Features:" + str(self.__PRJ.FeatureMappingDictionary.return_dictionary_length()))
            train_x , train_y , train_sentence_index, train_pair_tracking = self.__GenerateANNMatrices("TrainingSet"   , FeatureGenerationParams, returnLabels=True)
            devel_x , devel_y , devel_sentence_index, devel_pair_tracking = self.__GenerateANNMatrices("DevelopmentSet", FeatureGenerationParams, returnLabels=True) 
            test_x  , test_y  , test_sentence_index , test_pair_tracking  = self.__GenerateANNMatrices("TestSet"       , FeatureGenerationParams, returnLabels=self.__RE_PARAMS.EvaluateTestSet)
            
            from scipy.sparse import vstack
            TRAIN_X = vstack    ( [train_x, devel_x] )
            TRAIN_Y = np.hstack ( [train_y, devel_y] )

            from sklearn.svm import SVC        
            self.__PRJ.lp ("Training on TrainingSet+DevelopmentSet using found Best_C_Value")
            clf = SVC(C=devel_Best_C_value, kernel='linear', class_weight=self.__RE_PARAMS.Classification_class_weights) 
            clf.fit (TRAIN_X,TRAIN_Y)
            self.__PRJ.lp ("Predicting TestSet")
            y_pred = clf.predict (test_x)
            
            if self.__RE_PARAMS.EvaluateTestSet:
                y_true = test_y
                RES = self.__lowelevel_Evaluate (y_true, y_pred,test_pair_tracking,"TestSet")
                MSG = ["Test Set Prediction Results:" , "-"*30]
                for key in sorted(RES.keys()):
                    MSG.append (SF.NVLR (key,25) + ": " + str(RES[key]))
                MSG.append ("-"*80)
                MSG.append (self.__RE_PARAMS.Classification_OptimizationMetric + " : " + str(RES[self.__RE_PARAMS.Classification_OptimizationMetric]))
                test_Best_Metric_Value = RES[self.__RE_PARAMS.Classification_OptimizationMetric]
                self.__PRJ.lp(MSG)
                
            if self.__RE_PARAMS.WriteBackTestSetPredictions:
                GOLD_XML_FileAddress = self.__RE_PARAMS.TestSet_Files_Lists[0]
                OutputFileAddress    = self.__RE_PARAMS.TestSetPredictionOutputFolder + TestSetPredictionFileName 
                self.__lowelevel_WriteBackSetPredictionResults (y_pred, test_pair_tracking, GOLD_XML_FileAddress, OutputFileAddress)
                
        if self.__RE_PARAMS.EvaluateTestSet:
            return self.__PRJ.FeatureMappingDictionary.return_dictionary_length(), devel_Best_Metric_Value , test_Best_Metric_Value 
        else:
            return self.__PRJ.FeatureMappingDictionary.return_dictionary_length(), devel_Best_Metric_Value 
    """    
            
    def RunSimplePipeline_TrainOnTrainAndDevel(self,DataPreparationSteps, FeatureGenerationParams, Archictectures_MaxEpoch, useOnlyXTopPathsForFeatureGeneration=None, RandomSeed=None):
        self.__PRJ.lp (["-"*80,"FEATURE GENERATION PARAMS:" , "-"*30 , str(FeatureGenerationParams) , "-"*80])

        self.RunPreProcessing(DataPreparationSteps, FeatureGenerationParams)
        
        #train on train, optimize on devel-set, get best C_value
        #<<<CRITICAL>>> Train+Devel can add to feature dictionary, test cannot
        self.__ResetFeatures()
        self.__GenerateFeatures ("TrainingSet"   , FeatureGenerationParams, True  , useOnlyXTopPathsForFeatureGeneration)
        self.__GenerateFeatures ("DevelopmentSet", FeatureGenerationParams, True  , useOnlyXTopPathsForFeatureGeneration)
        self.__GenerateFeatures ("TestSet"       , FeatureGenerationParams, False , useOnlyXTopPathsForFeatureGeneration)

        self.__PRJ.lp ("Total Extracted Features:" + str(self.__PRJ.FeatureMappingDictionary.return_dictionary_length()))
        
        selected_sets = ["TrainingSet" , "DevelopmentSet" , "TestSet"]
             
        if useOnlyXTopPathsForFeatureGeneration == None:
            _MaxSPCount = self.__GetSPMaxCount (sets=selected_sets)
        else:
            _MaxSPCount = useOnlyXTopPathsForFeatureGeneration

        _MaxSPLength = self.__GetSPMaxLength (useOnlyXTopPathsForFeatureGeneration, sets=selected_sets)
        _MaxFSLength = self.__GetFSMaxLength (useOnlyXTopPathsForFeatureGeneration, sets=selected_sets)
        
        self.__ResetMatrices () 
        self.__GenerateANNMatrices("TrainingSet"   , FeatureGenerationParams, _MaxSPCount, _MaxSPLength, _MaxFSLength) 
        self.__GenerateANNMatrices("DevelopmentSet", FeatureGenerationParams, _MaxSPCount, _MaxSPLength, _MaxFSLength) 
        self.__GenerateANNMatrices("TestSet"       , FeatureGenerationParams, _MaxSPCount, _MaxSPLength, _MaxFSLength) 
        
        EvalDevel_AllResults , EvalDevel_ArchsBestResult = [] , []
        EvalTest_AllResults  , EvalTest_ArchsBestResult  = [] , []
        
        for ARCHITECTURE,MAX_EPOCH in Archictectures_MaxEpoch:
            EvalDevel_ThisArchResults , EvalTest_ThisArchResults = [] , [] 
            
            #create DEVELOPMENT-SET predictin output SUB-folder (Architecture based)
            if (self.__RE_PARAMS.WriteBackDevelSetPredictions == True):
                    DevelSetPredictionOutputFolder = self.__RE_PARAMS.DevelSetPredictionOutputFolder + ARCHITECTURE + "_" + str(useOnlyXTopPathsForFeatureGeneration) + "/"
                    if not os.path.exists(DevelSetPredictionOutputFolder):
                        try:
                            self.__PRJ.lp (["Creating DevelSet predictions output folder:" , "   - " + DevelSetPredictionOutputFolder])
                            os.makedirs(DevelSetPredictionOutputFolder);
                        except Exception as E:
                            self.__PRJ.PROGRAM_Halt ("Error creating DevelSet predictions output folder. Error:" + E.message)
    
            #create TEST-SET predictin output SUB-folder (Architecture based)
            if (self.__RE_PARAMS.WriteBackTestSetPredictions == True):
                    TestSetPredictionOutputFolder = self.__RE_PARAMS.TestSetPredictionOutputFolder + ARCHITECTURE + "_" + str(useOnlyXTopPathsForFeatureGeneration) + "/" 
                    if not os.path.exists(TestSetPredictionOutputFolder):
                        try:
                            self.__PRJ.lp (["Creating TestSet predictions output folder:" , "   - " + TestSetPredictionOutputFolder])
                            os.makedirs(TestSetPredictionOutputFolder);
                        except Exception as E:
                            self.__PRJ.PROGRAM_Halt ("Error creating TestSet predictions output folder. Error:" + E.message)
            
            self.__BuildANNModel (ARCHITECTURE, _MaxSPCount, _MaxSPLength, _MaxFSLength, useOnlyXTopPathsForFeatureGeneration, RandomSeed) 

            for EpochCounter in range(MAX_EPOCH):
                self.__Train(["TrainingSet","DevelopmentSet"])
                Eval_Devel_Metric = self.__PredictEvaluateWriteback("DevelopmentSet")
                Eval_Test_Metric  = self.__PredictEvaluateWriteback("TestSet")

                param_arch  = self.__RuntimeStatistics["SelectedArchitecture"]
                param_topkp = self.__RuntimeStatistics["TopKP"]
                param_round = self.__RuntimeStatistics["ExecutionRound"] 
                param_epoch = self.__RuntimeStatistics["NoEpochs"] 
                param_res   = [param_arch , param_topkp, param_round, param_epoch]

                if Eval_Devel_Metric <> None:
                    this_arch_eval = param_res + [Eval_Devel_Metric]
                    EvalDevel_AllResults.append (this_arch_eval)
                    EvalDevel_ThisArchResults.append (this_arch_eval)
                    
                if Eval_Test_Metric <> None:
                    this_arch_eval = param_res + [Eval_Test_Metric]
                    EvalTest_AllResults.append (this_arch_eval)
                    EvalTest_ThisArchResults.append (this_arch_eval)
            
            if len(EvalDevel_ThisArchResults) > 0:
                EvalDevel_ArchsBestResult.append ( sorted(EvalDevel_ThisArchResults , key=lambda x:x[-1] , reverse=True)[0] )
                EvalDevel_ArchsBestResult = sorted(EvalDevel_ArchsBestResult , key=lambda x:x[-1] , reverse=True)
            
            if len(EvalTest_ThisArchResults) > 0:
                EvalTest_ArchsBestResult.append ( sorted(EvalTest_ThisArchResults , key=lambda x:x[-1] , reverse=True)[0] )
                EvalTest_ArchsBestResult = sorted(EvalTest_ArchsBestResult , key=lambda x:x[-1] , reverse=True)

            #PRINT BEST RESULTS SO FAR ... 
            if len(EvalDevel_ArchsBestResult) > 0:
                MSG = ["*"*30+"\tBEST RESULTS SO FAR ON: DEVELOPMENT SET\t"+"*"*30]
                for i in EvalDevel_ArchsBestResult:
                    MSG += [i]
                self.__PRJ.lp (MSG) 
            
            if len(EvalTest_ArchsBestResult) > 0: 
                MSG = ["*"*30+"\tBEST RESULTS SO FAR ON: TEST SET\t"+"*"*30]
                for i in EvalTest_ArchsBestResult:
                    MSG+= [i]
                self.__PRJ.lp (MSG)
            
        return EvalDevel_AllResults , EvalDevel_ArchsBestResult , EvalTest_AllResults  , EvalTest_ArchsBestResult  
    
    def __lowelevel_Evaluate_WithGold (self, y_true, y_pred, y_pred_pair_tracking, Set):
        if not Set in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("key should be in " + str(self.__DATA.keys())) 
        
        if not (len(y_true)==len(y_pred)):
            self.__PRJ.PROGRAM_Halt ("Length of y_true and y_pred should be the same!");
            
        y_pred_dict = {}
        for sid , pid , e1 , e2 , e1tp , e2tp in y_pred_pair_tracking:
            y_pred_dict[pid] = (e1 , e2)

        y_true_extension = []
        
        for pid in self.__GOLD[Set]["positives"]:
            if pid in y_pred_dict:
                assert self.__GOLD[Set]["positives"][pid]["E1"] == y_pred_dict[pid][0]
                assert self.__GOLD[Set]["positives"][pid]["E2"] == y_pred_dict[pid][1]
            else: #False-Negative (classifier has missed to predict this relation), so we do: y_pred_extension.append(0)
                y_true_extension.append (self.__PRJ.Configs["OneHotEncodingForMultiClass"][self.__GOLD[Set]["positives"][pid]["CLASS_TP"]])
                
        for pid in self.__GOLD[Set]["negatives"]:
            if pid in y_pred_dict:
                assert self.__GOLD[Set]["negatives"][pid]["E1"] == y_pred_dict[pid][0]
                assert self.__GOLD[Set]["negatives"][pid]["E2"] == y_pred_dict[pid][1]
            else: #like Shared Task evaluations we ASSUME they have predicted it correctly to be negative , so y_pred_extension.append(0)
                y_true_extension.append (0) #0 is always negatives label
        
        #are counted as false negatives ...
        if len(y_true_extension) > 0:
            y_true_extension = np.array (y_true_extension,dtype=np.int8) 
            y_pred_extension = np.zeros (len(y_true_extension),dtype=np.int8)

            y_true = np.hstack ((y_true,y_true_extension))
            y_pred = np.hstack ((y_pred,y_pred_extension))
        
        return self.__lowelevel_Evaluate (y_true,y_pred)
                
    def __lowelevel_Evaluate (self, y_true, y_pred):
        #this is with the assumption that all relations ARE converted into machine learning examples,
        #however, in reality sometimes they can't. for example: head(e1)==head(e2) and e1/e2 are multi-token ... 
        if not (len(y_true)==len(y_pred)):
            self.__PRJ.PROGRAM_Halt ("Length of y_true and y_pred should be the same!");
        
        if self.__PRJ.Configs["ClassificationType"]=="binary":
            Precision, Recall, FScore, Support = METRICS.precision_recall_fscore_support (y_true, y_pred);
            COUNTS = METRICS.confusion_matrix (y_true , y_pred); 
            TP = COUNTS[1,1];
            TN = COUNTS[0,0];
            FP = COUNTS[0,1]; 
            FN = COUNTS[1,0];
            return OrderedDict([
                       ("confusion_TP"       , TP), 
                       ("confusion_TN"       , TN), 
                       ("confusion_FP"       , FP), 
                       ("confusion_FN"       , FN),
                       ("negative_precision" , Precision[0]),
                       ("negative_recall"    , Recall[0]), 
                       ("negative_f1_score"  , FScore[0]), 
                       ("negative_support"   , Support[0]),
                       ("positive_precision" , Precision[1]),
                       ("positive_recall"    , Recall[1]),
                       ("positive_f1_score"  , FScore[1]),
                       ("positive_support"   , Support[1]),
                       ("total_f1_macro"     , np.mean (FScore)), 
                       ("total_f1_weighted"  , ((FScore[0]*Support[0])+(FScore[1]*Support[1]))/float(Support[0]+Support[1]) ),
                       ])
                        
        else: #multi-class
            RES = OrderedDict()
            CLASSES = self.__PRJ.Configs["OneHotEncodingForMultiClass"] 
            IGNORE_CLASS_LABELS_FOR_EVALUATION = self.__PRJ.Configs["EvaluationParameters"]["ExcludeClassLabelsList"]
            INV_CLASSES = {} 
            for key in CLASSES:
                INV_CLASSES[CLASSES[key]] = key ; 
            
            PerClass_Precision, PerClass_Recall, PerClass_FScore, PerClass_Support = METRICS.precision_recall_fscore_support (y_true, y_pred, labels = range(len(CLASSES))) ;
            for i in range(len(CLASSES)):
                precision = PerClass_Precision[i]
                recall    = PerClass_Recall[i]
                f1score   = PerClass_FScore[i]
                support   = PerClass_Support[i]
                class_name = INV_CLASSES[i] + "_" 
                RES[class_name + "precision"] = precision
                RES[class_name + "recall"]    = recall
                RES[class_name + "f1_score"]  = f1score
                RES[class_name + "support"]   = support
            
            RES["total_f1_macro"]    = METRICS.f1_score (y_true, y_pred, labels = range(len(CLASSES)), average = "macro")
            RES["total_f1_micro"]    = METRICS.f1_score (y_true, y_pred, labels = range(len(CLASSES)), average = "micro")
            RES["total_f1_weighted"] = METRICS.f1_score (y_true, y_pred, labels = range(len(CLASSES)), average = "weighted")
            
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
                        
                IG_LBL = "total_MINUS_" + "_".join ([INV_CLASSES[i] for i in IG_IDX]) + "_" 
                REMAIN_IDX = [i for i in range(len(CLASSES)) if not i in IG_IDX]
                f1_macro     = METRICS.f1_score (y_true, y_pred, labels = REMAIN_IDX , average = "macro")
                f1_micro     = METRICS.f1_score (y_true, y_pred, labels = REMAIN_IDX , average = "micro")
                f1_weighted  = METRICS.f1_score (y_true, y_pred, labels = REMAIN_IDX , average = "weighted")
                RES[IG_LBL + "f1_macro"]    = f1_macro  
                RES[IG_LBL + "f1_micro"]    = f1_micro 
                RES[IG_LBL + "f1_weighted"] = f1_weighted

            return RES     

    def __lowelevel_WriteBackSetPredictionResults (self, y_pred, confs, pair_tracking, GOLD_XML_FileAddress, OutputFileAddress):
        if len(y_pred) <> len(confs):
            self.__PRJ.PROGRAM_Halt ("len(y_pred) is not equal to len(confs).")
        
        if len(y_pred) <> len(pair_tracking):
            self.__PRJ.PROGRAM_Halt ("Inconsistency between y_pred and pair_tracking!")
        
        TEESXMLHelper = TEESDocumentHelper.FLSTM_TEESXMLHelper (self.__PRJ.Configs , self.__PRJ.lp , self.__PRJ.PROGRAM_Halt) ; 

        #<<<TODO>>> PLEASE PLEASE TEST .... 
        #<<<CRITICAL>>> NOT TESTED .... 
        MULTICLASS_LABELS_DICT = {}; 
        for LABEL in self.__PRJ.Configs["WRITEBACK_OneHotEncodingForMultiClass"]:
            INDEX = self.__PRJ.Configs["WRITEBACK_OneHotEncodingForMultiClass"][LABEL]
            MULTICLASS_LABELS_DICT[INDEX] = LABEL ; 
        labels = [MULTICLASS_LABELS_DICT[i] for i in y_pred] ;
        
        WRITE_BACK_RES = [] ; 
        for idx , [S_ID, T_ID, E1_ID, E2_ID, E1_TP, E2_TP] in enumerate (pair_tracking):
            TYPE = labels[idx]; 
            CONFS = confs[idx,:];
            CONFS_STR = "" ; 
            for c in sorted(MULTICLASS_LABELS_DICT.keys()):
                CONFS_STR += MULTICLASS_LABELS_DICT[c]+":"+str(CONFS[c])+",";
            CONFS_STR = CONFS_STR[:-1]; 
            #print (S_ID,T_ID,TYPE,CONF, E1_TP, E2_TP) ;
            WRITE_BACK_RES.append ((S_ID,T_ID,TYPE,CONFS_STR, E1_ID, E2_ID, E1_TP,E2_TP)) ;
        
        GOLD_XML_TREE = TEESXMLHelper.LoadTEESFile_GetTreeRoot (GOLD_XML_FileAddress, ReturnWholeTree=True); 
        TEESXMLHelper.TREEOPERATION_FILL_Version2 (GOLD_XML_TREE , WRITE_BACK_RES, RemoveAllParseInfo=True ) ; #RemoveAllParseInfo, so that file size decrease dramatically ... no need to have parse info in pred files.
        self.__PRJ.lp ("WRITING BACK PREDICTION RESULTS TO FILE:" + OutputFileAddress) ; 
        TEESXMLHelper.Save_TEES_File (GOLD_XML_TREE , OutputFileAddress);
        return GOLD_XML_TREE ; 
