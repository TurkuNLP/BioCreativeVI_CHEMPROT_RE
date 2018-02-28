import os
import numpy as np 
import sklearn.metrics as METRICS ; 
from collections import OrderedDict

try:
    import cPickle as pickle 
except:
    import pickle 

import TEESDocumentHelper;
import ProjectRunner; 
import FeatureGeneration as FeGen
import GeneralFunctions as GF
import SharedFunctions as SF 

class TSDP_TSVM_RE_PARAMS:
    def __init__(self):
        self.TrainingSet_Files_List    = []
        self.DevelopmentSet_Files_List = []
        self.TestSet_Files_Lists       = []

        self.TrainingSet_DuplicationRemovalPolicy    = None #None: select according to the given config file, no set-specific policy
        self.DevelopmentSet_DuplicationRemovalPolicy = None #None: select according to the given config file, no set-specific policy
        self.TestSet_DuplicationRemovalPolicy        = None #None: select according to the given config file, no set-specific policy

        self.ShuffleTrainingSentences = False  #<<<CRITICAL>>> default value is FALSE for replicablity purpose, when using a particular random seed.
             
        self.Classification_class_weights = None 
        self.Classification_SVM_C_values  = [2**(x) for x in range(-12,16)]
        self.Classification_MaxNumberOfCores   = 2
        self.Classification_OptimizationMetric = "positive_f1_score" 
        self.Classification_OptimizationOutput = "Y" 
        self.Classification_ExternalEvaluator  = None 
        
        self.PredictDevelSet  = True
        self.EvaluateDevelSet = True
        self.WriteBackDevelSetPredictions  = False 
        self.DevelSetPredictionOutputFolder= None

        self.PredictTestSet  = False 
        self.EvaluateTestSet = False 
        self.WriteBackTestSetPredictions   = False 
        self.TestSetPredictionOutputFolder = None

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
            \
            "TrainingSet_DuplicationRemovalPolicy     :" + str(self.TrainingSet_DuplicationRemovalPolicy)    + "\n" + \
            "DevelopmentSet_DuplicationRemovalPolicy  :" + str(self.DevelopmentSet_DuplicationRemovalPolicy) + "\n" + \
            "TestSet_DuplicationRemovalPolicy         :" + str(self.TestSet_DuplicationRemovalPolicy)        + "\n" + \
            \
            "ShuffleTrainingSentences:" + str(self.ShuffleTrainingSentences) + "\n" + \
            \
            "Classification_class_weights      :" + str(self.Classification_class_weights)      + "\n" + \
            "Classification_SVM_C_values       :" + str(self.Classification_SVM_C_values)       + "\n" + \
            "Classification_OptimizationMetric :" + str(self.Classification_OptimizationMetric) + "\n" + \
            "Classification_OptimizationOutput :" + str(self.Classification_OptimizationOutput) + "\n" + \
            "Classification_ExternalEvaluator  :" + str(self.Classification_ExternalEvaluator)  + "\n\n" + \
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
            "SkipWholeSentences_TrainingSet    :" + str(self.SkipWholeSentences_TrainingSet)    + "\n" + \
            "SkipWholeSentences_DevelopmentSet :" + str(self.SkipWholeSentences_DevelopmentSet) + "\n" + \
            "SkipWholeSentences_TestSet        :" + str(self.SkipWholeSentences_TestSet)        + "\n" + \
            \
            "WriteBackProcessedTrainingSet                :" + str(self.WriteBackProcessedTrainingSet) + "\n" + \
            "WriteBackProcessedDevelopmentSet             :" + str(self.WriteBackProcessedDevelopmentSet) + "\n" + \
            "WriteBackProcessedTestSet                    :" + str(self.WriteBackProcessedTestSet) + "\n" + \
            "WriteBackProcessedTrainingSet_OutputFileAddress    :" + str(self.WriteBackProcessedTrainingSet_OutputFileAddress) + "\n" + \
            "WriteBackProcessedDevelopmentSet_OutputFileAddress :" + str(self.WriteBackProcessedDevelopmentSet_OutputFileAddress) + "\n" + \
            "WriteBackProcessedTestSet_OutputFileAddress        :" + str(self.WriteBackProcessedTestSet_OutputFileAddress) + "\n" ;
        return S ; 
        
       
class TSDP_TSVM_RE_Pipeline:
    def __init__(self, ConfigFileAddress, LogFileAddress, RE_PARAMS):
        self.__PRJ   = ProjectRunner.TSDP_TSVM_ProjectRunner (ConfigFileAddress, LogFileAddress) 
        self.__DATA  = {"TrainingSet":{}, "DevelopmentSet":{}, "TestSet": {}}
        self.__GOLD  = {"TrainingSet":{"positives":{}, "negatives":{}}, "DevelopmentSet":{"positives":{}, "negatives":{}}, "TestSet": {"positives":{}, "negatives":{}}}
        self.__model = None 
        self.__WhichFeaturesToUse   = None 
        self.__WhichOutputsToPredit = None 
        self.__PerformanceResults = [] ; 
        self.__ResetTraining ()
        self.__DecimalPoints = self.__PRJ.Configs["EvaluationParameters"]["DecimalPoints"]; 
        if not isinstance(RE_PARAMS , TSDP_TSVM_RE_PARAMS):
            self.__PRJ.PROGRAM_Halt ("RE_PARAMS should be an instance of TSDP_TSVM_RE_PARAMS"); 
        else:
            self.__RE_PARAMS = RE_PARAMS ; 
            self.__Verify_RE_PARAMS();
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
            
    def __ResetTraining (self):
        self.__RuntimeStatistics = {"ExecutionRound":0 , "RandomSeed": None}; 
        if self.__model <> None:        
            del self.__model;
        self.__model = None; 
        self.__WhichFeaturesToUse = None 
        self.__WhichOutputsToPredit = None 
        
        
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
            
            import os ; 
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
                            
    
    def __GenerateSVMMatrices (self,key, FeatureGenerationParams, returnLabels=False):
        if not key in self.__DATA.keys():
            self.__PRJ.PROGRAM_Halt ("Invalid key. Key should be in " + str(self.__DATA.keys()))

        from scipy.sparse import csr_matrix
        import SharedFunctions as SF
        
        all_sentences = self.__DATA[key]["allsentences"] 
        pos_neg_dict, class_tp_dict, total_example_count = SF.CalculateHowManyRelationsWithShortestPathInDataset (all_sentences)
        
        if total_example_count < 1:
            self.__PRJ.PROGRAM_Halt (key + " has no examples!")
            
        SENTENCE_INDEX = np.zeros (total_example_count, dtype=np.int32) #for pairs .. so, if sentence one has 3 examples and sentence two has four --> 1112222 
        PAIR_TRACKING  = [] 

        if returnLabels:
            Y = np.zeros (total_example_count, dtype=np.int8)
        else:
            Y = None

        seen_pair_count = 0 

        if len(self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"]) > 1: #there are at least two entity types, so it worths to make clear which is which for each example 
            feature_entity_type = np.zeros ((total_example_count,2*len(self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"])), dtype=np.int64) 
            
        indptr = [0]
        indices = []
        for Sentence_Index , S in enumerate (all_sentences):
            for pair in S["PAIRS"]:
                if (not pair.has_key("TOPKP")) or (pair["TOPKP"] == None):
                    continue 
                if (not pair.has_key("TOPKP_Features")) or (pair["TOPKP_Features"] == None): #an example might have pair["TOPKP_Features"] = [] in a very rare and very very unlikely condition
                    continue 

                #1-Track: for later writing-back the prediction results into XML
                e1_type = S["ENTITIES"][pair["E1"]]["TYPE"]; 
                e2_type = S["ENTITIES"][pair["E2"]]["TYPE"]; 
                _e1tp = e1_type.capitalize()
                _e2tp = e2_type.capitalize()
                PAIR_TRACKING.append ( (S["ID"] , pair["ID"] , pair["E1"] , pair["E2"], _e1tp , _e2tp) )
                    
                #2-TOPKP_Features
                """
                the following code-snippet shows howo to efficiently create sparce matrix for SVM ...
                X = [[1,3,5],[0,2,4],[7,8,9],[4],[1],[],[],[1,9]]
                indptr = [0]
                indices = []
                for x in X:
                    indices.extend(x)
                    indptr.append(len(indices))
                print csr_matrix(([1]*len(indices), indices, indptr) ,dtype=int).toarray()
                """
                indices.extend (pair["TOPKP_Features"])
                indptr.append(len(indices))

                #3-entity type and their order features ...
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
                            
                #4-Label:
                if returnLabels:
                    if self.__PRJ.Configs["ClassificationType"]== "binary":
                        Y[seen_pair_count] = 1 if (pair["POSITIVE"]==True) else 0 ; 
                        
                    else: #MULTICLASS:
                        #Positive/Negative:
                        if pair["POSITIVE"]==False:
                            Y [seen_pair_count]=0;#index zero is always for negative class(es)
                        else:
                            class_label = pair["CLASS_TP"] 
                            OneHotIndex = self.__PRJ.Configs["OneHotEncodingForMultiClass"][class_label]
                            Y[seen_pair_count] = OneHotIndex
                            
                #5-Increament index ...
                SENTENCE_INDEX[seen_pair_count] = Sentence_Index;
                seen_pair_count+=1 ; 
        
        #<<<CRITICAL>>> giving shape is critical, because devel/test might have lower features ... 
        X = csr_matrix(([1]*len(indices), indices, indptr), shape = (seen_pair_count, self.__PRJ.FeatureMappingDictionary.return_dictionary_length()), dtype=np.int8)

        if len(self.__PRJ.Configs["OneHotEncodingForValidEnityTypesForRelations"]) > 1: #there are at least two entity types
            from scipy.sparse import hstack
            X = hstack ( [X, csr_matrix(feature_entity_type)] )
       
        return X, Y, SENTENCE_INDEX , PAIR_TRACKING 
        
    def RunPipeline(self,DataPreparationSteps, FeatureGenerationParams, useOnlyXTopPathsForFeatureGeneration=None,TestSetPredictionFileName=None):
        self.__PRJ.lp (["*"*80,"Running pipeline:","-"*17,"DataPreparationSteps:"+str(DataPreparationSteps),"MachineLearningTasks:"+str("MachineLearningTasks"),"*"*80]);

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
        
        if "findshortestpaths" in DataPreparationSteps and DataPreparationSteps["findshortestpaths"]==True:
            self.__RestShortestPathsAndFeatures ()
            for key in self.__DATA:            
                self.__FindTopKShortestPaths (key, FeatureGenerationParams)

        if "save_gold_and_data" in DataPreparationSteps and DataPreparationSteps["save_gold_and_data"] <> None:
            self.__load_save_GOLD_DATA ("save", fileaddress=DataPreparationSteps["save_gold_and_data"])

        MSG = ["MACHINE LEARNING EXAMPLES STATISTICS:", "-"*130]
        MSG.append (" "*20+SF.NVLL("Positives",30)+SF.NVLL("Negatives",30)+SF.NVLL("Total",30))
        for key in self.__DATA:
            all_sentences = self.__DATA[key]["allsentences"] 
            pos_neg_dict, class_tp_dict, total_example_count = SF.CalculateHowManyRelationsWithTOPKPaths (all_sentences)
            MSG.append ( SF.NVLR (key+" GOLD", 20) + SF.NVLL (str(len(self.__GOLD[key]["positives"])),30) + SF.NVLL (str(len(self.__GOLD[key]["negatives"])),30) + SF.NVLL (str(len(self.__GOLD[key]["positives"])+len(self.__GOLD[key]["negatives"])),30) )
            MSG.append ( SF.NVLR (key+" MLEx", 20) + SF.NVLL (str(pos_neg_dict["Positives"]),30) + SF.NVLL (str(pos_neg_dict["Negatives"]),30) +  SF.NVLL (str(total_example_count),30))
            MSG.append ("-"*130)
        self.__PRJ.lp (MSG)
            
                  
        #train on train, optimize on devel-set, get best C_value
        #<<<CRITICAL>>> Train can add to feature dictionary, devel cannot
        self.__ResetFeatures()
        self.__GenerateFeatures ("TrainingSet"   , FeatureGenerationParams, True  ,useOnlyXTopPathsForFeatureGeneration)
        self.__GenerateFeatures ("DevelopmentSet", FeatureGenerationParams, False ,useOnlyXTopPathsForFeatureGeneration)
        self.__PRJ.lp ("Total Extracted Features:" + str(self.__PRJ.FeatureMappingDictionary.return_dictionary_length()))
        train_x , train_y , train_sentence_index, train_pair_tracking = self.__GenerateSVMMatrices("TrainingSet"    ,FeatureGenerationParams, returnLabels=True)
        devel_x , devel_y , devel_sentence_index, devel_pair_tracking = self.__GenerateSVMMatrices("DevelopmentSet", FeatureGenerationParams, returnLabels=True) 

        devel_Best_C_value , devel_Best_Metric_Value = self.__lowlevel_OptimizeOnDevel_Parallel (train_x,train_y,devel_x,devel_y,devel_pair_tracking)
        
        if self.__RE_PARAMS.PredictTestSet:
            #train on train+devel (using Best_C_value), predict test...
            #<<<CRITICAL>>> Train and Devel can add to feature dictionary, Test cannot
            self.__ResetFeatures()
            self.__GenerateFeatures ("TrainingSet"   , FeatureGenerationParams, True  ,useOnlyXTopPathsForFeatureGeneration)
            self.__GenerateFeatures ("DevelopmentSet", FeatureGenerationParams, True  ,useOnlyXTopPathsForFeatureGeneration)
            self.__GenerateFeatures ("TestSet"       , FeatureGenerationParams, False ,useOnlyXTopPathsForFeatureGeneration)
            self.__PRJ.lp ("Total Extracted Features:" + str(self.__PRJ.FeatureMappingDictionary.return_dictionary_length()))
            train_x , train_y , train_sentence_index, train_pair_tracking = self.__GenerateSVMMatrices("TrainingSet"   , FeatureGenerationParams, returnLabels=True)
            devel_x , devel_y , devel_sentence_index, devel_pair_tracking = self.__GenerateSVMMatrices("DevelopmentSet", FeatureGenerationParams, returnLabels=True) 
            test_x  , test_y  , test_sentence_index , test_pair_tracking  = self.__GenerateSVMMatrices("TestSet"       , FeatureGenerationParams, returnLabels=self.__RE_PARAMS.EvaluateTestSet)
            
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
                RES = self.__lowelevel_Evaluate_WithGold (y_true, y_pred,test_pair_tracking,"TestSet")
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
            
    def __lowlevel_OptimizeOnDevel_Parallel (self,train_x,train_y,devel_x,devel_y,devel_pair_tracking):
        import multiprocessing
        #decide number of parallel processes:

        MAX_CORE_COUNT = multiprocessing.cpu_count()
        NumberOfParallelProcesses = self.__RE_PARAMS.Classification_MaxNumberOfCores 
        
        if (not isinstance(NumberOfParallelProcesses,int)):
            NumberOfThreads = MAX_CORE_COUNT - 1
        elif NumberOfParallelProcesses == -1:
            NumberOfThreads = MAX_CORE_COUNT - 1
        elif NumberOfParallelProcesses in range(1,MAX_CORE_COUNT): # range(1,4) --> 1,2,3 
            NumberOfThreads = NumberOfParallelProcesses 
        else:
            NumberOfThreads = MAX_CORE_COUNT - 1
        
        #important ...         
        if len (self.__RE_PARAMS.Classification_SVM_C_values) < NumberOfThreads:
            NumberOfThreads = len (self.__RE_PARAMS.Classification_SVM_C_values)
        
        self.__PRJ.lp (["*"*80,"Running optimization sub-pipeline with " + str(NumberOfThreads) + " cores.",
                        "C-Values: " + str(self.__RE_PARAMS.Classification_SVM_C_values),"-"*80])
        
        #for retrieving return values ...        
        manager = multiprocessing.Manager()
        return_Dict = manager.dict()

        #partitioning sentences ...         
        SubCValues = SF.PartitionIntoEXACTLYNPartitions (self, self.__RE_PARAMS.Classification_SVM_C_values,NumberOfThreads)
        
        my_all_threads = [] 
        for i in range(NumberOfThreads):
            p = multiprocessing.Process (target=self.__lowlevel_OptimizeOnDevel, args=(i, SubCValues[i],train_x,train_y,devel_x,devel_y,devel_pair_tracking,return_Dict))
            my_all_threads.append (p)
            p.start()
        
        #force NOT TO quite this function until all jobs are done!
        for p in my_all_threads:
            p.join()
        
        #gather results in a nice way and return ... 
        MSG = ["-"*80, "Optimization on Development Set results:", "-"*40]
        
        assert len(return_Dict) == len (self.__RE_PARAMS.Classification_SVM_C_values)
        results = [] 
        for c_value in return_Dict.keys():
            results.append ((c_value, return_Dict[c_value][self.__RE_PARAMS.Classification_OptimizationMetric]))
        
        results = sorted(results, key=lambda x:x[1], reverse=True)
        for c_value , metric_value in results:
            MSG.append ("C_value: " + SF.NVLL(str(c_value),15) + "\t" + self.__RE_PARAMS.Classification_OptimizationMetric + " : " + str(metric_value)) 
        MSG.append ("-"*80)
        Best_Metric_Value = results[0][1]
        MSG.extend (["Best_Metric_Value  = " + str(Best_Metric_Value), "-"*40])
        
        #<<<CRITICAL>>> if there are multiple c_values giving the best score, 
        #let's select their smallest, cause it gives highest regularization (simplest model), i.e., 
        # model that has higher training error, but larger margin hyperplane
        final_c_values = [] 
        for c_value , metric_value in results:
            if metric_value == Best_Metric_Value:
                final_c_values.append (c_value)
        Best_C_value = min(final_c_values)
        MSG.extend (["Best_C_value (min) = " + str(Best_C_value), "-"*40])
        self.__PRJ.lp (MSG)
        return Best_C_value , Best_Metric_Value

    def __lowlevel_OptimizeOnDevel (self, i, SubCValues,train_x,train_y,devel_x,devel_y,devel_pair_tracking,return_Dict):
        from sklearn.svm import SVC        
        y_true = devel_y 
        for c_value in SubCValues:
            print "process#"+str(i) + " Training on Training Set, C_Value : " + str(c_value)
            clf = SVC(C=c_value, kernel='linear', class_weight=self.__RE_PARAMS.Classification_class_weights) 
            clf.fit (train_x,train_y)
            print "process#"+str(i) + " Predicting Development Set, C_Value : " + str(c_value)
            y_pred = clf.predict (devel_x)
            RES = self.__lowelevel_Evaluate_WithGold (y_true, y_pred, devel_pair_tracking, "DevelopmentSet")
            return_Dict[c_value] = RES
    
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
            return {
                   "positive_precision": float(GF.FDecimalPoints(Precision[1], self.__DecimalPoints)),
                   "positive_recall"   : float(GF.FDecimalPoints(Recall[1]   , self.__DecimalPoints)), 
                   "positive_f1_score" : float(GF.FDecimalPoints(FScore[1]   , self.__DecimalPoints)),
                   "positive_support"  : Support[1],
                   "negative_precision": float(GF.FDecimalPoints(Precision[0], self.__DecimalPoints)),
                   "negative_recall"   : float(GF.FDecimalPoints(Recall[0]   , self.__DecimalPoints)), 
                   "negative_f1_score" : float(GF.FDecimalPoints (FScore[0]  , self.__DecimalPoints)), 
                   "negative_support"  : Support[0],
                   "total_f1_macro"    : float(GF.FDecimalPoints(np.mean (FScore), self.__DecimalPoints)) , 
                   "total_f1_weighted" : float(GF.FDecimalPoints( (((FScore[0]*Support[0])+(FScore[1]*Support[1]))/float(Support[0]+Support[1])) , self.__DecimalPoints)),
                   "confusion_TP"      : TP , 
                   "confusion_TN"      : TN , 
                   "confusion_FP"      : FP , 
                   "confusion_FN"      : FN ,
                   "y_pred"            : y_pred, 
                  } 
        else: #multi-class
            RES = {} 
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
                RES[class_name + "precision"] = float(GF.FDecimalPoints(precision, self.__DecimalPoints))
                RES[class_name + "recall"]    = float(GF.FDecimalPoints(recall   , self.__DecimalPoints))
                RES[class_name + "f1_score"]  = float(GF.FDecimalPoints(f1score  , self.__DecimalPoints))
                RES[class_name + "support"]   = support

            
            RES["total_f1_macro"]    = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred, labels = range(len(CLASSES)), average = "macro")    , self.__DecimalPoints)
            RES["total_f1_micro"]    = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred, labels = range(len(CLASSES)), average = "micro")    , self.__DecimalPoints)
            RES["total_f1_weighted" ]= GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred, labels = range(len(CLASSES)), average = "weighted") , self.__DecimalPoints)
            
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
                f1_macro     = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred, labels = REMAIN_IDX , average = "macro")    , self.__DecimalPoints)
                f1_micro     = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred, labels = REMAIN_IDX , average = "micro")    , self.__DecimalPoints)
                f1_weighted  = GF.FDecimalPoints ( METRICS.f1_score (y_true, y_pred, labels = REMAIN_IDX , average = "weighted") , self.__DecimalPoints)
                RES[IG_LBL + "f1_macro"]    = f1_macro  
                RES[IG_LBL + "f1_micro"]    = f1_micro 
                RES[IG_LBL + "f1_weighted"] = f1_weighted

            return RES     

    def __lowelevel_WriteBackSetPredictionResults (self, y_pred, PAIR_TRACKING, GOLD_XML_FileAddress, OutputFileAddress):
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
        
        WRITE_BACK_RES = [] 
        CONFS_STR = None
        for idx , [S_ID, T_ID, E1_ID, E2_ID, E1_TP, E2_TP] in enumerate (PAIR_TRACKING):
            TYPE = labels[idx]; 
            WRITE_BACK_RES.append ((S_ID,T_ID,TYPE,CONFS_STR, E1_ID, E2_ID, E1_TP,E2_TP)) ;
        
        GOLD_XML_TREE = TEESXMLHelper.LoadTEESFile_GetTreeRoot (GOLD_XML_FileAddress, ReturnWholeTree=True)
        TEESXMLHelper.TREEOPERATION_FILL_Version2 (GOLD_XML_TREE , WRITE_BACK_RES, RemoveAllParseInfo=True ) #RemoveAllParseInfo, so that file size decrease dramatically ... no need to have parse info in pred files.
        self.__PRJ.lp ("WRITING BACK PREDICTION RESULTS TO FILE:" + OutputFileAddress) ; 
        TEESXMLHelper.Save_TEES_File (GOLD_XML_TREE , OutputFileAddress);
        return GOLD_XML_TREE ; 
            