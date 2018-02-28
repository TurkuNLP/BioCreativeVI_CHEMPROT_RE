import os  ; 
import sys ; 
import inspect ; 
import json ; 

import GeneralFunctions as GF ; 
import Preprocessing as Preprocessor ; 
import FeatureMappingDictionary 
import FeatureGeneration

# to get current function name: inspect.stack()[0][3]  ... or: inspect.currentframe().f_code.co_name
# to get caller  function name: inspect.stack()[1][3]  ... or: inspect.currentframe().f_back.f_code.co_name  #second does not work for decorated method !

#deleted configs:
#Configs["ExampleGeneration"]["Generate_Reversed_SDP_Features"]

class TSDP_TSVM_ProjectRunner:
    def __init__ (self, PARAM_ConfigFileAddress , PARAM_LogFileAddress):
        self.__DO_NOT_SET_ME = None ; 
        
        try:
            print "-----------------------------------------------------";
            print "trying to create log file at:" , PARAM_LogFileAddress ; 
            self._LogFileHandler = open (PARAM_LogFileAddress , "wt");
            print "log file created successfully." ; 
            print "-----------------------------------------------------";
        except Exception as E:
            print "error creating the log file." ; 
            print "Error:" , E.message ; 
            print "Halting program!!!" ;
            sys.exit (-1); 
            
        self.lp ("Checking for configuration file address:" + PARAM_ConfigFileAddress); 
        PARAM_ConfigFileAddress = str(PARAM_ConfigFileAddress) ;
        if not os.path.isfile (PARAM_ConfigFileAddress):
            self.PROGRAM_Halt ("Configuration File Not Found");
        else:
            self.lp ("Config file found. Checking File ..."); 
        try:
            with open (PARAM_ConfigFileAddress) as LOCAL_json_fh:
                self.__DO_NOT_SET_ME = json.load (LOCAL_json_fh);
            self.lp ("FILE IS OKAY!" );
            self.lp (["CONFIGS:", "-"*20 , json.dumps (self.__DO_NOT_SET_ME,indent=4,sort_keys=True),"-"*80]) ; 
        except Exception as E:
            self.PROGRAM_Halt ("ERROR READING JSon Config File!\nError:"+E.message);
    
        #Are "Critical" Configs are given and correct ? 
        self.ConfigFileVerification (); 
        
        #What Objects should I have ... 
        self.wv = FeatureMappingDictionary.Word2VecEmbedding (self.lp , self.PROGRAM_Halt , self.Configs["W2V_Model"]["Model_Address"],self.Configs["W2V_Model"]["MaxWordsInMemory"])
        self.Preprocessor             = Preprocessor.TSDP_TSVM_Preprocessor (self.lp , self.PROGRAM_Halt , self.Configs)
        self.FeatureMappingDictionary = FeatureMappingDictionary.TSDP_TSVM_FeatureMappingDictionary (self.lp , self.PROGRAM_Halt)    
        self.POSTagsEmbeddings        = FeatureMappingDictionary.EmbeddingMappingDictionary (self.lp , self.PROGRAM_Halt , "POSTagsEmbeddings") 
        self.DPTypesEmbeddings        = FeatureMappingDictionary.EmbeddingMappingDictionary (self.lp , self.PROGRAM_Halt , "DPTypesEmbeddings") 
        self.PositionEmbeddings       = FeatureMappingDictionary.PositionEmbeddings (self.lp , self.PROGRAM_Halt)
        self.FeatureGenerator         = FeatureGeneration.TSDP_TSVM_FeatureGenerator (self.lp , self.PROGRAM_Halt , self.Configs, self.wv, self.FeatureMappingDictionary, self.POSTagsEmbeddings , self.DPTypesEmbeddings, self.PositionEmbeddings)
        
    def ConfigFileVerification (self):
        #1: Checking Class types
        if not "CLASSES" in self.Configs:
            self.PROGRAM_Halt ("Missing CLASSES dictionary in Config file."); 
        LOCAL_CLASSES = self.Configs["CLASSES"]; 
        if not "Negative" in LOCAL_CLASSES:
            self.PROGRAM_Halt ("Negative class should be defined in the config file."); 
        if not "Positive" in LOCAL_CLASSES:
            self.PROGRAM_Halt ("Positive class should be defined in the config file.");
        if len(LOCAL_CLASSES) <> 2:
            self.PROGRAM_Halt ('There should be only Positive and Negatives classes. For Multiclass use e.g., "Positive": ["class1", "class2"]');

        N , P = [] , [] ; 
        if isinstance (LOCAL_CLASSES["Negative"] , basestring):
            N.append (LOCAL_CLASSES["Negative"]);
        elif isinstance (LOCAL_CLASSES["Negative"] , list):
            N.extend (LOCAL_CLASSES["Negative"]); 
        if len(N) != len(set(N)):
            self.PROGRAM_Halt ("Negative class cannot have duplicate defined types."); 

        if isinstance (LOCAL_CLASSES["Positive"] , basestring):
            P.append (LOCAL_CLASSES["Positive"]);
        elif isinstance (LOCAL_CLASSES["Positive"] , list):
            P.extend (LOCAL_CLASSES["Positive"]); 
        if len(P) != len(set(P)):
            self.PROGRAM_Halt ("Positive class cannot have duplicate defined types."); 
        if len (set(N).intersection(set(P))) > 0:
            self.PROGRAM_Halt ("Shared defined class type between Positive and Negative class."); 
        
        self.__DO_NOT_SET_ME["CLASSES"] = {"Negative": None, "Positive":None} ; 
        self.__DO_NOT_SET_ME["CLASSES"]["Negative"] = set (i.lower() for i in N); #<<<CRITICAL>>> LOWER CLASS NAMES
        self.__DO_NOT_SET_ME["CLASSES"]["Positive"] = set (i.lower() for i in P); #<<<CRITICAL>>> LOWER CLASS NAMES
        
        #What labels should be used when writing back prediction results into xmls: 
        self.__DO_NOT_SET_ME["WRITEBACK_CLASSES"] = {"Negative": None, "Positive":None} ; 
        self.__DO_NOT_SET_ME["WRITEBACK_CLASSES"]["Negative"] = set (i for i in N); #<<<CRITICAL>>> LOWER CLASS NAMES
        self.__DO_NOT_SET_ME["WRITEBACK_CLASSES"]["Positive"] = set (i for i in P); #<<<CRITICAL>>> LOWER CLASS NAMES
        
        #Renaming relations into other relations ...
        #Good for lots of things, for example integrating many non-relevant relations into a single group
        if not "RENAME_CLASSES" in self.Configs:
            self.__DO_NOT_SET_ME["RENAME_CLASSES"] = {}
        else:
            if not isinstance (self.Configs["RENAME_CLASSES"], dict):
                self.PROGRAM_Halt ("RENAME_CLASSES in the config file should be a dictionary or not given at all."); 

            temp_rename_class = {}
            
            for key in self.Configs["RENAME_CLASSES"]:
                if not isinstance(key, basestring):
                    self.PROGRAM_Halt ("Problem in RENAME_CLASSES in the config file. Each keys in dictionary should be string. Problematic: " + str(key))
                    
                if isinstance (self.Configs["RENAME_CLASSES"][key],basestring):
                    x = self.Configs["RENAME_CLASSES"][key].lower()
                    if (not x in self.__DO_NOT_SET_ME["CLASSES"]["Negative"]) and (not x in self.__DO_NOT_SET_ME["CLASSES"]["Positive"]):
                        self.PROGRAM_Halt ("Problem in RENAME_CLASSES in the config file. Value should belong to defined positive or negative classes. Problematic: " + x); 
                    temp_rename_class[key.lower()] = x
                    
                elif isinstance(self.Configs["RENAME_CLASSES"][key],list):
                    allvalues = []
                    for x in self.Configs["RENAME_CLASSES"][key]:
                        x = x.lower()
                        if (not x in self.__DO_NOT_SET_ME["CLASSES"]["Negative"]) and (not x in self.__DO_NOT_SET_ME["CLASSES"]["Positive"]):
                            self.PROGRAM_Halt ("Problem in RENAME_CLASSES in the config file. Value should belong to defined positive or negative classes. Problematic: " + x); 
                        allvalues.append(x)
                    temp_rename_class[key.lower()] = allvalues

                else:
                    self.PROGRAM_Halt ("RENAME_CLASSES in the config file should be a dictionary, each value should be string or list of strings")
                    
            self.__DO_NOT_SET_ME["RENAME_CLASSES"] = temp_rename_class
            self.lp (["-"*80 , "[WARNING]: RENAMING RELATIONS INTO OTHER RELATIONS:" , str(self.__DO_NOT_SET_ME["RENAME_CLASSES"]) , "-"*80])
        #<<<CRITICAL>>>
        """
        # - OneHotEncodingForMultiClass: is used for real classification and prediction and creating the softmax columns. 
                   If we have 10 classes in total, and define 3 classes (i.e., "a","b","c") as Negative in the config file, 
                   the other 7 will be regarded as positive, and will have 8 columns in the softmax. Index 0 in the softmax
                   will be always for negative label(S). This has the artificial name "negative". 
                   
                   
                   1) Function: ProcessRoot in Preprocessing.py file:
                   ----------------------------------------------------------
                   if pair_type in self.Configs["CLASSES"]["Negative"]:
                       positive=False;
                       class_tp=None ;
                   elif pair_type in self.Configs["CLASSES"]["Positive"]:
                       positive=True;
                       class_tp=pair_type ; 
                   else:
                       self.PROGRAM_Halt ("Unknown class type for interaction:" + pair_type + "\n" + str(pair_attrib)); 



                   2) Function: Create_FeatureMatrix in NetworkInputGeneration.py file:
                   ----------------------------------------------------------
                    HowManyColumnsForOneHotEncoding = len (self.Configs["OneHotEncodingForMultiClass"]); 
                    y = np.zeros ((Total_Example_CNT, HowManyColumnsForOneHotEncoding),dtype=np.int16); 

                    ....
                    
                    if self.Configs["ClassificationType"]== "binary":
                        y [seen_pair_count] = 1 if (pair["POSITIVE"]==True) else 0 ; 
                    else: 
                        if pair["POSITIVE"]==False:
                            y[seen_pair_count,0]=1;#index zero is always for negative class(ES)!!!
                        else:
                            OneHotIndex = self.Configs["OneHotEncodingForMultiClass"][pair["CLASS_TP"]]; 
                            y[seen_pair_count, OneHotIndex] = 1 ;
        
                   3) Function __Evaluate in RelationExtractionPipeline.py:
                   ----------------------------------------------------------
                   USes for evaluation.
                  
        #- WRITEBACK_OneHotEncodingForMultiClass:
                  Like above, but this is used for writing back the prediction results into XML file. 
                  <<<CRITICAL>>>:
                  ALL negative predictions will be written with the negative class label of index 0 in list of "Negative" in the config file.
                  So:

                  "CLASSES" : 
                          {  "Negative" : ["neg" , "d"],   
                             "Positive" : ["a","b",c"]
                          }    
                  ---> ALL NEGATIVE PREDICTIONS WILL GET "neg" class label.


                  "CLASSES" : 
                          {  "Negative" : ["d" , "neg"],   
                             "Positive" : ["a","b",c"]
                          }    
                  ---> ALL NEGATIVE PREDICTIONS WILL GET "d" class label.
        """
        
        FIRSTneg = N[0]
        OneHotEncodingForMultiClass = {u"negative":0};
        WRITEBACK_OneHotEncodingForMultiClass = {FIRSTneg:0};
        for i, j in enumerate (sorted(P)):
            OneHotEncodingForMultiClass[j.lower()] = i+1 ;
            WRITEBACK_OneHotEncodingForMultiClass[j] = i+1 ;
        self.__DO_NOT_SET_ME["OneHotEncodingForMultiClass"] = OneHotEncodingForMultiClass ; 
        self.__DO_NOT_SET_ME["WRITEBACK_OneHotEncodingForMultiClass"] = WRITEBACK_OneHotEncodingForMultiClass ; 
        
        #Coarse class labels 
        #Example: Cause_Effect(e1,e2) and Cause_Effect(e2,e1) ==> Cause_Effect
        #First get list, then use SET (to avoird duplicates), then list and sorted. 
        Coarse_Classes = sorted(list(set([class_tp.split("(e1,e2)")[0].split("(e2,e1)")[0] for class_tp in self.__DO_NOT_SET_ME["CLASSES"]["Positive"]])));
        OneHotEncodingForCoarseMultiClass = {u"negative":0};
        for i, j in enumerate(Coarse_Classes):
            OneHotEncodingForCoarseMultiClass[j.lower()] = i+1 ;
        self.__DO_NOT_SET_ME["OneHotEncodingForCoarseMultiClass"] = OneHotEncodingForCoarseMultiClass ; 
            
        #2: ClassificationType
        if not "ClassificationType" in self.Configs:
            self.PROGRAM_Halt ("Missing Section: ClassificationType (should be either 'Binary' or 'Multiclass'."); 
        if not self.Configs["ClassificationType"].lower() in ['binary','multiclass']:
            self.PROGRAM_Halt ("Missing Section: ClassificationType (should be either 'Binary' or 'Multiclass'. GIVEN:" + str(self.Configs["ClassificationType"])); 
        self.__DO_NOT_SET_ME["ClassificationType"] = self.Configs["ClassificationType"].lower(); # <<<CRITICAL>>> LOWER CLASSIFICATION TYPE
        
        #3: ExampleGeneration
        if not "ExampleGeneration" in self.Configs:
            self.PROGRAM_Halt ("Missing ExampleGeneration dictionary in the config file."); 
        
        """
        if not "HaltIfNoSDP" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing bool HaltIfNoSDP in ExampleGeneration dictionary in the config file."); 
        if not isinstance (self.Configs["ExampleGeneration"]["HaltIfNoSDP"], bool):
            self.PROGRAM_Halt ("HaltIfNoSDP in the ExampleGeneration dictionary in the config file should be either true or false."); 

        if not "SDP_DIRECTION" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing string SDP_DIRECTION in ExampleGeneration dictionary in the config file."); 
        if not isinstance (self.Configs["ExampleGeneration"]["SDP_DIRECTION"], unicode):
            self.PROGRAM_Halt ("SDP_DIRECTION in ExampleGeneration dictionary in the config file should be string."); 
        SDP_DIRECTION = self.Configs["ExampleGeneration"]["SDP_DIRECTION"].lower(); 
        if not SDP_DIRECTION in ["from_e1value_to_e2value" , "from_e2value_to_e1value" , "from_firstoccurring_to_second" , "from_secondoccurring_to_first"]:
            self.PROGRAM_Halt ("SDP_DIRECTION in ExampleGeneration dictionary in the config file should be one of 'from_e1value_to_e2value' , 'from_e2value_to_e1value' , 'from_firstoccurring_to_second' , 'from_secondoccurring_to_first'."); 
        self.__DO_NOT_SET_ME["ExampleGeneration"]["SDP_DIRECTION"] = SDP_DIRECTION ; 
        
        
        if not "Generate_Reversed_SDP_Features" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing boolean Generate_Reversed_SDP_Features in ExampleGeneration dictionary in the config file."); 
        if not isinstance (self.Configs["ExampleGeneration"]["Generate_Reversed_SDP_Features"], bool):
            self.PROGRAM_Halt ("Generate_Reversed_SDP_Features in ExampleGeneration dictionary in the config file should be bool."); 
        
        if not "SDP_MAXLEN_BECAREFUL" in self.Configs["ExampleGeneration"]:
            self.__DO_NOT_SET_ME["ExampleGeneration"]["SDP_MAXLEN_BECAREFUL"] = None ; 
            self.lp ("[INFO]: SDP_MAXLEN_BECAREFUL is ignored. SDP will have length as max of bags.")
        else:
            if (not isinstance(self.Configs["ExampleGeneration"]["SDP_MAXLEN_BECAREFUL"], int)) or (self.Configs["ExampleGeneration"]["SDP_MAXLEN_BECAREFUL"] <= 1):
                self.PROGRAM_Halt ("SDP_MAXLEN_BECAREFUL in ExampleGeneration dictionary in the config file should be int and > 1."); 
        
        if not "Directional_Dependency_Types" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing boolean Directional_Dependency_Types in ExampleGeneration dictionary in the config file."); 
        else:
            if (not isinstance(self.Configs["ExampleGeneration"]["Directional_Dependency_Types"], bool)):
                self.PROGRAM_Halt ("Directional_Dependency_Types in ExampleGeneration dictionary in the config file should be either true or false."); 

        if not "Use_General_prep_prepc_conj_DT" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing boolean Use_General_prep_prepc_conj_DT in ExampleGeneration dictionary in the config file."); 
        else:
            if (not isinstance(self.Configs["ExampleGeneration"]["Use_General_prep_prepc_conj_DT"], bool)):
                self.PROGRAM_Halt ("Use_General_prep_prepc_conj_DT in ExampleGeneration dictionary in the config file should be either true or false."); 

        """
        
        #ActionON_CrossSentenceExamples: (1)Halt (2)Discard
        if not "ActionON_CrossSentenceExamples" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing string ActionON_CrossSentenceExamples in ExampleGeneration dictionary in the config file."); 
        else:
            if (not isinstance(self.Configs["ExampleGeneration"]["ActionON_CrossSentenceExamples"], unicode)):
                self.PROGRAM_Halt ("ActionON_CrossSentenceExamples in ExampleGeneration dictionary in the config file should be STRING and either 'Halt' or 'Discard'."); 

            self.Configs["ExampleGeneration"]["ActionON_CrossSentenceExamples"] = self.Configs["ExampleGeneration"]["ActionON_CrossSentenceExamples"].upper()
            if not (self.Configs["ExampleGeneration"]["ActionON_CrossSentenceExamples"] in ["HALT" , "DISCARD"]):
                self.PROGRAM_Halt ("ActionON_CrossSentenceExamples in ExampleGeneration dictionary in the config file should be either 'Halt' or 'Discard'."); 

            if self.Configs["ExampleGeneration"]["ActionON_CrossSentenceExamples"] == "DISCARD":
                self.lp (["*"*40 , "*"*40 ,"*"*40 , "[WARNING]: DISCARDING ALL CROSS-SENTECE RELATIONS IF THERE IS ANY !!!" , "*" *40, "*"*40 ,"*"*40]) ; 


        #ActionON_MissingRelations: (1)Halt (2)GenerateAsNegatives
        if not "ActionON_MissingRelations" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing string ActionON_MissingRelations in ExampleGeneration dictionary in the config file."); 
        else:
            if (not isinstance(self.Configs["ExampleGeneration"]["ActionON_MissingRelations"], unicode)):
                self.PROGRAM_Halt ("ActionON_MissingRelations in ExampleGeneration dictionary in the config file should be STRING and either 'Halt' or 'GenerateAsNegatives'."); 

            self.Configs["ExampleGeneration"]["ActionON_MissingRelations"] = self.Configs["ExampleGeneration"]["ActionON_MissingRelations"].upper()
            if not (self.Configs["ExampleGeneration"]["ActionON_MissingRelations"] in ["HALT" , "GENERATEASNEGATIVES"]):
                self.PROGRAM_Halt ("ActionON_MissingRelations in ExampleGeneration dictionary in the config file should be either 'Halt' or 'GenerateAsNegatives'."); 

            if self.Configs["ExampleGeneration"]["ActionON_MissingRelations"] == "GENERATEASNEGATIVES":
                self.lp (["*"*40,"*"*40,"*"*40, "[WARNING]: Missing Relations will be generated as Negatives !!!" ,"*"*40 ,"*"*40,"*" *40]); 


        #ActionON_DuplicateRelations
        if not "ActionON_DuplicateRelations" in self.Configs["ExampleGeneration"]:
            self.PROGRAM_Halt ("Missing string ActionON_DuplicateRelations in ExampleGeneration dictionary in the config file."); 
        else:
            if (not isinstance(self.Configs["ExampleGeneration"]["ActionON_DuplicateRelations"], unicode)):
                self.PROGRAM_Halt ("ActionON_DuplicateRelations in ExampleGeneration dictionary in the config file should be string and either 'Halt', 'Ignore', or 'Discard'."); 

            self.Configs["ExampleGeneration"]["ActionON_DuplicateRelations"] = self.Configs["ExampleGeneration"]["ActionON_DuplicateRelations"].upper()
            if not (self.Configs["ExampleGeneration"]["ActionON_DuplicateRelations"] in ["HALT" , "IGNORE" , "DISCARD"]):
                self.PROGRAM_Halt ("ActionON_MissingRelations in ExampleGeneration dictionary in the config file should be either 'Halt' or 'Ignore', or 'Discard'."); 

            if self.Configs["ExampleGeneration"]["ActionON_MissingRelations"] == "DISCARD":
                self.lp (["*"*40,"*"*40,"*"*40, "[WARNING]: DISCARDING ANY DUPLICATE RELATIONS !!!" ,"*"*40 ,"*"*40,"*" *40]); 

            
        #4: ValidEnityTypesForRelations
        if not "ValidEnityTypesForRelations" in self.Configs:
            self.PROGRAM_Halt ("Missing ValidEnityTypesForRelations dictionary in the config file."); 
        self.__DO_NOT_SET_ME["ValidEnityTypesForRelations"] = set ([i.lower() for i in self.Configs["ValidEnityTypesForRelations"]]); #<<<CRITICAL>>> LOWER

        OneHotEncodingForValidEnityTypesForRelations = {} ;
        for i , e_tp in enumerate(self.Configs["ValidEnityTypesForRelations"]):
            OneHotEncodingForValidEnityTypesForRelations[e_tp.lower()] = i ; 
        self.__DO_NOT_SET_ME["OneHotEncodingForValidEnityTypesForRelations"] = OneHotEncodingForValidEnityTypesForRelations ; 
        
        #5: InteractionElementName
        if not "InteractionElementName" in self.Configs:
            self.PROGRAM_Halt ("Missing InteractionElementName in the config file."); 
        if self.Configs["InteractionElementName"] == None:
            self.PROGRAM_Halt ("InteractionElementName in the config file should be either interaction or pair."); 
        if not self.Configs["InteractionElementName"].lower() in ['interaction','pair']:
            self.PROGRAM_Halt ("InteractionElementName in the config file should be either 'interaction' or 'pair'."); 
        self.__DO_NOT_SET_ME["InteractionElementName"] = self.Configs["InteractionElementName"].lower() ; #<<<CRITICAL>>> LOWER
            
        #6: InteractionElementClassAttributeName : in which attribute, class for interaction is given
        if not "InteractionElementClassAttributeName" in self.Configs:
            self.PROGRAM_Halt ("Missing InteractionElementClassAttributeName in the config file."); 
        if self.Configs["InteractionElementClassAttributeName"] == None:
            self.PROGRAM_Halt ("InteractionElementClassAttributeName in the config file should be either interaction or pair."); 
        if not self.Configs["InteractionElementClassAttributeName"].lower() in ['interaction','type']:
            self.PROGRAM_Halt ("InteractionElementClassAttributeName in the config file should be either 'interaction' or 'type'."); 
        self.__DO_NOT_SET_ME["InteractionElementClassAttributeName"] = self.Configs["InteractionElementClassAttributeName"].lower() ;
        
        #7: W2V
        if not "W2V_Model" in self.Configs:
            self.PROGRAM_Halt ("Missing W2V_Model dictionary in the config file."); 
        if not "Model_Address" in self.Configs["W2V_Model"]:
            self.PROGRAM_Halt ("Missing Model_Address in W2V_Model dictionary in config file.");
        if not "MaxWordsInMemory" in self.Configs["W2V_Model"]:
            self.PROGRAM_Halt ("Missing MaxWordsInMemory in W2V_Model dictionary in config file.");
        if not GF.FILE_CheckFileExists(self.Configs["W2V_Model"]["Model_Address"]):
            self.PROGRAM_Halt ("File address for W2V_Model is not valid: file not exists.");
        
        #8: Replace W2V vector of mentions with a better general W2V vector if mention not found in W2V model ...
        if "ReplaceVectorForEntityTypeIfTokenNotFound" in self.Configs["ExampleGeneration"]:
            VectorReplacementDict = {};
            for Entity_Type , ReplacementVector in self.Configs["ExampleGeneration"]["ReplaceVectorForEntityTypeIfTokenNotFound"]:
                VectorReplacementDict[Entity_Type.lower()] = ReplacementVector ;
            if len (VectorReplacementDict) > 0:
                self.lp (["REPLACING VECTORS FOR SOME ENTITIY TYPES IF NOT FOUND IN W2V Model" , str(VectorReplacementDict)]);
            self.__DO_NOT_SET_ME["ExampleGeneration"]["ReplaceVectorForEntityTypeIfTokenNotFound_Dict"]= VectorReplacementDict ; 
        else:
            self.__DO_NOT_SET_ME["ExampleGeneration"]["ReplaceVectorForEntityTypeIfTokenNotFound_Dict"]= {}; 
            
        
        #9: Check if ValidInteractingPairEntityTypes is given ...
        if not "ValidInteractingPairEntityTypes" in self.Configs:
            self.PROGRAM_Halt ("Missing ValidInteractingPairEntityTypes list in the config file."); 
        if self.Configs["ValidInteractingPairEntityTypes"] == None:
            self.PROGRAM_Halt ("ValidInteractingPairEntityTypes list is null in the config file."); 
        if len(self.Configs["ValidInteractingPairEntityTypes"]) < 1:
            self.PROGRAM_Halt ("ValidInteractingPairEntityTypes in the config file should be list with at least one pair."); 
        L = self.Configs["ValidInteractingPairEntityTypes"] ; 
        ListOf_ValidInteractingPairEntityTypes = [] ;  
        for EntityTypePair in L:
            if (not EntityTypePair[0].lower() in self.Configs["ValidEnityTypesForRelations"]) or \
               (not EntityTypePair[1].lower() in self.Configs["ValidEnityTypesForRelations"]) :  
                self.PROGRAM_Halt ("Invalid entity pairs for section ValidInteractingPairEntityTypes:" + str(EntityTypePair)); 
            ListOf_ValidInteractingPairEntityTypes.append ([EntityTypePair[0].lower(),EntityTypePair[1].lower()]);
            #Important comment out: 
            #    now we do not put both (e1tp,e2tp) and (e2tp,e1tp) here ...
            #    only preserve the original, so that we can create negatives according to this order for now ....
            #ListOf_ValidInteractingPairEntityTypes.append ([EntityTypePair[1].lower(),EntityTypePair[0].lower()]);
        self.__DO_NOT_SET_ME["ListOf_ValidInteractingPairEntityTypes"] = ListOf_ValidInteractingPairEntityTypes ; 
            
        #10: Can interact with itself: for example self-interacting proteins or drug ...
        if (not "SelfInteractingEntities" in self.Configs):
            self.PROGRAM_Halt ("Missing SelfInteractingEntities list in the config file. Set it to 'null' if you need nothing."); 
        if self.Configs["SelfInteractingEntities"] == None:
            self.__DO_NOT_SET_ME["SelfInteractingEntities"] = [] ; 
        else:
            L = [i.lower() for i in self.Configs["SelfInteractingEntities"]];
            for i in L:
                if not i in self.__DO_NOT_SET_ME["ValidEnityTypesForRelations"]:
                    self.PROGRAM_Halt ("A self-interacting entity type is defined in section SelfInteractingEntities, but not in ValidEnityTypesForRelations in Config file!"); 
                if not [i,i] in self.__DO_NOT_SET_ME["ListOf_ValidInteractingPairEntityTypes"]:
                    self.PROGRAM_Halt ("A self-interacting entity type is defined in section SelfInteractingEntities, but no relation in ValidInteractingPairEntityTypes in Config file!"); 
            self.__DO_NOT_SET_ME["SelfInteractingEntities"] = L ; 
            
        #11: RemoveSentenceIfNoParseExists
        if (not "RemoveSentenceIfNoParseExists" in self.Configs):
            self.PROGRAM_Halt ("Missing boolean RemoveSentenceIfNoParseExists in the config file."); 
        if (not isinstance (self.Configs["RemoveSentenceIfNoParseExists"] , bool)):
            self.PROGRAM_Halt ("RemoveSentenceIfNoParseExists in the config file should be either true or false."); 
            
        
        #12: Max Sentence Length
        if (not "MAX_SENTENCE_LENGTH" in self.Configs["ExampleGeneration"]):
            self.PROGRAM_Halt ("MAX_SENTENCE_LENGTH in ExampleGeneration should be either null or an integer bigger than zero in the config file."); 

        if self.Configs["ExampleGeneration"]["MAX_SENTENCE_LENGTH"] == None:
            self.Configs["ExampleGeneration"]["MAX_SENTENCE_LENGTH"] = -1 ;
        else:
            try:
                if isinstance (self.Configs["ExampleGeneration"]["MAX_SENTENCE_LENGTH"] , bool): raise Exception ("") ;  
                if not isinstance (self.Configs["ExampleGeneration"]["MAX_SENTENCE_LENGTH"] , int): raise Exception ("") ;  
                L = int (self.Configs["ExampleGeneration"]["MAX_SENTENCE_LENGTH"]) ; 
                if not (L>0): raise Exception ("") ; 
            except:
                self.PROGRAM_Halt ("MAX_SENTENCE_LENGTH in ExampleGeneration should be either null or an integer bigger than zero in the config file."); 
       
        #13: Runtime Parameters:
        if (not "ExecutionParameters" in self.Configs):
            self.PROGRAM_Halt ("Missing dictionary ExecutionParameters in the config file."); 
        if (not "DoNotAskAnyQuestions" in self.Configs["ExecutionParameters"]):
            self.PROGRAM_Halt ("Missing boolean DoNotAskAnyQuestions in ExecutionParameters dictionary in the config file."); 
        if not isinstance (self.Configs["ExecutionParameters"]["DoNotAskAnyQuestions"], bool):
            self.PROGRAM_Halt ("DoNotAskAnyQuestions in ExecutionParameters dictionary in the config file should be true/false."); 
        
        #14: Evaluation Parameters
        if (not "EvaluationParameters" in self.Configs):
            self.PROGRAM_Halt ("Missing dictionary EvaluationParameters in the config file."); 
        
        if (not "ExcludeClassLabelsList" in self.Configs["EvaluationParameters"]):
            self.PROGRAM_Halt ("Missing list ExcludeClassLabelsList in EvaluationParameters dictionary in the config file."); 
        if not isinstance (self.Configs["EvaluationParameters"]["ExcludeClassLabelsList"], list):
            self.PROGRAM_Halt ("ExcludeClassLabelsList in EvaluationParameters dictionary in the config file should be a list. Put [] if you don't want to exclude anything."); 
        self.Configs["EvaluationParameters"]["ExcludeClassLabelsList"] = [i.lower() for i in self.Configs["EvaluationParameters"]["ExcludeClassLabelsList"]]; 
        if len (self.Configs["EvaluationParameters"]["ExcludeClassLabelsList"]) > 0:
            ALL_CLASSES = self.Configs["CLASSES"]["Positive"] | self.Configs["CLASSES"]["Negative"] ; 
            for i in self.Configs["EvaluationParameters"]["ExcludeClassLabelsList"]:
                if not i in ALL_CLASSES:
                    self.PROGRAM_Halt (i + " class label is defined in [EvaluationParameters][ExcludeClassLabelsList] in the config file  but not defined as a valid class in the CLASSES dictionary."); 
            
        if (not "DecimalPoints" in self.Configs["EvaluationParameters"]):
            self.PROGRAM_Halt ("Missing integer DecimalPoints in EvaluationParameters dictionary in the config file."); 
        if not isinstance (self.Configs["EvaluationParameters"]["DecimalPoints"], int):
            self.PROGRAM_Halt ("DecimalPoints in EvaluationParameters dictionary in the config file should be a positive integer. Put 2 for default."); 
        if self.Configs["EvaluationParameters"]["DecimalPoints"] <=0 :
            self.PROGRAM_Halt ("DecimalPoints in EvaluationParameters dictionary in the config file should be a positive integer. Put 2 for default."); 
        

    def __exit__ (self):
        self.lp ("Running destructor and closing log file."); 
        self.lp ("Exiting program."); 
        self.lp ("END."); 
        self._LogFileHandler.close (); 
        
    @property
    def Configs(self):
        return self.__DO_NOT_SET_ME

    def lp (self, ARG_msg): #log and print message
        try:        
            LOCAL_CallerClassName = str(inspect.stack()[1][0].f_locals["self"].__class__) ; 
        except:
            LOCAL_CallerClassName = "" ; 
            
        LOCAL_CallerFunction = inspect.currentframe().f_back.f_code.co_name ; 
        if isinstance(ARG_msg, basestring):
            ARG_msg = [ARG_msg] ;
        HEADER = "[" + GF.DATETIME_GetNowStr() + "] [" + LOCAL_CallerClassName + "." + LOCAL_CallerFunction + "]: " ; 
        print HEADER ;
        self._LogFileHandler.write (HEADER+"\n") ; 
        for itemstr in ARG_msg:
            try:
                itemstr = str(itemstr).replace ('\r','\n'); 
            except:
                itemstr = itemstr.encode('utf-8').replace ('\r','\n')

            for item in itemstr.split ("\n"):
                if len(item)==0:
                    item = "-" ;
                item = "      "+item ; 
                print item ; 
                self._LogFileHandler.write (item+"\n") ;
        print "" ;
        self._LogFileHandler.write ("\n") ; 
        self._LogFileHandler.flush () 
        
    def PROGRAM_Halt (self, ARG_HaltErrMSG):
        PARAM_CallerFunctionName = inspect.currentframe().f_back.f_code.co_name ; 
        self.lp (["*"*80 , "HALT REQUESTED BY FUNCTION: " + PARAM_CallerFunctionName , "HALT MESSAGE: "+ ARG_HaltErrMSG , "HALTING PROGRAM!!!" , "*"*80]);
        self.__exit__ (); 
        sys.exit (-1); 


