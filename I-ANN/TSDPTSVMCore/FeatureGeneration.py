"""
    def BuildBagsOfFeatures (self, Sentences):
        self.lp (["-"*80,"-"*80,"-"*80,"-"*20+"    BUILDING BAGS OF WORDS    "+"-"*30]); 
        for sentence in Sentences:
            # IMPORTANT: There can be some ISOLATED-components (nodes not connected to others) in the graph.
            DependencyParseGraph = self.LOWLEVEL_MakeDependencyParseGraphForOneSentence (sentence) ; 

            for pair in sentence["PAIRS"]:
                #IMPORTANT: MULTI-TOKEN ENTITIES ARE ALREADY RESOLVED TO THEIR SYNTACTIC [HEAD] SINGLE-TOKENS ...
                e1_charoffsets = sentence["ENTITIES"][pair["E1"]]["CHAROFFSETS"] ; 
                e2_charoffsets = sentence["ENTITIES"][pair["E2"]]["CHAROFFSETS"] ; 
                
                #<<<CRITICAL>>> VERY VERY VERY CRITICAL PART OF CODE ... sometimes offsets of the e2 is less than e1. <<<CRITICAL>>>
                if e2_charoffsets[0] < e1_charoffsets[0]:
                    (e1_charoffsets , e2_charoffsets) = (e2_charoffsets , e1_charoffsets) ;
                    
                e1_bgn , e1_end = e1_charoffsets[0] , e1_charoffsets[1]-1 ; #VERYYYYYYYYYYY CRITICAL ... 
                e2_bgn , e2_end = e2_charoffsets[0] , e2_charoffsets[1]-1 ; #VERYYYYYYYYYYYYYY CRITICAL ...
                
                h1_bgn,h1_end = sentence["ENTITIES"][pair["E1"]]["HEADOFFSET"][0] , sentence["ENTITIES"][pair["E1"]]["HEADOFFSET"][1]-1
                h2_bgn,h2_end = sentence["ENTITIES"][pair["E2"]]["HEADOFFSET"][0] , sentence["ENTITIES"][pair["E2"]]["HEADOFFSET"][1]-1
                
                TOKENS_INFO = sentence["TOKENS_INFO"] ; 
                pair["BAGS"] = {
                      "M1_Before"      : self.LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (TOKENS_INFO , 0      , e1_end                 , True  , True ) ,
                      "M1_Middle"      : self.LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (TOKENS_INFO , e1_bgn , e2_end                 , True  , True ) , 
                      "M1_After"       : self.LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (TOKENS_INFO , e2_bgn , len (sentence["TEXT"]) , True  , True ) , 
                      "M2_ForeBetween" : self.LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (TOKENS_INFO , 0      , e2_end                 , True  , True ) , 
                      "M2_Between"     : self.LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (TOKENS_INFO , e1_bgn , e2_end                 , True  , True ) , 
                      "M2_BetweenAfter": self.LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (TOKENS_INFO , e1_bgn , len (sentence["TEXT"]) , True  , True ) , 
                      "FullSentence"   : self.LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (TOKENS_INFO , 0      , len (sentence["TEXT"]) , True  , True ) , 
                      #"FullSentenceC"  : self.LOWLEVEL_GetTokensAndPosTags_FullSentenceCoding  (TOKENS_INFO , h1_bgn, h1_end , h2_bgn, h2_end) , 
                               }
                pair["TOPKP"] = self.LOWLEVEL_GenerateShortestPathFeatures (sentence, pair , DependencyParseGraph , ReverseDirection = False);
                
                if self.Configs["ExampleGeneration"]["Generate_Reversed_SDP_Features"]==True:
                    pair["TOPKP_REV"] = self.LOWLEVEL_GenerateShortestPathFeatures (sentence, pair , DependencyParseGraph , ReverseDirection = True); 

    def LOWLEVEL_MakeDependencyParseGraphForOneSentence (self, sentence):
        G = nx.MultiDiGraph () ; 
        for token_info in sentence["TOKENS_INFO"]:
            G.add_node (token_info[0] , POS = token_info[2] , TXT = token_info[1]);
        
        for edge_info in sentence["PARSE_EDGES"]:
            G.add_edge (edge_info[1] , edge_info[2] , TYPE= edge_info[-1]);
        return G ; 


    def LOWLEVEL_GetTokensAndPosTags_From_To_CharOffsets (self, TOKENS_INFO , charoff_beg, charoff_end , include_start , include_end):
        T , P = [] , [] ; #Tokens , POSIDX ... 

        if (include_start==True and include_end==True):
            for index , token_info in enumerate(TOKENS_INFO):
                #IsLastToken = (index == (len(TOKENS_INFO)-1)); 
                if (token_info[-1][0] >= charoff_beg) and (token_info[-1][0] <= charoff_end):
                    T.append (token_info[1]);#token itself
                    P.append (self.LOWLEVEL_GetPOSTAG_Coding (token_info[2]));#POS_TAG_IDX 

        elif (include_start==False and include_end==True):
            for index , token_info in enumerate(TOKENS_INFO):
                #IsLastToken = (index == (len(TOKENS_INFO)-1)); 
                if (token_info[-1][0] > charoff_beg) and (token_info[-1][0] <= charoff_end):
                    T.append (token_info[1]);#token itself
                    P.append (self.LOWLEVEL_GetPOSTAG_Coding (token_info[2]));#POS_TAG_IDX 
    
        elif (include_start==True and include_end==False):
            for index , token_info in enumerate(TOKENS_INFO):
                #IsLastToken = (index == (len(TOKENS_INFO)-1)); 
                if (token_info[-1][0] >= charoff_beg) and (token_info[-1][0] < charoff_end):
                    T.append (token_info[1]);#token itself
                    P.append (self.LOWLEVEL_GetPOSTAG_Coding (token_info[2]));#POS_TAG_IDX 
    
        elif (include_start==False and include_end==False):
            for index , token_info in enumerate(TOKENS_INFO):
                #IsLastToken = (index == (len(TOKENS_INFO)-1)); 
                if (token_info[-1][0] > charoff_beg) and (token_info[-1][0] < charoff_end):
                    T.append (token_info[1]);#token itself
                    P.append (self.LOWLEVEL_GetPOSTAG_Coding (token_info[2]));#POS_TAG_IDX 
    
        return T , P ;     

    def LOWLEVEL_GetTokensAndPosTags_FullSentenceCoding (self, TOKENS_INFO , foeh_start, foeh_end , soeh_start , soeh_end):
        #foeh_start  : first occuring entity head start
        #foeh_end    : first occuring entity head end
        #soeh_start  : second occuring entity head start
        #soeh_end    : second occuring entity head end    
        #coding: 1111123333334555555
        T , P , C = [] , [] , [] #Tokens , POSIDX , CODING

        #<<<CRITICAL>>>
        if soeh_start < foeh_start:
            foeh_start , soeh_start = soeh_start , foeh_start
            foeh_end   , soeh_end   = soeh_end   , foeh_end
            
        for index , token_info in enumerate(TOKENS_INFO):
            #IsLastToken = (index == (len(TOKENS_INFO)-1)); 
            T.append (token_info[1]);#token itself
            P.append (self.LOWLEVEL_GetPOSTAG_Coding (token_info[2]));#POS_TAG_IDX 
            token_start = token_info[-1][0]
            token_end   = token_info[-1][1]
            
            if (token_end < foeh_start):
                C.append (1)
            
            elif (token_start == foeh_start) and (token_end == foeh_end):
                C.append (2)

            elif (token_start > foeh_end) and (token_end < soeh_start):
                C.append (3)

            elif (token_start == soeh_start) and (token_end == soeh_end):
                C.append (4)
            
            elif (token_start > soeh_end):
                C.append (5)
        
        assert len(T) == len(C)
        
        return T,P,C

    def LOWLEVEL_GetPOSTAG_Coding (self, PosTag):
        if PosTag in PRJ_SharedVariables.PENN_TB_POS_TAGS:
            return PRJ_SharedVariables.PENN_TB_POS_TAGS[PosTag][0]; 
        else:
            self.NotFoundPOSTags.add (PosTag); 
            return PRJ_SharedVariables.PENN_TB_POS_TAGS["ADDED_OTHER"][0]


    def LOWLEVEL_Get_Stanford_DPTAG_Coding (self, DPTag , Forward_Direction):
        #S = DPTag + "   " + str(Forward_Direction) 
        
        use_direction  = self.Configs["ExampleGeneration"]["Directional_Dependency_Types"]
        use_general_dt = self.Configs["ExampleGeneration"]["Use_General_prep_prepc_conj_DT"]
        
        if use_general_dt:
            DPTag_prefix = DPTag.split ("_")[0] ; 
            if DPTag_prefix in ["prep" , "prepc" , "conj"]:
                DPTag = DPTag_prefix 
        
        if not use_direction:
            DPTag = "R_" + DPTag 
        else:
            if Forward_Direction:
                DPTag = "R_" + DPTag 
            else:
                DPTag = "L_" + DPTag 

        #self.lp (S+ "    >>> " + DPTag) ; 
                
        if DPTag in PRJ_SharedVariables.STANFORD_DEPTP_TAGS:
            return PRJ_SharedVariables.STANFORD_DEPTP_TAGS[DPTag][0]
            
        elif DPTag.split("_")[0] in PRJ_SharedVariables.STANFORD_DEPTP_TAGS:
            self.NotFoundDTTags_Mapped.add (DPTag)
            return PRJ_SharedVariables.STANFORD_DEPTP_TAGS[ DPTag.split("_")[0] ][0] 

        else:
            self.NotFoundDTTags_NotMapped.add (DPTag)
            return PRJ_SharedVariables.STANFORD_DEPTP_TAGS["ADDED_OTHER"][0]
            
    
    def LOWLEVEL_GenerateShortestPathFeatures (self, sentence, pair , DependencyParseGraph , ReverseDirection):
        #Initialization        
        HaltIfNoShortestPath = self.Configs["ExampleGeneration"]["HaltIfNoSDP"] ; 
        wv = self.wv ; 
        G = nx.MultiGraph (DependencyParseGraph); 
        GD = DependencyParseGraph ; 
        SP_FEATURE_WORDS , SP_FEATURE_POSTGS , SP_FEATURE_EDGE_DPTGS = [] , [] , [] ; 
        
        #Function to get HeadTokenID
        def GetHeadTokenID (E_ID):
            H_bgn = sentence["ENTITIES"][E_ID]["HEADOFFSET"][0];
            H_end = sentence["ENTITIES"][E_ID]["HEADOFFSET"][1];
            for token_info in sentence["TOKENS_INFO"]:
                token_bgn = token_info[-1][0];
                token_end = token_info[-1][1];
                if (token_bgn >= H_bgn) and (token_end <= H_end):
                    return token_info[0];
            self.PROGRAM_Halt ("HEAD_TOKEN_ID_NOT_FOUND! SID:" + sentence["ID"] + "\n" + str(pair));
        
        #What Policy should be used to generate SDP:
        #TODO: ADD Functionality to find shortest path from ?
        # For BioNLP ST: always from E1 to E2, and because in BB TEES E1 is always Bacteria and B2 is always LOCATION then no problem! This is what we wanted for paper.
        SDP_DIRECTION = self.Configs["ExampleGeneration"]["SDP_DIRECTION"].lower(); 
        if not SDP_DIRECTION in ["from_e1value_to_e2value" , "from_e2value_to_e1value"]:
            self.PROGRAM_Halt ("SDP_DIRECTION METHOD : " + SDP_DIRECTION + " IS NOT IMPLEMENTED YET!");
            
        if SDP_DIRECTION == "from_e1value_to_e2value":
            E1_ID = pair["E1"]; # ---> i.e, BB_EVENT_16.d34.s0.e3
            E2_ID = pair["E2"];
        elif SDP_DIRECTION == "from_e2value_to_e1value":
            E1_ID = pair["E2"]; # ---> i.e, BB_EVENT_16.d34.s0.e3
            E2_ID = pair["E1"];
        else:
            self.PROGRAM_Halt ("SDP_DIRECTION METHOD : " + SDP_DIRECTION + " IS NOT IMPLEMENTED YET!");
            
        #<<<CRITICAL>>>: Regardless of the policy, ReverseDirection will always generate features in the Reverse Direction!
        if ReverseDirection==True:
            E1_ID , E2_ID = E2_ID , E1_ID #<<<CRITICAL>>>
            
        #Now policy and direction are handled , lets get headtokens and start generating features ... oh lah lah! :D             
        E1_Head_Token_ID = GetHeadTokenID (E1_ID); #---> e.i., bt_14
        E2_Head_Token_ID = GetHeadTokenID (E2_ID);
        
        #assert int(E1_Head_Token_ID.split("_")[-1]) > int(E2_Head_Token_ID.split("_")[-1]) ; 
        
        E1_TYPE = sentence["ENTITIES"][E1_ID]["TYPE"].lower() ; 
        E2_TYPE = sentence["ENTITIES"][E2_ID]["TYPE"].lower() ; 
        
        TOKENS_DICT = {}; 
        for token_info in sentence["TOKENS_INFO"]:
            TOKENS_DICT[token_info[0]] = token_info ; # id --> info ...
        
        
        try:
            SHORTEST_PATH = nx.shortest_path (G, source = E1_Head_Token_ID , target= E2_Head_Token_ID); 
        except:
            self.lp ("[WARNING]: No Shortest-Path for sentence-id: " + str(sentence["ID"]) + "  pair-id: " + str(pair["ID"])); 
            if HaltIfNoShortestPath:
                #print "line 479" ;                 
                #import pdb ; 
                #pdb.set_trace (); 
                self.PROGRAM_Halt ("No Shortest Path !!! - \nsource: " + str(E1_Head_Token_ID)+"\ntarget: "+str(E2_Head_Token_ID));
            else:
                return None ; 
            
        def SpecialTreatmentForTwoEntities (token_id, E_TYPE, SP_FEATURE_WORDS):
            token_TEXT = TOKENS_DICT[token_id][1] ; 
            if not token_TEXT in wv:
                if E_TYPE in self.Configs["ExampleGeneration"]["ReplaceVectorForEntityTypeIfTokenNotFound_Dict"]:
                    REPLACEMENT_TXT = self.Configs["ExampleGeneration"]["ReplaceVectorForEntityTypeIfTokenNotFound_Dict"][E_TYPE];
                    SP_FEATURE_WORDS.append (REPLACEMENT_TXT);
                    self.lp ("REPLACEMENT -------------------- " + token_TEXT + " >>> " + REPLACEMENT_TXT) ; 
                else:
                    SP_FEATURE_WORDS.append (token_TEXT);#add the word itself. Will be mapped to unknown shared vector later!
            else:
                idx=wv[token_TEXT]; 
                if idx >= (wv.max_rank_mem - 1):
                    if E_TYPE in self.Configs["ExampleGeneration"]["ReplaceVectorForEntityTypeIfTokenNotFound_Dict"]:
                        REPLACEMENT_TXT = self.Configs["ExampleGeneration"]["ReplaceVectorForEntityTypeIfTokenNotFound_Dict"][E_TYPE];
                        SP_FEATURE_WORDS.append (REPLACEMENT_TXT);
                        self.lp ("REPLACEMENT -------------------- " + token_TEXT + " >>> " + REPLACEMENT_TXT) ; 
                    else:
                        SP_FEATURE_WORDS.append (token_TEXT);#add the word itself. Will be mapped to unknown shared vector later!
                else:
                    SP_FEATURE_WORDS.append (token_TEXT);

        for token_id in SHORTEST_PATH:
            #SHOULD WE CHANGE BACTERIA IF NOT IN THE MODEL ? for example "Salmonellae" ---> "bacteria" ... 
            #CRITICAL ..... 
            if (token_id == E1_Head_Token_ID):
                SpecialTreatmentForTwoEntities (token_id, E1_TYPE, SP_FEATURE_WORDS);
            
            elif (token_id == E2_Head_Token_ID):
                SpecialTreatmentForTwoEntities (token_id, E2_TYPE, SP_FEATURE_WORDS);

            else:
                SP_FEATURE_WORDS.append ( TOKENS_DICT[token_id][1] ); #WORD ITSELF
            
            SP_FEATURE_POSTGS.append(self.LOWLEVEL_GetPOSTAG_Coding (TOKENS_DICT[token_id][2]));
        
        for i in range(len(SHORTEST_PATH)-1):
            #<<<CRITICAL>>>
            if GD.has_edge (SHORTEST_PATH[i] , SHORTEST_PATH[i+1]):
                Forward_Direction = True
                edge_type = GD.edge[SHORTEST_PATH[i]][SHORTEST_PATH[i+1]][0]["TYPE"];#<<<CRITICAL>>>

            elif GD.has_edge(SHORTEST_PATH[i+1],SHORTEST_PATH[i]):
                Forward_Direction = False
                edge_type = GD.edge[SHORTEST_PATH[i+1]][SHORTEST_PATH[i]][0]["TYPE"];#<<<CRITICAL>>> sometimes, in reverse path there is another edge ...
                #example : ('t4', 't10', {'TYPE': 'rcmod'}), ('t4', 't0', {'TYPE': 'nsubj'})

            else:
                self.PROGRAM_Halt ("Error in finding dependenct type direction!\n"+str(GD.edges())) ; 
                
            SP_FEATURE_EDGE_DPTGS.append (self.LOWLEVEL_Get_Stanford_DPTAG_Coding(edge_type, Forward_Direction)) ; 

        return SP_FEATURE_WORDS , SP_FEATURE_POSTGS , SP_FEATURE_EDGE_DPTGS ; 

    def CalculateNumberOfExamplesPerSentence (self, Sentences):
        TOTAL_POSITIVES = 0 ;
        TOTAL_NEGATIVES = 0 ; 
        for S in Sentences:
            NumberOfExamples = {"negatives":0} ; 
            if (len(S["ENTITIES"])==0) or (len(S["PAIRS"])==0): #Second condition for those sentences that have one entity but self interaction which are apparently discarded in ANTTI work ...
                S["NumberOfExamples"] = NumberOfExamples; 
            else:
                for pair in S["PAIRS"]:
                    if pair.has_key("BAGS"):
                        if pair["TOPKP"] <> None:
                            if pair["POSITIVE"] == True:
                                if self.Configs["ClassificationType"]=="binary":

                                    if NumberOfExamples.has_key("positives"):
                                        NumberOfExamples["positives"]+=1 ; 
                                    else:
                                        NumberOfExamples["positives"]=1 ; 

                                else:
                                    CLASS_TP = pair["CLASS_TP"];
                                    if NumberOfExamples.has_key(CLASS_TP):
                                        NumberOfExamples[CLASS_TP]+=1 ; 
                                    else:
                                        NumberOfExamples[CLASS_TP]=1 ; 
                            else:
                                NumberOfExamples["negatives"]+=1;
                
                S["NumberOfExamples"] = NumberOfExamples; 
                TOTAL_POSITIVES += sum (S["NumberOfExamples"][t] for t in S["NumberOfExamples"] if t <> "negatives");
                TOTAL_NEGATIVES += S["NumberOfExamples"]["negatives"];
        self.lp (["-"*50 , "Calculating Number of Examples per sentence." , "-"*20, 
                  "EXAMPLE STATISTICS:", 
                  "NUMBER OF POSITIVES : " + str(TOTAL_POSITIVES) ,
                  "NUMBER OF NEGATIVES : " + str(TOTAL_NEGATIVES) , 
                  "TOTAL               : " + str(TOTAL_POSITIVES + TOTAL_NEGATIVES) , "-"*50]) ;


"""
""" DELETED PARAMS
           "ReplaceVectorForEntityTypeIfTokenNotFound" : [["Bacteria","bacteria"]],   
           "Method" : "M2" , 
           "SetAllBagsLengthsEqual" : true , 
           "PadVectorsFromLeft": false , 
           "M1_Bags_MaxLength" : 150 , 
           "M2_Bags_MaxLength" : 200 , 
           "SDP_MAXLEN_BECAREFUL" : 25, 
           "HaltIfNoSDP"   : false,
           "SDP_DIRECTION" : "from_e1value_to_e2value",
           "Generate_Reversed_SDP_Features" : false, 
           "Directional_Dependency_Types"   : false, 
       	 "Use_General_prep_prepc_conj_DT" : false,

           "TokenizationAndPOSTagging" : {
           "SpecialTagForLastToken" : false   
   },

"""
import networkx as nx
import GraphUtils2
import numpy as np 
class TSDP_TSVM_FS_PositionsMeaning:
    P1distanceTo1stOccurring_P2distanceTo2ndOccurring , P1distanceToe1value_P2distanceToe2value = range(2)
           
class TSDP_TSVM_SDPFeatureGenerationDirection:
    from_e1value_to_e2value, from_e2value_to_e1value , from_1stOccurring_to_2nd , from_2ndOccurring_to_1st = range(4)

class TSDP_TSVM_WhatToUseToCodeNode:
    none, word , lemma , pos = range(4)

class TSDP_TSVM_FeatureGenerationParams:
    def __init__(self):
        #Graph building and finding top K shortest paths
        self.SDPDirection = TSDP_TSVM_SDPFeatureGenerationDirection.from_1stOccurring_to_2nd
        self.EdgeWeight_WordAdjacency   = 4
        self.EdgeWeight_ParseDependency = 1
        self.TopKShortestPathNumber     = 1
        self.NumberOfParallelProcesses = -1 #use all core minus 1 , valid values: -1,1,2,3,4,5, ... 
        self.UsePathLengthCutOffHeuristic = True 
        self.PathLengthCutOffHeuristicMAXLEN = 20
        
        #Feature generation along the SDPs
        self.ReplaceVectorForEntityTypeIfTokenNotFound = {}        

        #For SHORTEST PATHS (ANN)        
        self.SP_DirectionalFeatureGeneration = True 
        self.SP_LowerAllTokens = False 
        
        #FOR ALL TOP_K_PATHS         
        self.TOPKP_DirectionalFeatureGeneration = True 
        self.TOPKP_LowerAllTokens = False 
        self.TOPKP_UseWord2VecForWords = False

        #Which features to generate ... 
        self.WholePathEncoding_edges_words      = True
        self.WholePathEncoding_edges_lemmas     = False
        self.WholePathEncoding_edges_POSTags    = True
        self.WholePathEncoding_edges_None       = True
        self.PathNGrams_words     = [1,2,3,4,5]
        self.PathNGrams_lemmas    = []
        self.PathNGrams_POSTags   = [1,2,3,4,5]
        self.PathNGrams_ConsecutiveEdges = [2,3]

        #Full-sentence feature generation
        self.FS_PositionEmbeddingsMeanings = TSDP_TSVM_FS_PositionsMeaning.P1distanceTo1stOccurring_P2distanceTo2ndOccurring
        self.FS_LowerAllTokens = False
        
    def __str__(self):
        S = "" + \
        "SDPDirection                              : " + str(self.SDPDirection) + "\n" + \
        "EdgeWeight_WordAdjacency                  : " + str(self.EdgeWeight_WordAdjacency) + "\n" + \
        "EdgeWeight_ParseDependency                : " + str(self.EdgeWeight_ParseDependency) + "\n" + \
        "TopKShortestPathNumber                    : " + str(self.TopKShortestPathNumber) + "\n" + \
        "NumberOfParallelProcesses                 : " + str(self.NumberOfParallelProcesses) + "\n" + \
        "UsePathLengthCutOffHeuristic              : " + str(self.UsePathLengthCutOffHeuristic) + "\n" + \
        "PathLengthCutOffHeuristicMAXLEN           : " + str(self.PathLengthCutOffHeuristicMAXLEN) + "\n" + \
        \
        "ReplaceVectorForEntityTypeIfTokenNotFound : " + str(self.ReplaceVectorForEntityTypeIfTokenNotFound) + "\n" + \
        "SP_DirectionalFeatureGeneration           : " + str(self.SP_DirectionalFeatureGeneration) + "\n" + \
        "SP_LowerAllTokens                         : " + str(self.SP_LowerAllTokens) + "\n" + \
        "TOPKP_DirectionalFeatureGeneration        : " + str(self.TOPKP_DirectionalFeatureGeneration) + "\n" + \
        "TOPKP_LowerAllTokens                      : " + str(self.TOPKP_LowerAllTokens) + "\n" + \
        "TOPKP_UseWord2VecForWords                 : " + str(self.TOPKP_UseWord2VecForWords) + "\n" + \
        \
        "WholePathEncoding_edges_words             : " + str(self.WholePathEncoding_edges_words) + "\n" + \
        "WholePathEncoding_edges_lemmas            : " + str(self.WholePathEncoding_edges_lemmas) + "\n" + \
        "WholePathEncoding_edges_POSTags           : " + str(self.WholePathEncoding_edges_POSTags) + "\n" + \
        "WholePathEncoding_edges_None              : " + str(self.WholePathEncoding_edges_None) + "\n" + \
        "PathNGrams_words                          : " + str(self.PathNGrams_words) + "\n" + \
        "PathNGrams_lemmas                         : " + str(self.PathNGrams_lemmas) + "\n" + \
        "PathNGrams_POSTags                        : " + str(self.PathNGrams_POSTags) + "\n" + \
        "PathNGrams_ConsecutiveEdges               : " + str(self.PathNGrams_ConsecutiveEdges) + "\n" + \
        "FS_PositionEmbeddingsMeanings             : " + str(self.FS_PositionEmbeddingsMeanings) + "\n" + \
        "FS_LowerAllTokens                         : " + str(self.FS_LowerAllTokens) + "\n" 
        return S ; 
        
class TSDP_TSVM_FeatureGenerator:
    def __init__(self, lp , PROGRAM_Halt , Configs, wv, FeatureMappingDictionary,POSTagsEmbeddings,DPTypesEmbeddings,PositionEmbeddings):
        self.lp = lp
        self.PROGRAM_Halt = PROGRAM_Halt
        self.Configs = Configs
        self.wv = wv
        self.FeatureMappingDictionary = FeatureMappingDictionary 
        self.POSTagsEmbeddings  = POSTagsEmbeddings
        self.DPTypesEmbeddings  = DPTypesEmbeddings
        self.PositionEmbeddings = PositionEmbeddings
        
    def CalculateAndAddTopKSDPs_Parallel (self, Sentences, FeatureGenerationParams,SetName=None):
        if not isinstance (FeatureGenerationParams,TSDP_TSVM_FeatureGenerationParams):
            self.PROGRAM_Halt ("Invalid FeatureGenerationParams argument")
        
        original_sentences_order_ids = [x["ID"] for x in Sentences]

        import SharedFunctions as SF
        import multiprocessing
        #decide number of parallel processes:
        MAX_CORE_COUNT = multiprocessing.cpu_count()
        if (not isinstance(FeatureGenerationParams.NumberOfParallelProcesses,int)):
            NumberOfThreads = MAX_CORE_COUNT - 1
        elif FeatureGenerationParams.NumberOfParallelProcesses == -1:
            NumberOfThreads = MAX_CORE_COUNT - 1
        elif FeatureGenerationParams.NumberOfParallelProcesses in range(1,MAX_CORE_COUNT): # range(1,4) --> 1,2,3 
            NumberOfThreads = FeatureGenerationParams.NumberOfParallelProcesses 
        else:
            NumberOfThreads = MAX_CORE_COUNT - 1
        
        #important ...         
        if len (Sentences) < NumberOfThreads:
            NumberOfThreads = len (Sentences) 
        
        #print info:         
        MSG = ["Top-K Shortest Path Algorithm." , "-"*30]
        if SetName:
            MSG.append ("SET NAME                    : " + str(SetName)) 
        MSG.extend (["NUMBER OF SENTENCES         : " + str(len(Sentences)) , \
                     "NUMBER OF PARALLEL PROCESSES: " + str(NumberOfThreads), "-"*30, \
                     "FEATURE GENERATION PARAMS:","-"*30,str(FeatureGenerationParams),"-"*30])
        self.lp (MSG) 
        #for retrieving return values ...        
        manager = multiprocessing.Manager()
        return_Dict = manager.dict()

        #partitioning sentences ...         
        SubSentences = SF.PartitionIntoEXACTLYNPartitions (self, Sentences,NumberOfThreads)
        
        my_all_threads = [] 
        for i in range(NumberOfThreads):
            p = multiprocessing.Process (target=self.__lowlevel_CalculateAndAddTopKSDPs, args=(SubSentences[i],FeatureGenerationParams,i,return_Dict))
            my_all_threads.append (p)
            p.start()
        
        #force NOT TO quite this function until all jobs are done!
        for p in my_all_threads:
            p.join()
        
        #gather results in a nice way and return ... 
        result_sentences = [] 
        for i in range(NumberOfThreads):
            result_sentences.extend (return_Dict[i])
        assert len(Sentences) == len (result_sentences)
        
        """
        <<<CRITICAL>>>
        Because path-finding algorithm runs in parallel, everytime the path finding is run, the order of sentences
        will be different from the original order of the sentences. 
        
        Consequently, the results (train/prediction loss or f-score) WILL NOT be reproducable!
        i.e., WILL NOT be replicatable. 
        TO AVOID THIS, HERE I ALWAYS sort the sentences based on their original order in the XML files, to be able to get reproducable results!
        
        VERY IMPORTANT (1): Set the shuffle to False in keras fit function if you do want to be able to reprlicate the results later!
        VERY IMPORTANT (2): BY DEFAULT, TSP_TANN_RE_PARAMS.KERAS_shuffle_training_examples IS SET TO TRUE, because we do not want to optimize on a particular order of sentences used for training. 

        The following line sorts the sentences according to their original order: 
        """
        result_sentences = sorted(result_sentences, key = lambda x: original_sentences_order_ids.index(x["ID"]))
        return result_sentences

    def __lowlevel_CalculateAndAddTopKSDPs (self, Sentences, FeatureGenerationParams, ProcessNumberID, Return_Dict):
        for sentence in Sentences:
            G = self.__lowlevel_MakeDependencyParseGraphForOneSentence (sentence, FeatureGenerationParams)
            
            if FeatureGenerationParams.UsePathLengthCutOffHeuristic:
                token_index = {}
                sentence_length = len(sentence["TOKENS_INFO"])
                for index, token_info in enumerate(sorted (sentence["TOKENS_INFO"] , key = lambda x: x[3][0])): #sort based on token's begin offset
                    token_index[token_info[0]] = index + 1 
            else:
                token_index = sentence_length = None 
                
            for pair in sentence["PAIRS"]:
                SDP_DIRECTION = FeatureGenerationParams.SDPDirection

                #What Policy should be used to generate SDP:
                if SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_e1value_to_e2value:
                    E1_ID , E2_ID = pair["E1"] , pair["E2"]
                
                elif SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_e2value_to_e1value:
                    E1_ID , E2_ID = pair["E2"] , pair["E1"]
                
                elif SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_1stOccurring_to_2nd: 
                    E1_ID , E2_ID = pair["E1"] , pair["E2"]
                    E1_H_bgn = sentence["ENTITIES"][E1_ID]["HEADOFFSET"][0];
                    E2_H_bgn = sentence["ENTITIES"][E2_ID]["HEADOFFSET"][0];
                    if E1_H_bgn > E2_H_bgn: 
                        E1_ID , E2_ID = E2_ID , E1_ID
                
                elif SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_2ndOccurring_to_1st:
                    E1_ID , E2_ID = pair["E1"] , pair["E2"]
                    E1_H_bgn = sentence["ENTITIES"][E1_ID]["HEADOFFSET"][0];
                    E2_H_bgn = sentence["ENTITIES"][E2_ID]["HEADOFFSET"][0];
                    if E1_H_bgn < E2_H_bgn: 
                        E1_ID , E2_ID = E2_ID , E1_ID
        
                else:
                    self.PROGRAM_Halt ("SDP_DIRECTION METHOD : " + SDP_DIRECTION + " IS NOT IMPLEMENTED YET!");
                
                #Now that we have found what should be start and end point for SDP traversal, 
                #because E1 and E2 can be multi-token entities, we always use their corresponding head_tokens. 
                #IMPORTANT: If an entity is single token, then Ei==head_token(Ei), so this will not make any difference for single token entities. 
                E1_Head_Token_ID = self.__lowlevel_GetHeadTokenID (sentence,E1_ID)
                E2_Head_Token_ID = self.__lowlevel_GetHeadTokenID (sentence,E2_ID)
                
                #If headtokens are the same, we should solve this problem, because now path_length ==0 !
                if E1_Head_Token_ID == E2_Head_Token_ID:
                    #If head_tokens are the same, AAAAANNNNNDDDDD ALSO, E1_ID == E2_ID we can do nothing! 
                    if E1_ID == E2_ID: 
                        self.lp ("[WARNING][DISCARDING EXAMPLE !!!][PATH_DETECTION]: head(E1)==head(E2) AND E1==E2. " + str(pair["E1"]) + " , " + str(pair["E2"])) 
                        pair["TOPKP"] = None
                        continue 

                    else: #E1 <> E2, so we try to use E1 and E2 THEMSELVES instead of their Head_Tokens
                        
                        if SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_e1value_to_e2value:
                            E1_ID , E2_ID = pair["E1"] , pair["E2"]

                        elif SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_e2value_to_e1value:
                            E1_ID , E2_ID = pair["E2"] , pair["E1"]
                        
                        elif SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_1stOccurring_to_2nd:
                            E1_ID , E2_ID = pair["E1"] , pair["E2"]
                            E1_bgnOffsetOrig = sentence["ENTITIES"][E1_ID]["ORIGOFFSETS"][0][0]
                            E2_bgnOffsetOrig = sentence["ENTITIES"][E2_ID]["ORIGOFFSETS"][0][0]
                            if E1_bgnOffsetOrig > E2_bgnOffsetOrig: 
                                E1_ID , E2_ID = E2_ID , E1_ID
                        
                        elif SDP_DIRECTION == TSDP_TSVM_SDPFeatureGenerationDirection.from_2ndOccurring_to_1st:
                            E1_ID , E2_ID = pair["E1"] , pair["E2"]
                            E1_bgnOffsetOrig = sentence["ENTITIES"][E1_ID]["ORIGOFFSETS"][0][0]
                            E2_bgnOffsetOrig = sentence["ENTITIES"][E2_ID]["ORIGOFFSETS"][0][0]
                            if E1_bgnOffsetOrig < E2_bgnOffsetOrig: 
                                E1_ID , E2_ID = E2_ID , E1_ID
                
                        else:
                            self.PROGRAM_Halt ("SDP_DIRECTION METHOD : " + SDP_DIRECTION + " IS NOT IMPLEMENTED YET!");

                        E1_Head_Token_ID = self.__lowlevel_GetTokenID (sentence,E1_ID) #TokenID not HeadTokenID
                        E2_Head_Token_ID = self.__lowlevel_GetTokenID (sentence,E2_ID) #TokenID not HeadTokenID 
                        
                        if (E1_Head_Token_ID == E2_Head_Token_ID) or (E1_Head_Token_ID == None) or (E2_Head_Token_ID == None):
                            #compare to None is needed cause sometimes annotated entity is not in the tokens, like "aaa bbb" ,  origoff --> "aaa"
                            self.lp ("[WARNING][PATH_DETECTION]: head(E1)==head(E2) AND E1<>E2. RESOLUTION DIDN'T WORK ! --> " + str(pair["E1"]) + " , " + str(pair["E2"])) 
                            pair["TOPKP"] = None
                            continue 
                        else:
                            self.lp ("[INFO][PATH_DETECTION]: head(E1)==head(E2) AND E1<>E2. RESOLUTION OKAY. --> " + str(pair["E1"]) + " , " + str(pair["E2"])) 
                        
                if FeatureGenerationParams.UsePathLengthCutOffHeuristic:
                    path_adj_len   = abs(token_index[E1_Head_Token_ID] - token_index[E2_Head_Token_ID])
                    #cutoff = path_adj_len + int(round(path_adj_len/4.0))
                    #if cutoff > sentence_length:
                    #    cutoff = path_adj_len 
                    #<<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>>
                    #<<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>>
                    #<<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>>
                    #<<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>><<<CRITICAL>>>
                    #cutoff = min (path_adj_len + 1 , FeatureGenerationParams.PathLengthCutOffHeuristicMAXLEN)
                    cutoff = FeatureGenerationParams.PathLengthCutOffHeuristicMAXLEN
                else:
                    cutoff = None 

                #print "process#"+str(ProcessNumberID), pair["ID"], sentence_length , path_adj_len, cutoff 
                    
                #TopKSDPs,TopKSDPsNodesAttr = GraphUtils.get_all_topK_paths_withweights_RAMFRIENDLY (G, E1_Head_Token_ID, E2_Head_Token_ID , edge_weight_name="w", K=FeatureGenerationParams.TopKShortestPathNumber, path_length_cutoff=cutoff) 
                TopKSDPs,TopKSDPsNodesAttr = GraphUtils2._all_TopKPaths_Info (G, E1_Head_Token_ID, E2_Head_Token_ID , KTop=FeatureGenerationParams.TopKShortestPathNumber, cutoff=cutoff)                     
                if TopKSDPs == []:
                    self.lp ("[WARNING][PATH_DETECTION]: NO PATHS FOUND. --> " + str(pair["E1"]) + "," + str(pair["E2"])) 
                    pair["TOPKP"] = None
                    print "process#"+str(ProcessNumberID), pair["ID"], sentence_length , path_adj_len, cutoff , "FOUND PATHS: 0"
                else:
                    pair["TOPKP"] = [TopKSDPs,TopKSDPsNodesAttr]
                    print "process#"+str(ProcessNumberID), pair["ID"], sentence_length , path_adj_len, cutoff , "FOUND PATHS:" , len (TopKSDPs)
                    

        #return Sentences
        self.lp ("process#"+str(ProcessNumberID)+ " finished successfully.")
        Return_Dict[ProcessNumberID] = Sentences 

    def __lowlevel_GetHeadTokenID (self,sentence,E_ID):
        #Function to get HeadTokenID
        H_bgn = sentence["ENTITIES"][E_ID]["HEADOFFSET"][0];
        H_end = sentence["ENTITIES"][E_ID]["HEADOFFSET"][1];
        for token_info in sentence["TOKENS_INFO"]:
            token_bgn = token_info[-1][0];
            token_end = token_info[-1][1];
            if (token_bgn >= H_bgn) and (token_end <= H_end):
                return token_info[0];
        return None

    def __lowlevel_GetTokenID (self,sentence,E_ID):
        H_bgn = sentence["ENTITIES"][E_ID]["ORIGOFFSETS"][0][0] #first token in multi token entities. Ex: [[12,15],[18,24],[44,49]] --> 12
        H_end = sentence["ENTITIES"][E_ID]["ORIGOFFSETS"][0][1] #first token in multi token entities. Ex: [[12,15],[18,24],[44,49]] --> 15
        for token_info in sentence["TOKENS_INFO"]:
            token_bgn = token_info[-1][0];
            token_end = token_info[-1][1];
            if (token_bgn >= H_bgn) and (token_end <= H_end):
                return token_info[0];
        return None
        
        
    def __lowlevel_MakeDependencyParseGraphForOneSentence (self, sentence, FeatureGenerationParams):
        G = nx.MultiGraph () ; 
        for token_info in sentence["TOKENS_INFO"]:
            G.add_node (token_info[0] , TXT = token_info[1] , POS = token_info[2]);
        
        for edge_info in sentence["PARSE_EDGES"]:
            #mtp: main_type (p:parse/w:word_adjajency) b:begin_node_id e=end_node_id, tp=type
            G.add_edge (edge_info[1],edge_info[2], mtp="p" , b=edge_info[1], e=edge_info[2], tp=edge_info[3], w=FeatureGenerationParams.EdgeWeight_ParseDependency);
        
        #add word adjajency edges if requested
        if isinstance(FeatureGenerationParams.EdgeWeight_WordAdjacency, int) and (FeatureGenerationParams.EdgeWeight_WordAdjacency > 0):
            #sort tokens based on begin-offset     
            sorted_tokens = sorted(sentence["TOKENS_INFO"], key=lambda x: x[3][0])
            for i in xrange(len(sorted_tokens)-1):
                this_token_id = sorted_tokens[i][0]
                next_token_id = sorted_tokens[i+1][0]
                G.add_edge (this_token_id,next_token_id, mtp="w", w=FeatureGenerationParams.EdgeWeight_WordAdjacency)
        return G ; 
   
    def ExtractFeatures (self,Sentences, FeatureGenerationParams, AddFeatureToDictIfNotExists,useOnlyXTopPathsForFeatureGeneration=None):
        import SharedFunctions as SF 
        if useOnlyXTopPathsForFeatureGeneration is not None:
            if not isinstance (useOnlyXTopPathsForFeatureGeneration,int):
                self.PROGRAM_Halt ("useOnlyXTopPathsForFeatureGeneration should be either None or integer and >0")
            elif useOnlyXTopPathsForFeatureGeneration < 1:
                self.PROGRAM_Halt ("useOnlyXTopPathsForFeatureGeneration should be either None or integer and >0")

        allowed_to_add = AddFeatureToDictIfNotExists
        all_mapped_words = set()
        all_not_mapped_words = set()
        
        for s in Sentences:
           for pair in s["PAIRS"]:
              if not pair.has_key("TOPKP"):
                  continue
              if pair["TOPKP"] == None:
                  continue 
              
              all_paths, nodes_attr = pair["TOPKP"] 
              all_paths_features = set() 

              if useOnlyXTopPathsForFeatureGeneration is not None:
                  K = useOnlyXTopPathsForFeatureGeneration
                  tmp_paths = all_paths[0:K]
                  max_weight = tmp_paths[-1][0]
                  # if weights = [2,3,5,5,6,7,11,15,34] and K = 3 ---> [2,3,5,5] because 
                  for X in all_paths[K:]:
                      this_weight = X[0]
                      if this_weight == max_weight:
                          tmp_paths.append(X)
                      else:
                          break ; 
                  all_paths = tmp_paths
                  
              for path_info in all_paths:
                  path_weight , path_edges = path_info
                  this_path_features = []
                  whole_path = [] 
                  for count in range(len(path_edges)):
                      #example: 
                      #    ('bt_16', 'bt_17', {'mtp': 'p', 'b': 'bt_17', 'e': 'bt_16', 'tp': 'nn', 'w': 1}), 
                      #    ('bt_17', 'bt_18', {'mtp': 'p', 'b': 'bt_17', 'e': 'bt_18', 'tp': 'prep', 'w': 1}), 
                      #    ('bt_18', 'bt_20', {'mtp': 'p', 'b': 'bt_18', 'e': 'bt_20', 'tp': 'pobj', 'w': 1})] 
                      # note that in the first edge, the head_token_id is actually bt_16, while edge direction is from bt_17 to bt_16
                      # path_edges[count][0] --> node1id
                      # path_edges[count][1] --> node2id
                      # path_edges[count][2] --> edge_info
                      this_node = nodes_attr[path_edges[count][0]]
                      this_node["id"] = path_edges[count][0] 
                      whole_path.extend ([this_node, path_edges[count]])
                  #add last node_info to the whole_path    
                  this_node = nodes_attr[path_edges[count][1]]
                  this_node["id"] = path_edges[count][1]
                  whole_path.append(this_node)

                  if FeatureGenerationParams.WholePathEncoding_edges_words  == True:
                      this_path_features.append (self.FeatureMappingDictionary.get_add (
                          self.__lowleve_encode_one_path (whole_path,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.word),allowed_to_add))
               
                  if FeatureGenerationParams.WholePathEncoding_edges_lemmas == True:
                      this_path_features.append (self.FeatureMappingDictionary.get_add (
                          self.__lowleve_encode_one_path (whole_path,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.lemma),allowed_to_add))

                  if FeatureGenerationParams.WholePathEncoding_edges_POSTags == True:
                      this_path_features.append (self.FeatureMappingDictionary.get_add (
                          self.__lowleve_encode_one_path (whole_path,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.pos),allowed_to_add))    
                  
                  if FeatureGenerationParams.WholePathEncoding_edges_None == True:
                      #<<<CRITICAL>>>
                      this_path_features.append (self.FeatureMappingDictionary.get_add (
                          self.__lowleve_encode_one_path (whole_path,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.none),allowed_to_add))
                      
                      this_path_features.append (self.FeatureMappingDictionary.get_add (
                          self.__lowleve_encode_one_path (path_edges,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.none),allowed_to_add))
                      
                  for i in FeatureGenerationParams.PathNGrams_words:
                      for r in self.__lowleve_find_ngrams(whole_path,i):
                          this_path_features.append (self.FeatureMappingDictionary.get_add (
                              self.__lowleve_encode_one_path (r,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.word),allowed_to_add))

                  for i in FeatureGenerationParams.PathNGrams_lemmas:
                      for r in self.__lowleve_find_ngrams(whole_path,i):
                          this_path_features.append (self.FeatureMappingDictionary.get_add (
                              self.__lowleve_encode_one_path (r,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.lemmas),allowed_to_add))
                  
                  for i in FeatureGenerationParams.PathNGrams_POSTags:
                      for r in self.__lowleve_find_ngrams(whole_path,i):
                          this_path_features.append (self.FeatureMappingDictionary.get_add (
                              self.__lowleve_encode_one_path (r,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.pos),allowed_to_add))

                  #<CRITICAL> this should receive only edges , so we pass path_edges not whole_path!!!
                  for i in FeatureGenerationParams.PathNGrams_ConsecutiveEdges:
                      for r in self.__lowleve_find_ngrams(path_edges,i):
                          this_path_features.append (self.FeatureMappingDictionary.get_add (
                              self.__lowleve_encode_one_path (r,FeatureGenerationParams,TSDP_TSVM_WhatToUseToCodeNode.none),allowed_to_add))
                          
                  #print "-"*80, "\n", pair["ID"],"WEIGHT:",path_weight,"\n",sorted(set(this_path_features)-set([None])),"\n"
                  all_paths_features = all_paths_features.union(set(this_path_features))
              
              pair["TOPKP_Features"] = sorted(all_paths_features - set([None]))
              print "feature extraction: " + SF.NVLR(pair["ID"],30) + "used paths: " + SF.NVLL(str(len(all_paths)),3) + "   extracted features: " + str(len(pair["TOPKP_Features"]))
              mapped_words , not_mapped_words, pair["SP_Features"] = self.__lowlevel_calculate_SP_Features (s, pair, all_paths, nodes_attr,FeatureGenerationParams, AddFeatureToDictIfNotExists)
              all_mapped_words |= mapped_words
              all_not_mapped_words |= not_mapped_words
              
              mapped_words , not_mapped_words, pair["FS_Features"] = self.__lowlevel_calculate_FullSentence_Features (s, pair, FeatureGenerationParams, AddFeatureToDictIfNotExists)
              all_mapped_words |= mapped_words
              all_not_mapped_words |= not_mapped_words
              
        
        MSG = ["-"*30 + " LIST OF MAPPED WORDS FOR ENTITIES: " + "-"*30]
        for mp in sorted(all_mapped_words):
            MSG.append (mp)
        MSG.append ("-"*30)
        self.lp (MSG)

        MSG = ["[WARNING]:" , "-"*30 + " LIST OF UN-MAPPED WORDS FOR ENTITIES: " + "-"*30]
        for mp in sorted(all_not_mapped_words):
            MSG.append (mp)
        MSG.append ("-"*30)
        self.lp (MSG)
        
    def __lowleve_find_ngrams(self,input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def __lowleve_encode_one_path (self,path,FeatureGenerationParams,WhatToCodeForNode):
        #WhatToCodeForNode should TSDP_TSVM_WhatToUseToCodeNode.X
        
        #FeatureGenerationParams.ReplaceVectorForEntityTypeIfTokenNotFound 
        #FeatureGenerationParams.TOPKP_DirectionalFeatureGeneration 
        #FeatureGenerationParams.TOPKP_UseWord2VecForWords 
        #FeatureGenerationParams.TOPKP_LowerAllTokens 

        if WhatToCodeForNode == TSDP_TSVM_WhatToUseToCodeNode.word:
            ts,te = "_w[" , "]w"
        elif WhatToCodeForNode == TSDP_TSVM_WhatToUseToCodeNode.lemma:
            ts,te = "_l[" , "]l"
        elif WhatToCodeForNode == TSDP_TSVM_WhatToUseToCodeNode.pos:
            ts,te = "_(" , ")"

        if type(path[0]) == dict: #first item is node
            s= "W_"
        else:
            s= "E_"
            
        d = "" 
        t = ""
        for pi in path:
            if type(pi) == dict: #this is a word
                if WhatToCodeForNode == TSDP_TSVM_WhatToUseToCodeNode.none:
                    continue

                elif WhatToCodeForNode == TSDP_TSVM_WhatToUseToCodeNode.pos:
                    t+=  ts + pi["POS"] + te
                
                elif WhatToCodeForNode == TSDP_TSVM_WhatToUseToCodeNode.word:
                    if FeatureGenerationParams.TOPKP_LowerAllTokens:
                        t+= ts + pi["TXT"].lower() + te
                    else:
                        t+= ts + pi["TXT"] + te
            
            else: #this is an edge
                if pi[2]["mtp"]=="w": #word_adjacency_edge
                    d+= "n" #this is a non-directional edge 
                    t+= "_<_WAE_>"
                else: #this is a parse edge
                    if pi[0] == pi[2]["b"]:
                        d+= "f"
                    else:
                        d+= "b" 
                    t+= "_<" + pi[2]["tp"] + ">" 
        return s + d + t

    def __lowlevel_calculate_SP_Features (self, sentence, pair, all_paths, nodes_attr, FeatureGenerationParams, AddFeatureToDictIfNotExists):
        allowed_to_add = AddFeatureToDictIfNotExists
        shortest_path_weight = all_paths[0][0]
        #all_shortest_paths = [path[1] for path in all_paths if path[0]==shortest_path_weight]
        final_results = [] 
        mapped_words = set() 
        not_mapped_words = set() 
        
        for path_weight, path in all_paths:
            #0-build tokens first             
            tokens = [i[0] for i in path] + [path[-1][1]] 
            
            #import pdb ; pdb.set_trace()
            #1-resolve postages along the path
            postags = [self.POSTagsEmbeddings.get_add ( nodes_attr[token_id]["POS"] , allowed_to_add) for token_id in tokens]
            
            #2-resolve dependency types along the path
            dptypes = []
            for edge_bgn , edge_end, edge_info in path:
                if edge_info["mtp"]=="w": #this is a word-adjacency edge, with no direction
                    dptypes.append (self.DPTypesEmbeddings.get_add ("_<WAE>_" , allowed_to_add))
                else: #this is a parse-edge
                    if not FeatureGenerationParams.SP_DirectionalFeatureGeneration:
                        dptypes.append (self.DPTypesEmbeddings.get_add (edge_info["tp"] , allowed_to_add))
                    else:
                        if edge_bgn == edge_info["b"]:
                            dptypes.append (self.DPTypesEmbeddings.get_add (edge_info["tp"] + "_FORWARD" , allowed_to_add))
                        else:
                            dptypes.append (self.DPTypesEmbeddings.get_add (edge_info["tp"] + "_BACKWARD", allowed_to_add))
                            
            #3-resolve E1 and E2 types along the path
            head_token_id_e1 = self.__lowlevel_GetHeadTokenID (sentence,pair["E1"])
            head_token_id_e2 = self.__lowlevel_GetHeadTokenID (sentence,pair["E2"])

            if head_token_id_e1 <> head_token_id_e2: # the generated path is created between head_tokenssince there has been no conflicts
                if (tokens[0] == head_token_id_e1) and (tokens[-1] == head_token_id_e2):
                    E1_TP = sentence["ENTITIES"][pair["E1"]]["TYPE"] 
                    E2_TP = sentence["ENTITIES"][pair["E2"]]["TYPE"] 
                elif (tokens[0] == head_token_id_e2) and (tokens[-1] == head_token_id_e1):
                    E1_TP = sentence["ENTITIES"][pair["E2"]]["TYPE"] 
                    E2_TP = sentence["ENTITIES"][pair["E1"]]["TYPE"] 
                else:
                    self.PROGRAM_Halt ("Could not determine which entity is located in the begining/end of the path! (info:heads are different.)")
            else:
                token_id_e1 = self.__lowlevel_GetTokenID (sentence,pair["E1"])
                token_id_e2 = self.__lowlevel_GetTokenID (sentence,pair["E2"])
                if (tokens[0] == token_id_e1) and (tokens[-1] == token_id_e2):
                    E1_TP = sentence["ENTITIES"][pair["E1"]]["TYPE"] 
                    E2_TP = sentence["ENTITIES"][pair["E2"]]["TYPE"] 
                elif (tokens[0] == token_id_e2) and (tokens[-1] == token_id_e1):
                    E1_TP = sentence["ENTITIES"][pair["E2"]]["TYPE"] 
                    E2_TP = sentence["ENTITIES"][pair["E1"]]["TYPE"] 
                else:
                    self.PROGRAM_Halt ("Could not determine which entity is located in the begining/end of the path! (info:heads are the same.)")
               
            #4-resolve words along the path
            words = []
            for token_index, token_id in enumerate(tokens):
                if FeatureGenerationParams.SP_LowerAllTokens:
                    word = nodes_attr[token_id]["TXT"].lower()
                else:
                    word = nodes_attr[token_id]["TXT"] 
                    
                if token_index in [0,len(tokens)-1]:#special-treatment for the two entities (first/last tokens in the path sequence)
                    E_TP = E1_TP if token_index==0 else E2_TP
                    idx = self.wv.get_word_index (word)
                    if idx <> self.wv.NOT_FOUND:
                        words.append (idx)
                    else:
                        if E_TP in FeatureGenerationParams.ReplaceVectorForEntityTypeIfTokenNotFound:
                            new_word = FeatureGenerationParams.ReplaceVectorForEntityTypeIfTokenNotFound[E_TP]
                            idx = self.wv.get_word_index (new_word)
                            words.append (idx)
                            if idx <> self.wv.NOT_FOUND:
                                mapped_words.add ("ENTITY TYPE: " + E_TP + "\t" + word + " > " + new_word)
                            else:
                                not_mapped_words.add ("ENTITY TYPE: " + E_TP + "\t" + word)
                        else:
                            words.append (idx)
                            not_mapped_words.add ("ENTITY TYPE: " + E_TP + "\t" + word)
                else: #tokens in the middle 
                    words.append (self.wv.get_word_index (word))
            
            #5-calculate normalized path weight:
            normalized_path_weight = np.float32(shortest_path_weight/float(path_weight))
            
            #6-add results to final results
            res = (path_weight,normalized_path_weight,words,postags,dptypes)                    
            final_results.append (res)
        
        return mapped_words , not_mapped_words , final_results    
        
    def __lowlevel_calculate_FullSentence_Features (self, sentence, pair, FeatureGenerationParams, AddFeatureToDictIfNotExists):
        allowed_to_add = AddFeatureToDictIfNotExists
        mapped_words , not_mapped_words = set() , set()
        words , postags , p1 , p2 , chunks = [],[],[],[],[]

        #1-Dictionary for chunk mapping ...         
        CHUNK_MAPPING = {"before" : 1,
                         "middle" : 2,
                         "after"  : 3}
        
        for _entity_tp in self.Configs["OneHotEncodingForValidEnityTypesForRelations"]:
            #Note: Project.Configs["OneHotEncodingForValidEnityTypesForRelations"] --> {u'chemical': 0, u'gene': 1} , hence we add 4 
            CHUNK_MAPPING[_entity_tp] = self.Configs["OneHotEncodingForValidEnityTypesForRelations"][_entity_tp] + 4
        
        #2-Build tokens_info dictionary         
        tokens_info = {x[0]:(x[1],x[2],x[3]) for x in sentence["TOKENS_INFO"]}
        
        #3-resolve E1 and E2 types along the path
        head_token_e1_id = self.__lowlevel_GetHeadTokenID (sentence,pair["E1"])
        head_token_e1_tp = sentence["ENTITIES"][pair["E1"]]["TYPE"] 
        head_token_e1_offset_bgn , head_token_e1_offset_end = tokens_info[head_token_e1_id][2] 
        
        head_token_e2_id = self.__lowlevel_GetHeadTokenID (sentence,pair["E2"])
        head_token_e2_tp = sentence["ENTITIES"][pair["E2"]]["TYPE"] 
        head_token_e2_offset_bgn , head_token_e2_offset_end = tokens_info[head_token_e2_id][2] 
        
        """
        Here, we first check if head_tokens are the same or different. If they are different, we have already used correct head_token ids for finding their begin-offset and end-offset.
        If head_tokens are the same, we are 100% sure that TOPKP is <<<definitely>>> generated based on token_ids <<<themselves>>>. Hence, we use token_ids for finding offsets. 
        Why definitely? because TOPKP is discovered previously! Otherwise, this function wouldn't be called by self.ExtractFeatures
        """        
        if head_token_e1_id == head_token_e2_id: #head tokens are the same! let's use token_ids for finding offsets
            head_token_e1_id = self.__lowlevel_GetTokenID (sentence,pair["E1"]) #we are SURE __lowlevel_GetTokenID (sentence,pair["E1"]) IS NOT returning None, because TOPKP is constructed. 
            head_token_e1_offset_bgn , head_token_e1_offset_end = tokens_info[head_token_e1_id][2] 
            
            head_token_e2_id = self.__lowlevel_GetTokenID (sentence,pair["E2"]) #we are SURE __lowlevel_GetTokenID (sentence,pair["E2"]) IS NOT returning None, because TOPKP is constructed. 
            head_token_e2_offset_bgn , head_token_e2_offset_end = tokens_info[head_token_e2_id][2] 
        
        #4-Find first and second occuring entities
        head_tokens_occuring = [{},{}] # head_tokens_occuring[0] will be the first occuring token AND head_tokens_occuring[1] will be the second occuring token
        if head_token_e1_offset_bgn < head_token_e2_offset_bgn: #sentence : ...E1...E2....
            head_tokens_occuring[0]["id"]  = head_token_e1_id
            head_tokens_occuring[0]["tp"]  = head_token_e1_tp 
            head_tokens_occuring[0]["bgn"] = head_token_e1_offset_bgn
            head_tokens_occuring[0]["end"] = head_token_e1_offset_end
            head_tokens_occuring[1]["id"]  = head_token_e2_id
            head_tokens_occuring[1]["tp"]  = head_token_e2_tp 
            head_tokens_occuring[1]["bgn"] = head_token_e2_offset_bgn
            head_tokens_occuring[1]["end"] = head_token_e2_offset_end
        else: #sentence : ...E2...E1....
            head_tokens_occuring[1]["id"]  = head_token_e1_id
            head_tokens_occuring[1]["tp"]  = head_token_e1_tp 
            head_tokens_occuring[1]["bgn"] = head_token_e1_offset_bgn
            head_tokens_occuring[1]["end"] = head_token_e1_offset_end
            head_tokens_occuring[0]["id"]  = head_token_e2_id
            head_tokens_occuring[0]["tp"]  = head_token_e2_tp 
            head_tokens_occuring[0]["bgn"] = head_token_e2_offset_bgn
            head_tokens_occuring[0]["end"] = head_token_e2_offset_end
            
        #5-build features ...  
        sorted_tokens = sorted (sentence["TOKENS_INFO"] , key = lambda x: x[3][0]) #sort based on begin-offset 
        sorted_tokens_ids = [x[0] for x in sorted_tokens] 
        
        def get_token_position_difference (token_id , head_token_id):
            return sorted_tokens_ids.index(token_id) - sorted_tokens_ids.index(head_token_id)
            
        for token in sorted_tokens:
            token_id , token_word , token_postag , (token_offset_bgn , token_offset_end) = token 
            
            #1-word
            if FeatureGenerationParams.FS_LowerAllTokens:
                token_word = token_word.lower()

            if token_id in [head_token_e1_id,head_token_e2_id]: #This is one of the entities ... treat differently if requested
                if token_id == head_token_e1_id:
                    E_TP = head_token_e1_tp
                else:
                    E_TP = head_token_e2_tp
                
                idx = self.wv.get_word_index (token_word)
                if idx <> self.wv.NOT_FOUND:
                    words.append (idx)
                else:
                    if E_TP in FeatureGenerationParams.ReplaceVectorForEntityTypeIfTokenNotFound:
                        new_word = FeatureGenerationParams.ReplaceVectorForEntityTypeIfTokenNotFound[E_TP]
                        idx = self.wv.get_word_index (new_word)
                        words.append (idx)
                        if idx <> self.wv.NOT_FOUND:
                            mapped_words.add ("ENTITY TYPE: " + E_TP + "\t" + token_word + " > " + new_word)
                        else:
                            not_mapped_words.add ("ENTITY TYPE: " + E_TP + "\t" + token_word)
                    else:
                        words.append (idx)
                        not_mapped_words.add ("ENTITY TYPE: " + E_TP + "\t" + token_word)
            else: #tokens in the middle 
                words.append (self.wv.get_word_index (token_word))
            
            #2-postag
            postags.append (self.POSTagsEmbeddings.get_add (token_postag,allowed_to_add))
            
            #3-chunk
            if token_id in [head_token_e1_id,head_token_e2_id]:
                if token_id == head_token_e1_id:
                    E_TP = head_token_e1_tp
                else:
                    E_TP = head_token_e2_tp
                chunks.append (CHUNK_MAPPING[E_TP])
            else:
                if (token_offset_end <= head_tokens_occuring[0]["bgn"]):
                    chunks.append (CHUNK_MAPPING["before"])
                
                elif (token_offset_bgn >= head_tokens_occuring[0]["end"]) and (token_offset_end <= head_tokens_occuring[1]["bgn"]):
                    chunks.append (CHUNK_MAPPING["middle"])
                
                elif (token_offset_bgn >= head_tokens_occuring[1]["end"]):
                    chunks.append (CHUNK_MAPPING["after"])
            
            #4-p1, p2
            if FeatureGenerationParams.FS_PositionEmbeddingsMeanings == TSDP_TSVM_FS_PositionsMeaning.P1distanceTo1stOccurring_P2distanceTo2ndOccurring:
                p1.append (self.PositionEmbeddings.get_embedding_index(get_token_position_difference (token_id,head_tokens_occuring[0]["id"])))
                p2.append (self.PositionEmbeddings.get_embedding_index(get_token_position_difference (token_id,head_tokens_occuring[1]["id"])))
            
            elif FeatureGenerationParams.FS_PositionEmbeddingsMeanings == TSDP_TSVM_FS_PositionsMeaning.P1distanceToe1value_P2distanceToe2value:
                p1.append (self.PositionEmbeddings.get_embedding_index(get_token_position_difference (token_id,head_token_e1_id)))
                p2.append (self.PositionEmbeddings.get_embedding_index(get_token_position_difference (token_id,head_token_e2_id)))
            
        return mapped_words , not_mapped_words , [words,postags,chunks,p1,p2]
        
                
                    
                    
                    