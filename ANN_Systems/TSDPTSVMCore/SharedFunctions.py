def CalculateHowManyRelationsWithShortestPathInDataset (Sentences):
    POS_NEG_DICT  = {"Positives":0 , "Negatives":0};
    CLASS_TP_DICT = {"NEG":0} ; 
    for sentence in Sentences:
        for pair in sentence["PAIRS"]:
            if (pair.has_key("TOPKP")) and (pair.has_key("TOPKP_Features")):
                if (pair["TOPKP"] <> None) and (pair["TOPKP_Features"] <> None): #an example might have pair["TOPKP_Features"] = [] in a very rare and very very unlikely condition
                    #1-Is any type of relation (=Positive) or not (=Negative)
                    if pair["POSITIVE"]:
                        POS_NEG_DICT["Positives"] += 1;
                    else:
                        POS_NEG_DICT["Negatives"] += 1;
                
                    #2-Is any type of relation (=Positive) or not (=Negative)
                    if pair["CLASS_TP"]==None:
                        CLASS_TP_DICT["NEG"]+=1 ;
                    else:
                        pair_class_tp = pair["CLASS_TP"] ;
                        if CLASS_TP_DICT.has_key (pair_class_tp):
                            CLASS_TP_DICT[pair_class_tp]+=1;
                        else:
                            CLASS_TP_DICT[pair_class_tp]=1;
    Total_Example_CNT = POS_NEG_DICT["Positives"] + POS_NEG_DICT["Negatives"] ; 
    return POS_NEG_DICT , CLASS_TP_DICT , Total_Example_CNT; 

def CalculateHowManyRelationsWithTOPKPaths (Sentences):
    POS_NEG_DICT  = {"Positives":0 , "Negatives":0};
    CLASS_TP_DICT = {"NEG":0} ; 
    for sentence in Sentences:
        for pair in sentence["PAIRS"]:
            if pair.has_key("TOPKP"):
                if (pair["TOPKP"] <> None): 
                    #1-Is any type of relation (=Positive) or not (=Negative)
                    if pair["POSITIVE"]:
                        POS_NEG_DICT["Positives"] += 1;
                    else:
                        POS_NEG_DICT["Negatives"] += 1;
                
                    #2-Is any type of relation (=Positive) or not (=Negative)
                    if pair["CLASS_TP"]==None:
                        CLASS_TP_DICT["NEG"]+=1 ;
                    else:
                        pair_class_tp = pair["CLASS_TP"] ;
                        if CLASS_TP_DICT.has_key (pair_class_tp):
                            CLASS_TP_DICT[pair_class_tp]+=1;
                        else:
                            CLASS_TP_DICT[pair_class_tp]=1;
    Total_Example_CNT = POS_NEG_DICT["Positives"] + POS_NEG_DICT["Negatives"] ; 
    return POS_NEG_DICT , CLASS_TP_DICT , Total_Example_CNT; 


def PartitionIntoEXACTLYNPartitions(self, L, N):
    if N > len(L):
        self.PROGRAM_Halt ("Error in LIST_PartitionIntoNPartitions: N should be in [1,len(L)]")
    RES = [] ;
    for i in range (0, N):
        RES.append ([]) ;
    for i in range(0,len(L)):
        RES[i%N].append (L[i]) ;
    return RES ; 

def NVLR (S, Cnt):
    if S == None:
        return " " * Cnt ;
    if len(S) < Cnt:
        return str(S) + (" " * (Cnt - len(S)));
    else:
        return S[0:Cnt]

def NVLL (S, Cnt):
    if S == None:
        return " " * Cnt ;
    if len(S) < Cnt:
        return (" " * (Cnt - len(S))) + str(S) ;
    else:
        return S[0:Cnt]

"""
def CalculateNumberOfExamplesPerSentence (self, Sentences):
    TOTAL_POSITIVES = 0 ;
    TOTAL_NEGATIVES = 0 ; 
    for S in Sentences:
        NumberOfExamples = {"negatives":0} ; 
        if (len(S["ENTITIES"])==0) or (len(S["PAIRS"])==0): #Second condition for those sentences that have one entity but self interaction which are apparently discarded in ANTTI work ...
            S["NumberOfExamples"] = NumberOfExamples; 
        else:
            for pair in S["PAIRS"]:
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

def CalculateDocumentWiseStatistics (self, Sentences , PrintStatistics = False):
    docs = {}; 
    for sentence in Sentences:
        if docs.has_key(sentence["DOC_ID"]):
            docs[sentence["DOC_ID"]].append (sentence);
        else:
            docs[sentence["DOC_ID"]] = [sentence];

    docs_info = {}; 
    for doc_id in docs:
        docs_info[doc_id] = {"Interactions":{} , "Sentences":{} , "Sentence_Count":0};
        for sentence in docs[doc_id]:
            HAS_RELATION = False ; 
            for pair in sentence["PAIRS"]:
                if pair["TOPKP"] <> None:
                    HAS_RELATION = True ; 
                    if not sentence["ID"] in docs_info[doc_id]["Sentences"]:
                        docs_info[doc_id]["Sentences"][sentence["ID"]] = {} ; 

                    if self.Configs["ClassificationType"] == "binary":
                        if pair["POSITIVE"]==True:
                            if docs_info[doc_id]["Interactions"].has_key("Positives"):
                                docs_info[doc_id]["Interactions"]["Positives"]+=1;
                            else:
                                docs_info[doc_id]["Interactions"]["Positives"]=1;
                            
                            if docs_info[doc_id]["Sentences"][sentence["ID"]].has_key("Positives"):
                                docs_info[doc_id]["Sentences"][sentence["ID"]]["Positives"]+=1 ;
                            else:
                                docs_info[doc_id]["Sentences"][sentence["ID"]]["Positives"]=1 ;

                        else:
                            if docs_info[doc_id]["Interactions"].has_key("Negatives"):
                                docs_info[doc_id]["Interactions"]["Negatives"]+=1;
                            else:
                                docs_info[doc_id]["Interactions"]["Negatives"]=1;

                            if docs_info[doc_id]["Sentences"][sentence["ID"]].has_key("Negatives"):
                                docs_info[doc_id]["Sentences"][sentence["ID"]]["Negatives"]+=1 ;
                            else:
                                docs_info[doc_id]["Sentences"][sentence["ID"]]["Negatives"]=1 ;

                    else: #multiclass
                        if pair["CLASS_TP"]==None:
                            if docs_info[doc_id]["Interactions"].has_key("Negatives"):
                                docs_info[doc_id]["Interactions"]["Negatives"]+=1;
                            else:
                                docs_info[doc_id]["Interactions"]["Negatives"]=1;
                            
                            if docs_info[doc_id]["Sentences"][sentence["ID"]].has_key("Negatives"):
                                docs_info[doc_id]["Sentences"][sentence["ID"]]["Negatives"]+= 1 ; 
                            else:
                                docs_info[doc_id]["Sentences"][sentence["ID"]]["Negatives"]= 1 ; 
                                
                        else:
                            if docs_info[doc_id]["Interactions"].has_key(pair["CLASS_TP"]):
                                docs_info[doc_id]["Interactions"][pair["CLASS_TP"]]+=1;
                            else:
                                docs_info[doc_id]["Interactions"][pair["CLASS_TP"]]=1;

                            if docs_info[doc_id]["Sentences"][sentence["ID"]].has_key(pair["CLASS_TP"]):
                                docs_info[doc_id]["Sentences"][sentence["ID"]][pair["CLASS_TP"]]+= 1 ; 
                            else:
                                docs_info[doc_id]["Sentences"][sentence["ID"]][pair["CLASS_TP"]]= 1 ; 
            if HAS_RELATION:
                if docs_info[doc_id].has_key("Sentence_Count"):
                    docs_info[doc_id]["Sentence_Count"] += 1 ;
                else:
                    docs_info[doc_id]["Sentence_Count"] = 1 ;
                    
    if PrintStatistics:
        MSG = ["-"*55 , " -------- STATISTICS ABOUT EXAMPLES IN EACH DOCUMENT --------"] ; 
        for d_id in docs_info:
            MSG.append (d_id + ":   Sentences:" + str(docs_info[d_id]["Sentence_Count"]) + "    Total Examples:" + str(sum([docs_info[d_id]["Interactions"][i] for i in docs_info[d_id]["Interactions"]])));
            for s_id in docs_info[d_id]["Sentences"]:
                MSG.append ("\t"+s_id+":  " + str(docs_info[d_id]["Sentences"][s_id])) ; 
            MSG.append ("-");
        self.lp (MSG); 
        
    return docs, docs_info ;                          
"""        