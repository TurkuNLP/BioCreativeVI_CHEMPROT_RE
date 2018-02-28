import sys; 
import shutil ; 
import datetime ; 
import numpy as np;

def OS_MakeDir (FolderAddress):
    if not shutil.os.path.exists(FolderAddress):
        try:        
            shutil.os.makedirs(FolderAddress)
            print "Creating Folder: " , FolderAddress
        except:#for multi-threaded, raises error if already exists!
            pass ;

def OS_RemoveDirectoryWithContent(FolderAddress):
    if shutil.os.path.exists(FolderAddress):
        shutil.rmtree(FolderAddress)

def OS_IsDirectory (FolderAddress):
    return shutil.os.path.isdir (FolderAddress)
    
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

def DATETIME_GetNowStr():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") ;  
    
def FILE_CheckFileExists (fname):
    return shutil.os.path.isfile(fname); 

def GetAllFilesPathAndNameWithExtensionInFolder (FolderAddress, FileExtension, ProcessSubFoldersAlso = True):
    #IMPORTANT ... Extenstion should be like : "txt" , "a2"  ... WITHOUT DOT !    
    FILES = []; 
    if ProcessSubFoldersAlso:    
        for root, dirs, files in shutil.os.walk(FolderAddress):
            for file in files:
                if file.endswith("." + FileExtension):
                    FILES.append(shutil.os.path.join(root, file));
        return (FILES);
    else:         
        for file in shutil.os.listdir(FolderAddress):
            if file.endswith("." + FileExtension): #".txt" ;
                FILES.append(FolderAddress + file);
        return (FILES) ; 

def FDecimalPoints (NUM, N):
    R = "{:."+str(N)+"f}"
    return R.format (NUM) ; 

class CorpusHelperClass:
    def CalculateMaxLengthForAllBags (self, Sentences):
        M1_Before        = [] ; 
        M1_Middle        = [] ; 
        M1_After         = [] ; 
        M2_ForeBetween   = [] ; 
        M2_Between       = [] ;
        M2_BetweenAfter  = [] ; 
    
        for S in Sentences:
            for pair in S["PAIRS"]:
                if pair.has_key("BAGS"):
                     M1_Before.append   (len(pair["BAGS"]["M1_Before"][0])); 
                     M1_Middle.append   (len(pair["BAGS"]["M1_Middle"][0])); 
                     M1_After.append    (len(pair["BAGS"]["M1_After"][0]));      
                     M2_ForeBetween.append  (len(pair["BAGS"]["M2_ForeBetween"][0]));
                     M2_Between.append       (len(pair["BAGS"]["M2_Between"][0]));
                     M2_BetweenAfter.append (len(pair["BAGS"]["M2_BetweenAfter"][0]));  
        return np.max (M1_Before) , np.max (M1_Middle) , np.max (M1_After) , np.max (M2_ForeBetween) , np.max (M2_Between) , np.max (M2_BetweenAfter); 

    def HowManyRelationsWithShortestPathInDataset (self, Sentences):
        POS_NEG_DICT  = {"Positives":0 , "Negatives":0};
        CLASS_TP_DICT = {"NEG":0} ; 
        for sentence in Sentences:
            for pair in sentence["PAIRS"]:
                if (pair.has_key("BAGS")):
                    if (pair["SHORTEST_PATH_F"] <> None):
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

    def ReturnMAXSDPLength (self, Sentences):
        MAX_LEN = -1 ; 
        for sentence in Sentences:
            for pair in sentence["PAIRS"]:
                if (pair.has_key("BAGS")):
                    if (pair["SHORTEST_PATH_F"] <> None):
                        if len(pair["SHORTEST_PATH_F"][0])>MAX_LEN:
                            MAX_LEN = len(pair["SHORTEST_PATH_F"][0]) ; 
        return MAX_LEN ; 
        
        
CorpusHelper = CorpusHelperClass () ; 

