# -*- coding: utf-8 -*-
import os ;
import sys;
import glob;
import pickle ;
import numpy as np ;
import MySQLdb as mdb
import MySQLdb.cursors as cursors
import shutil, errno;
import datetime as dt ;  

def connectToEVEXDB (_DB_ADDR, _UserName, _Password, _DBName , UTF8 = False , SSCursor = False, ShowLog = True):
    try:
        if SSCursor:
            if ShowLog: sys.stdout.write ("Connecting to EVEX DB and using SSCursor ...");
            EVEXDBcon = mdb.connect(_DB_ADDR,_UserName, _Password, _DBName , cursorclass = cursors.SSCursor);
        else:
            if ShowLog: sys.stdout.write ("Connecting to EVEX DB and NOT using SScursor ...");
            EVEXDBcon = mdb.connect(_DB_ADDR,_UserName, _Password, _DBName);
            
        if ShowLog: sys.stdout.write ("Done.\n");

        if UTF8:
            if ShowLog: sys.stdout.write ("Requesting UTF-8 DB Character_set: ...");
            EVEXDBcon.set_character_set("utf8");
            if ShowLog: sys.stdout.write ("Done.\n");
            
        return EVEXDBcon;
    except mdb.Error, e:
        sys.stdout.write  ("\nError %d: %s" % (e.args[0],e.args[1]));
        sys.exit(1)


def CopyFolderOrFile(src, dst):
    try:
        shutil.copytree(src, dst);
        print "COPIED FOLDER FROM: " , src, " TO: " , dst ;
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
            print "COPIED FROM: " , src, " TO: " , dst ;
        else: raise
        
def GetAllFilesPathAndNameWithExtensionInFolder (FolderAddress, FileExtension, ProcessSubFoldersAlso = True):
    #IMPORTANT ... Extenstion should be like : "txt" , "a2"  ... WITHOUT DOT !    
    FILES = []; 
    if ProcessSubFoldersAlso:    
        for root, dirs, files in os.walk(FolderAddress):
            for file in files:
                if file.endswith("." + FileExtension):
                    FILES.append(os.path.join(root, file));
        return (FILES);
    else:         
        for file in os.listdir(FolderAddress):
            if file.endswith("." + FileExtension): #".txt" ;
                FILES.append(FolderAddress + file);
        return (FILES) ; 

def Get_immediate_subdirectories (a_dir, sort=True):
    if a_dir[-1] <> "/":
        a_dir+="/" ; 
    a_dir_pattern= a_dir + "*" ; 
    from glob import glob ; 
    L = [x for x in glob (a_dir_pattern) if os.path.isdir(os.path.join(a_dir, x))];
    if sort: L.sort ();
    return L ; 

def GetCurrentProgramExecutionPath ():
    return os.path.dirname(os.path.realpath(__file__)) ; 

def DeleteAllFileInFolderWithExtention (FolderAddress , FileExtension):
    #Usage: ("/home/farmeh/Desktop/MY_ALL_PROJECTS/EVEX_Build_3/CODE/LOG_FILES/" , "log")
    FileList = [f for f in glob.glob (FolderAddress + "*." + FileExtension)];
    for f in FileList:
        os.remove(f);

def FOUT (f, line , NL = True):
    print line;
    if f == None: return ;     
    if NL:
        line += "\n" ; 
    f.write (line);

def SOUT (S):
    sys.stdout.write (S);
    
def PICKLE_LoadFromFile (FileName , OBJECTName=None):
    if OBJECTName <> None:
        print  ("Loading pickled object:" + OBJECTName + " from file:" + FileName + " ...");
    try:    
        FH = open (FileName , "rb");
        DATA = pickle.load (FH);
        FH.close();
        if OBJECTName <> None:
            print ("Done.\n");
        return DATA ; 
    except Exception as E:
        print  ("\nError loading file." + E.message);
        sys.exit (1);

def PICKLE_SaveToFile (FileName, OBJECT, OBJECTName=None):
    if OBJECTName <> None:
        print ("Pickling object:" + OBJECTName + " into file:" + FileName + " ...");
    try:    
        FH = open (FileName, "wb");
        pickle.dump (OBJECT, FH);
        FH.close ();
        if OBJECTName <> None:
            print  ("Done.\n");
    except Exception as E:
        print  ("\nError:" + E.message);
        sys.exit (1);

def NUMPY_LoadFromFile (FileName , OBJECTName=None):
    if OBJECTName <> None:
        print  ("Loading numpy file:" + OBJECTName + " from file:" + FileName + " ...");
    try:    
        DATA = np.load (FileName ) ;
        if OBJECTName <> None:
            print  ("Done.\n");
        return DATA ; 
    except Exception as E:
        print  ("Error:" + E.message);
        sys.exit (1);
        
def NUMPY_SaveToFile (FileName, OBJECT, OBJECTName=None):
    if OBJECTName <> None:
        print  ("Saving numpy array:" + OBJECTName + " into file:" + FileName);
    try:    
        np.save (FileName, OBJECT);
        if OBJECTName <> None:
            print  (" ... Done.\n");
    except Exception as E:
        print  ("Error:" + E.message);
        sys.exit (1);

def NUMPY_SetSimplePrint (precision = 8):
    np.set_printoptions (precision = precision , suppress = True);

def SERIALIZE_ListIntoTextFile_OneLinePerEachItem (FileName, L, ListName=None):
    if L == None or len(L)<1:
        return; 
    if ListName <> None:
        print ("Serializing List:" + ListName + " into file:" + FileName);
    try:
        FH = open (FileName, "wb");
        for ITEM in L:
            FH.write (ITEM);
            FH.write ("\n");
        FH.close ();
        if ListName <> None:
            print  ("Done.\n");
    except Exception as E:
        print  ("Error:" + E.message);
        sys.exit (1);

def DESERIALIZE_FromTextFileIntoList_OneItemPerEachLine (FileName, ListName=None):
    if FileName == None or len(FileName)<1:
        return; 
    if ListName <> None:
        print ("De-Serializing List:" + ListName + " from file:" + FileName);
    try:
        L = [];        
        FH = open (FileName, "rb");
        for line in FH:
            L.append (line);
        FH.close ();
        if ListName <> None:
            print  ("Done.\n");
        return L ; 
    except Exception as E:
        print  ("Error:" + E.message);
        sys.exit (1);
    
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

def is_ascii(a):
    return a.encode("ascii", "ignore").decode("ascii") == a ;

def is_number(s):
    try:
        if s == None: 
            return False ;
        float(s)
        return True
    except ValueError:
        return False

   
def CheckFolderAndCreateIfnotExist (ParentFolder, Folder):
    if not os.path.exists(ParentFolder + "/" + Folder):
        os.makedirs(ParentFolder + "/" + Folder);
    

def SORT_ListOfTuplesByXthElementInTuple (ListOfTuples, ElementIndex , reverse=False):
    #Example : A = [("ID1" , 1, 4) , ("ID2" , 2, 3) , ("ID3", 3, 2) , ("ID4" , 4,1)] ;
    # Sort_ListOfTupleByXthElement (A,2); 
    try:    
        float (ElementIndex);
    except:
        print "Index is not number" ;
        sys.exit(1);
    ListOfTuples.sort(key=lambda x: x[ElementIndex] , reverse = reverse);

def LIST_PartitionIntoEvenlySizedChunks_RetsGenerator (L, size):
    """ Yield successive n-sized chunks from L."""
    for i in xrange(0, len(L), size):
        yield L[i:i+size];

def LIST_PartitionIntoEXACTLYNPartitions(L, N):
    if N > len(L):
        print "Error in LIST_PartitionIntoNPartitions: N should be in [1,len(L)]" ; 
        sys.exit(-1);
    RES = [] ;
    for i in range (0, N):
        RES.append ([]) ;
    for i in range(0,len(L)):
        RES[i%N].append (L[i]) ;
    return RES ; 

def LIST_FlatListFromListOfLists (L):
    # [[1,2,3],[4],[5,6,7,8]] ==> [1,2,3,4,5,6,7,8]
    return [item for sublist in L for item in sublist];
    
def KDD_GET_ITEM_FROM_CONDENSED_DISTANCE_MATRIX (i, j, TotalNumberOfItems , CondensedDistanceMatrix):
    if i == j:
        return 0 ; 
    if i > j :
        (i,j) = (j,i);
    I_GET_ITEM_FROM_CONDENSED_DISTANCE_MATRIX = lambda i,j,n: (n - np.array(xrange(1,i+1))).sum() + (j - 1 - i);
    return CondensedDistanceMatrix [I_GET_ITEM_FROM_CONDENSED_DISTANCE_MATRIX(i,j,TotalNumberOfItems)] ; 

def FORMAT_FloatNumberWithNDecimalPoints (NUM, N):
    if N <= 0: return NUM ; #Safe    
    from decimal import Decimal ; 
    try:
        L = float(Decimal(NUM).quantize (Decimal(10) ** -N)) ;
        return L;
    except:
        L = str(NUM);
        return [L[0:L.find(".")+N+1]];

def DATETIME_GetTodayDateTime_CompactFormat ():
    import time ;
    return time.strftime('%Y-%m-%d_%H:%M:%S') ;

def DATETIME_GetNowForPrinting ():
   L = dt.datetime.now () ; 
   L_Month  = NVLL (str(L.month), 2).replace (" " , "0") ;
   L_Day    = NVLL (str(L.day), 2).replace (" " , "0") ;
   L_Hour   = NVLL (str(L.hour), 2).replace (" " , "0") ;   
   L_Minute = NVLL (str(L.minute), 2).replace (" " , "0") ;   
   L_Second = NVLL (str(L.second), 2).replace (" " , "0") ;   
   DTNOW = str(L.year) + L_Month + L_Day + "_" + L_Hour + L_Minute + L_Second ; 
   return DTNOW ; 

def KERAS_GetLayerOutput(model, NI, layername, train_mode = False): 
    #function by farrokh for testing outputs ...
    from keras import backend as K
    LearningPhaseValue = 1 if train_mode else 0 # 0 when testing, 1 when training ...
    INPUTTENSORS = model.input if isinstance (model.input , list) else [model.input] ; 
    RetrieveLayerOutput_theanoFunction = K.function(INPUTTENSORS +[K.learning_phase()], [model.get_layer (layername).output]) 
    if not isinstance (NI , list):
        NI = [NI] ; 
    activations =  RetrieveLayerOutput_theanoFunction (NI + [LearningPhaseValue])
    return activations; 
