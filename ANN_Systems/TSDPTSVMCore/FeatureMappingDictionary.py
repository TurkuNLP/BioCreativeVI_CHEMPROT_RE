import SharedFunctions as GF ; 
try:
    import cPickle as pickle 
except:
    import pickle

class TSDP_TSVM_FeatureMappingDictionary:
    def __init__(self, lp , PROGRAM_Halt):
        self.lp = lp ; 
        self.PROGRAM_Halt = PROGRAM_Halt ;
        self.__FeatureMappingDictionary = {}
    
    def load_FeatureMappingDictionary (self,fileaddress):
        if not GF.FILE_CheckFileExists (fileaddress):
            self.PROGRAM_Halt ("Cannot load FeatureMappingDictionary file:"+fileaddress); 
        try:
            fhandle = open(fileaddress,"rb") 
            self.__FeatureMappingDictionary = pickle.load(fhandle)
            fhandle.close()
        except Exception as E:
            self.PROGRAM_Halt ("Error loading FeatureMappingDictionary from file. Error:"+E.message); 
                        
        if not isinstance(self.__FeatureMappingDictionary ,dict):
            self.PROGRAM_Halt ("Error loading FeatureMappingDictionary from file. Error: Not a pickled python dictionary."); 
        
        self.lp(["FeatureMappingDictionary is successfully loaded from file.", "file:"+fileaddress , "number of features:"+str(len(self.__FeatureMappingDictionary))])
    
    
    def save_FeatureMappingDictionary (self,fileaddress):
        try:
            fhandle = open(fileaddress,"wb") 
            pickle.dump(self.__FeatureMappingDictionary, fhandle, protocol=2)
            fhandle.close()
        except Exception as E:
            self.PROGRAM_Halt ("Error saving FeatureMappingDictionary into file. Error:"+E.message); 
                        
        self.lp(["FeatureMappingDictionary is successfully save into file.", "file:"+fileaddress , "number of features:"+str(len(self.__FeatureMappingDictionary))])
    
    def get_add (self,feature_name,allowed_to_add):
        if feature_name in self.__FeatureMappingDictionary:
            return self.__FeatureMappingDictionary[feature_name]
        else:
            if allowed_to_add:
                index = len(self.__FeatureMappingDictionary); 
                self.__FeatureMappingDictionary[feature_name] = index
                return index
            else:
                return None 
            
    def reset_dictionary(self):
        self.lp ("[INFO]: resetting FeatureMappingDictionary!")
        self.__FeatureMappingDictionary = {}
    
    def return_dictionary_length(self):
        return len(self.__FeatureMappingDictionary)
        
class EmbeddingMappingDictionary:
    #index 0 is reserved for masking ...
    #index 1 is reserved for items not found in the dictionary
    def __init__(self, lp , PROGRAM_Halt, Name):
        self.lp = lp 
        self.PROGRAM_Halt = PROGRAM_Halt 
        self.__Name = Name
        self.__d = {None: 1} 

    def get_add (self,key,allowed_to_add):
        if key in self.__d:
            return self.__d[key]
        else:
            if allowed_to_add:
                value = len(self.__d)+1 
                self.__d[key] = value
                return value
            else:
                return self.__d[None]

    def len(self):
        return len(self.__d)

    def reset_dictionary(self):
        self.lp ("[INFO]: resetting EmbeddingMappingDictionary: "+ self.__Name)
        self.__d = {None: 1} 

class Word2VecEmbedding:
    #index 0 is reserved for masking ...
    #index 1 is reserved for items not found in the dictionary
    def __init__(self, lp , PROGRAM_Halt, wv_fileaddress, max_rank):
        self.NOT_FOUND = 1
        self.lp = lp
        self.PROGRAM_Halt = PROGRAM_Halt
        
        import numpy         
        try:
            from wvlib import wvlib
        except:
            import wvlib
        try:
            self.lp ("Loading word-embeddings matrix from file:" + wv_fileaddress + "\nmax_rank: "+str(max_rank))
            self.__w2v_model = wvlib.load (wv_fileaddress , max_rank=max_rank)            
        except Exception as E:
            self.PROGRAM_Halt ("Error loading word2vec mode.\nError:"+E.message)
        
        self.lp ("Normalizing word-embeddings matrix.")
        self.__w2v_model.normalize()
        self.__words = [u'_<_MASK__>_', u'_<_OOV_>_'] + [w.decode('utf-8') for w in self.__w2v_model.words()]
        #use if error: self.__words = [u'_<_MASK__>_', u'_<_OOV_>_'] + [w.decode('utf-8','ignore') for w in self.__w2v_model.words()]
        self.__word_index_dict = {w:i for i, w in enumerate(self.__words)}
        latent_dim = self.__w2v_model._vectors.vectors[0].shape[0]
        self.__weights = numpy.concatenate([numpy.zeros((1,latent_dim)), numpy.random.randn(1,latent_dim), numpy.asarray(self.__w2v_model.vectors())])

    def get_word_index (self,word,allow_search_lowercase_if_not_found=True):
        if word in self.__word_index_dict:
            return self.__word_index_dict[word]
        else:
            if allow_search_lowercase_if_not_found:
                word = word.lower()
                if word in self.__word_index_dict:
                    return self.__word_index_dict[word]
                else:
                    return self.__word_index_dict[u'_<_OOV_>_']
            else:
                return self.__word_index_dict[u'_<_OOV_>_']
                
    def get_normalized_weights(self,deepcopy):
        import copy
        if deepcopy:
            return copy.deepcopy (self.__weights)
        else:
            return self.__weights
    
    def shape(self):
        return (len(self.__words), self.__w2v_model._vectors.vectors[0].shape[0])

class PositionEmbeddings:
    def __init__(self, lp , PROGRAM_Halt):
        self.lp = lp 
        self.PROGRAM_Halt = PROGRAM_Halt 
        self.__PositionEmbeddingMappingMatrix = [\
                                        [1,1,1,1,1,1,1,1,1,1,1], #0  Out of vocabulary""" \  
                                        [0,1,1,1,1,1,1,1,1,1,0], #1  -31-inf 
                                        [0,0,1,1,1,1,1,1,1,1,0], #2  -21-30
                                        [0,0,0,1,1,1,1,1,1,1,0], #3  -11-20
                                        [0,0,0,0,1,1,1,1,1,1,0], #4  -6-10
                                        [0,0,0,0,0,1,1,1,1,1,0], #5  -5
                                        [0,0,0,0,0,0,1,1,1,1,0], #6  -4
                                        [0,0,0,0,0,0,0,1,1,1,0], #7  -3
                                        [0,0,0,0,0,0,0,0,1,1,0], #8  -2
                                        [0,0,0,0,0,0,0,0,0,1,0], #9  -1
                                        [0,0,0,0,0,0,0,0,0,0,0], #10  0
                                        [1,0,0,0,0,0,0,0,0,1,0], #11  +1
                                        [1,0,0,0,0,0,0,0,1,1,0], #12  +2
                                        [1,0,0,0,0,0,0,1,1,1,0], #13  +3
                                        [1,0,0,0,0,0,1,1,1,1,0], #14  +4
                                        [1,0,0,0,0,1,1,1,1,1,0], #15  +5
                                        [1,0,0,0,1,1,1,1,1,1,0], #16  +6-10
                                        [1,0,0,1,1,1,1,1,1,1,0], #17 +11-20
                                        [1,0,1,1,1,1,1,1,1,1,0], #18 +21-30
                                        [1,1,1,1,1,1,1,1,1,1,0], #19 +31-inf
                                        ]
    
    def get_position_embedding_matrix(self):
        import copy
        return copy.deepcopy (self.__PositionEmbeddingMappingMatrix)
    
    def get_embedding_index(self,relativeDistance):
        x = relativeDistance
        if x<=-31:
            return 1
        elif (-30 <= x <= -21):
            return 2
        elif (-20 <= x <= -11):
            return 3
        elif (-10 <= x <=  -6):
            return 4
        elif (-5  <= x <= +5):
            return x+10
        elif (6   <= x <= 10):
            return 16
        elif (11  <= x <= 21):
            return 17
        elif (21  <= x <= 30):
            return 18
        else:
            return 19
    
    def shape(self):
        return (len(self.__PositionEmbeddingMappingMatrix) , len (self.__PositionEmbeddingMappingMatrix[0]))
        