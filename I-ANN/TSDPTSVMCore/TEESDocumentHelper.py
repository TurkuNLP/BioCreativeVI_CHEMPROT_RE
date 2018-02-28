import sys ; 
import inspect ; 
import GeneralFunctions as GF ; 
from xml.etree import cElementTree as ET
from gzip import GzipFile
import gzip
from itertools import product, permutations ; 
import cPickle as pickle

class FLSTM_TEESXMLHelper:
    def __init__(self, Configs, lp=None, PROGRAM_Halt=None):
        self.Configs = Configs ; 
        
        if lp <> None:
            self.lp = lp 
        else:
            def lp (ARG_msg):
                try:        
                    LOCAL_CallerClassName = str(inspect.stack()[1][0].f_locals["self"].__class__) ; 
                except:
                    LOCAL_CallerClassName = "" ; 
                    
                LOCAL_CallerFunction = inspect.currentframe().f_back.f_code.co_name ; 
                if isinstance(ARG_msg, basestring):
                    ARG_msg = [ARG_msg] ;
                HEADER = "[" + GF.DATETIME_GetNowStr() + "] [" + LOCAL_CallerClassName + "." + LOCAL_CallerFunction + "]: " ; 
                print HEADER ;
                for itemstr in ARG_msg:
                    itemstr = str(itemstr).replace ('\r','\n'); 
                    for item in itemstr.split ("\n"):
                        if len(item)==0:
                            item = "-" ;
                        item = "      "+item ; 
                        print item ; 
                print "" ;
            self.lp = lp ; 

        if PROGRAM_Halt <> None:
            self.PROGRAM_Halt = PROGRAM_Halt ;
        else:
            def PROGRAM_Halt (ARG_HaltErrMSG):
                PARAM_CallerFunctionName = inspect.currentframe().f_back.f_code.co_name ; 
                self.lp (["*"*80 , "HALT REQUESTED BY FUNCTION: " + PARAM_CallerFunctionName , "HALT MESSAGE: "+ ARG_HaltErrMSG , "HALTING PROGRAM!!!" , "*"*80]);
                sys.exit (-1); 
            self.PROGRAM_Halt = PROGRAM_Halt ;
            
    def LoadTEESFile_GetTreeRoot (self,m_FilePathName , ReturnWholeTree=False):
        #self.lp (["Loading TEES file from address and getting Root:" , m_FilePathName])
        try:    
            if m_FilePathName.endswith (".xml"):
                tree = ET.parse(m_FilePathName); 
                #self.lp ("tree sucessfully loaded from file.")
                if ReturnWholeTree:
                    return tree;
                else:
                    return tree.getroot (); 
            elif m_FilePathName.endswith (".gz"):
                f = gzip.open (m_FilePathName, "rb"); 
                root = ET.fromstring(f.read ()); 
                f.close ();
                #self.lp ("sucessfully loaded and Root retrieved.")
                return root ; 
            else:
                raise Exception ("unknown file type:" + m_FilePathName.split(".")[-1]); 
        except Exception as E:
            print "Error occurred:" + E.message ; 
            sys.exit(-1); 


    def Save_TEES_File (self, tree, out_path):
        #self.lp ("Saving Tree into XML file:" + out_path); 
        if out_path.endswith('.gz'):
            out_f = gzip.open(out_path, 'wb')
            tree.write(out_f)
            out_f.close()
        else:
            tree.write(out_path)

    def TREEOPERATION_RemoveAllInteractions (self, tree):
        #self.lp ("REMOVING ALL INTERACTIONS FROM TREE ...") ; 
        counter = 0 ; 
        for sentence in tree.findall('.//sentence'):
            for interaction in sentence.findall(self.Configs["InteractionElementName"]):
                counter += 1 ; 
                sentence.remove(interaction)
        #self.lp (str(counter) + " INTERACTIONS REMOVED ...") ; 

    def TREEOPERATION_RemoveAllParseInfo (self, tree):
        #self.lp ("REMOVING ALL PARSE-INFO FROM TREE ...") ; 
        counter = 0 ; 
        for sentence in tree.findall('.//sentence'):
            for interaction in sentence.findall("analyses"):
                counter += 1 ; 
                sentence.remove(interaction)
        #self.lp (str(counter) + " PARSE-INFO REMOVED ...") ; 


    def TREEOPERATION_CheckIfAllPossibleRelationsExist (self, Root):
        if (Root == None):
            self.PROGRAM_Halt ("Root is None"); 
        
        for document_element in Root.findall (".//document"):
            for sentence_element in document_element.findall (".//sentence"):
                
                #Get all valid entities:
                VALID_ENTITIES_BY_TYPE = {};
                VALID_ENTITIES_BY_ID = {};
                for entity_element in sentence_element.findall (".//entity"):
                    e_tp = entity_element.attrib["type"].lower() ;
                    if e_tp in self.Configs["ValidEnityTypesForRelations"]:
                        if VALID_ENTITIES_BY_TYPE.has_key (e_tp):
                            VALID_ENTITIES_BY_TYPE[e_tp].append (entity_element.attrib["id"]);
                        else:
                            VALID_ENTITIES_BY_TYPE[e_tp] = [entity_element.attrib["id"]] ; 
                        
                        VALID_ENTITIES_BY_ID[entity_element.attrib["id"]] = e_tp ;  
                        
                #Get all valid interactions:
                VALID_INTERACTIONS_IN_FILE = [] ; 
                LOCAL_InteractionElementName = self.Configs["InteractionElementName"] ; 
                for interaction_element in sentence_element.findall ("./" + LOCAL_InteractionElementName):
                    pair_attrib = interaction_element.attrib ; 
                    #check if valid relation:                     
                    e1 , e2 = pair_attrib["e1"] , pair_attrib["e2"] ; 
                    if (not e1 in VALID_ENTITIES_BY_ID) or (not e2 in VALID_ENTITIES_BY_ID):
                        self.PROGRAM_Halt ("Corresponding entity for the given interaction is not found:" + str(pair_attrib)) ;
                        
                    e1_tp , e2_tp = VALID_ENTITIES_BY_ID[e1] , VALID_ENTITIES_BY_ID[e2] ; 
                    if [e1_tp, e2_tp] in self.Configs["ListOf_ValidInteractingPairEntityTypes"] or \
                       [e2_tp, e1_tp] in self.Configs["ListOf_ValidInteractingPairEntityTypes"]:
                        VALID_INTERACTIONS_IN_FILE.append ((e1 , e2)) ; 
                
                #Check if all possible interactions exists:
                #TODO: now we support only one direction!!!
                #so only: bacteria-habitat, but not habitat-bacteria ... 
                for interacting_entitytype_pair in self.Configs["ListOf_ValidInteractingPairEntityTypes"]:
                    type1=interacting_entitytype_pair[0];
                    type2=interacting_entitytype_pair[1];
                    if (not type1 in VALID_ENTITIES_BY_TYPE) or (not type2 in VALID_ENTITIES_BY_TYPE):
                        continue; #for example habitat-geographical is valid, but sentence does not have any geographical entitiy. 
                    
                    Entity_Type1_List = VALID_ENTITIES_BY_TYPE[type1];
                    Entity_Type2_List = VALID_ENTITIES_BY_TYPE[type2];

                    if type1 <>  type2: #for example, bacteria-habitat--> ALL bacteria-habitat pairs should be in the XML for the sentence
                        PossibleRelations = product (Entity_Type1_List , Entity_Type2_List);

                    else: # for example, drug-drug or protein-protein
                        if type1 in self.Configs["SelfInteractingEntities"]: #for example, a drug can interact with itself ... (A,A) should be in the file, as well as (A.B) or (B,A)
                            PossibleRelations = product (Entity_Type1_List , Entity_Type2_List);
                        else: # only drug A can interact with drug B. So either (A,B) or (B,A) should be in the XML but not (A,A)
                            PossibleRelations = permutations (Entity_Type1_List , 2) ; 
                            
                    for candidate_pair in PossibleRelations:
                        if (not (candidate_pair[0],candidate_pair[1]) in VALID_INTERACTIONS_IN_FILE) and (not (candidate_pair[1],candidate_pair[0]) in VALID_INTERACTIONS_IN_FILE):
                            MSG  = "Necessary relation is missing for sentence:" + sentence_element.attrib["id"] + "\n" ; 
                            MSG+=  "either " + str ((candidate_pair[0],candidate_pair[1])) + "  or  " + str ((candidate_pair[1],candidate_pair[0])) + "should be in XML.\n" ; 
                            MSG+=  "entity types: (" + type1 + "," + type2 + ")." ; 
                            self.PROGRAM_Halt (MSG); 
                    

    def TREEOPERATION_FILL_Version1 (self, tree, pairs):
        #This version does not delete interactions in the original XML file, but sets everything to negatives,
        #and then based on what is predicted and defined in the given pairs, sets the labels.
        #if there are some artificially pairs created, those will cause problems, because those were not in the original
        #xml.
        #cross-sentence relations, duplicates will remain as they are eventhough they might not be predicted at all. 
        InteractionElementName = self.Configs["InteractionElementName"];
        InteractionElementClassAttributeName = self.Configs["InteractionElementClassAttributeName"]; 
        
        NEGATIVE_LABEL = list (self.Configs["WRITEBACK_CLASSES"]["Negative"])[0]; # <<<CRITICAL>>> Use WRITEBACK to get casesensitive classnames, not CLASSES in the config. CASE is Important for label! 
        self.lp ("Setting all elements in tree to negative class.")
        for interaction in tree.findall ('./document/sentence/' + InteractionElementName):
            interaction.set (InteractionElementClassAttributeName , NEGATIVE_LABEL) ; 
        
        ROLES_Dict = {} ;
        for Entity_Type, Role_Type in self.Configs["e1Role_e2Role"]:
            ROLES_Dict[Entity_Type.lower()] = Role_Type ; #<<<CRITICAL>>>
            
        for s_id, int_id, int_type, int_conf , e1_tp, e2_tp in pairs:
            interaction = tree.find('./document/sentence/' + InteractionElementName + '[@id="%s"]' % int_id) # This might be slow
            
            e1_tp , e2_tp = e1_tp.lower() , e2_tp.lower() ; #<<<CRITICAL>>>
            if e1_tp in ROLES_Dict:
                e1_Role = ROLES_Dict[e1_tp];
            else:
                e1_Role = "" ; 
                
            if e2_tp in ROLES_Dict:
                e2_Role = ROLES_Dict[e2_tp];
            else:
                e2_Role = "" ; 

            interaction.set(InteractionElementClassAttributeName, int_type)
            interaction.set('conf'  , int_conf)
            interaction.set('e1Role', e1_Role) 
            interaction.set('e2Role', e2_Role) 

    def TREEOPERATION_FILL_Version2 (self, tree, pairs, RemoveAllParseInfo=False):
        #This version first DELETES all interaction found in the GOLD from the tree, and 
        #then recreates the interaction elements given in the input pairs with their predicted types. 
        #Hence cross-sentence relations and duplicates might be absent in the final file.
        #besides, this version can handle artificially created and predicted interactions which are absent in the gold ...
        InteractionElementName = self.Configs["InteractionElementName"];
        InteractionElementClassAttributeName = self.Configs["InteractionElementClassAttributeName"]; 
        
        #remmoving any interaction element from the tree ...
        self.TREEOPERATION_RemoveAllInteractions (tree)
        
        #removing all parsing-info if requested
        if RemoveAllParseInfo:
            self.TREEOPERATION_RemoveAllParseInfo (tree)
        
        ROLES_Dict = {} ;
        for Entity_Type, Role_Type in self.Configs["e1Role_e2Role"]:
            ROLES_Dict[Entity_Type.lower()] = Role_Type ; #<<<CRITICAL>>>
            
        for s_id, int_id, int_type, int_conf , e1_id, e2_id, e1_tp, e2_tp in pairs:
            sentence = tree.find('./document/sentence[@id="%s"]' % s_id) # This might be slow
            
            e1_tp , e2_tp = e1_tp.lower() , e2_tp.lower() ; #<<<CRITICAL>>>
            if e1_tp in ROLES_Dict:
                e1_Role = ROLES_Dict[e1_tp];
            else:
                e1_Role = "" ; 
                
            if e2_tp in ROLES_Dict:
                e2_Role = ROLES_Dict[e2_tp];
            else:
                e2_Role = "" ; 
            
            #import pdb 
            #pdb.set_trace () 
            
            interaction = ET.Element(InteractionElementName);
            interaction.set('id',int_id) 
            interaction.set('e1', e1_id) 
            interaction.set('e2', e2_id) 
            interaction.set(InteractionElementClassAttributeName, int_type)
            interaction.set('e1Role', e1_Role) 
            interaction.set('e2Role', e2_Role) 
            if int_conf <> None: #used for prediction vs processed gold writeback
                interaction.set('conf'  , int_conf)
            sentence.append (interaction)

    def GetAllClassTypesFromTEESXML (self, FileAddress, InteractionElementName, InteractionAttributeName):
        Root = self.LoadTEESFile_GetTreeRoot (FileAddress) ; 
        all_class_tp = set(); 
        for interaction_element in Root.findall (".//document/sentence/"+InteractionElementName):
            all_class_tp.add (interaction_element.attrib[InteractionAttributeName]); 
        return sorted(list(all_class_tp)) ; 
        
            
    def GetAllInteractionElements (self, FileAddress, InteractionElementName, InteractionAttributeName=None):
        Root = self.LoadTEESFile_GetTreeRoot (FileAddress) ; 
        all_interaction_elements = list();
        for interaction_element in Root.findall (".//document/sentence/"+InteractionElementName):
            if InteractionAttributeName == None:
                all_interaction_elements.append ([interaction_element.attrib["id"] , interaction_element.attrib["e1"] , interaction_element.attrib["e2"]]); 
            else:
                all_interaction_elements.append ([interaction_element.attrib["id"] , interaction_element.attrib["e1"] , interaction_element.attrib["e2"], interaction_element.attrib[InteractionAttributeName]]); 
        return all_interaction_elements ; 
    
    def Assert_E1ValueAlwaysE1_E2ValueAlwaysE2 (self, FileAddress, InteractionElementName):
        R = self.GetAllInteractionElements (FileAddress, InteractionElementName);
        for i in R:
            assert i[1].split(".")[-1].lower()=="e1"
            assert i[2].split(".")[-1].lower()=="e2"
        return True ; 
    
    def TREEOPERATION_PreserveOnlyE1E2 (self, InputFileAddress, InteractionElementName, OutputFileAddress=None):
        Root = self.LoadTEESFile_GetTreeRoot  (InputFileAddress , ReturnWholeTree=True); 
        self.lp ("Removing all interaction elements for which e1 <> '.e1' or e1 <> '.e2'") ; 
        Removed = [] ; 
        for sentence_element in Root.findall (".//document/sentence"):
            for interaction_element in sentence_element.findall (".//"+InteractionElementName):
                e1_Value = interaction_element.attrib["e1"];
                e2_Value = interaction_element.attrib["e2"];
                Preserve = e1_Value.endswith(".e1") and e2_Value.endswith(".e2"); 
                if not Preserve:
                    Removed.append (str(interaction_element.attrib)) ; 
                    sentence_element.remove(interaction_element); 
        #self.lp (["Following interactions are removed:" , "-"*20] + [r+"\n" for r in Removed]); 
        if OutputFileAddress:
            self.Save_TEES_File (Root , OutputFileAddress); 
        return Root ;
        
if __name__ == "__main__":
    TEESHelper = FLSTM_TEESXMLHelper (None) ; 
    #root = TEESHelper.LoadTEESFile_GetTreeRoot ("/home/farmeh/Desktop/DATASETS/BioNLP/BB_Corpora/BB_EVENT_16_UsedForSharedTask/BB_EVENT_16-train.xml")
    