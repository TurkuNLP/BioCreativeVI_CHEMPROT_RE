import sys, os , argparse 
import numpy as np
from collections import OrderedDict
from xml.etree import cElementTree as ET

from TSDPTSVMCore import TEESDocumentHelper
from TSDPTSVMCore import ProjectRunner

import chemprot_tools 
import EVEX_GENERAL_TOOLS as EVEX

confs_index = OrderedDict([
             ("neg"   , 0),
             ("CPR:1" , 1), 
             ("CPR:2" , 2), 
             ("CPR:3" , 3), 
             ("CPR:4" , 4), 
             ("CPR:5" , 5), 
             ("CPR:6" , 6), 
             ("CPR:7" , 7), 
             ("CPR:8" , 8), 
             ("CPR:9" , 9), 
             ("CPR:10", 10)])

confs_reverse_index = {confs_index[key]:key for key in confs_index}

def decode_interaction_conf (conf,lp,PROGRAM_Halt):
    #conf = "neg:0.563632,CPR:1:0.0300603,CPR:10:0.0198846,CPR:2:0.270922,CPR:3:0.0147823,CPR:4:0.0194888,CPR:5:0.0223783,CPR:6:0.0405166,CPR:7:0.00400404,CPR:8:0.00325608,CPR:9:0.0110745" 
    conf = conf.split(",") 
    all_confs_np = np.zeros(11,dtype=np.float)
    
    for confidence_str in conf:
        if "neg:" in confidence_str:
            all_confs_np[confs_index["neg"]] = float(confidence_str.split("neg:")[-1])
        elif "CPR:1:" in confidence_str:
            all_confs_np[confs_index["CPR:1"]] = float(confidence_str.split("CPR:1:")[-1])
        elif "CPR:2:" in confidence_str:
            all_confs_np[confs_index["CPR:2"]] = float(confidence_str.split("CPR:2:")[-1])
        elif "CPR:3:" in confidence_str:
            all_confs_np[confs_index["CPR:3"]] = float(confidence_str.split("CPR:3:")[-1])
        elif "CPR:4:" in confidence_str:
            all_confs_np[confs_index["CPR:4"]] = float(confidence_str.split("CPR:4:")[-1])
        elif "CPR:5:" in confidence_str:
            all_confs_np[confs_index["CPR:5"]] = float(confidence_str.split("CPR:5:")[-1])
        elif "CPR:6:" in confidence_str:
            all_confs_np[confs_index["CPR:6"]] = float(confidence_str.split("CPR:6:")[-1])
        elif "CPR:7:" in confidence_str:
            all_confs_np[confs_index["CPR:7"]] = float(confidence_str.split("CPR:7:")[-1])
        elif "CPR:8:" in confidence_str:
            all_confs_np[confs_index["CPR:8"]] = float(confidence_str.split("CPR:8:")[-1])
        elif "CPR:9:" in confidence_str:
            all_confs_np[confs_index["CPR:9"]] = float(confidence_str.split("CPR:9:")[-1])
        elif "CPR:10:" in confidence_str:
            all_confs_np[confs_index["CPR:10"]] = float(confidence_str.split("CPR:10:")[-1])
        else:
           PROGRAM_Halt ("Error in confidence string: " +str(conf)) 
    return all_confs_np        
    
def get_focus_pred_files(PredictionFolder, EpochFocus,lp):
    focus_files = [] 
    all_files = EVEX.GetAllFilesPathAndNameWithExtensionInFolder(PredictionFolder, "xml", ProcessSubFoldersAlso = False) 
    for fileaddr in all_files:
        if "_EpochNo"+ str(EpochFocus)+".xml" in fileaddr:
            focus_files.append(fileaddr)

    MSG = ["-"*30+"FOCUS FILES:"+"-"*30]
    for fileaddr in focus_files:
        MSG.append (fileaddr) 
    MSG.append ("-"*80)
    lp(MSG)
    return focus_files    

def ReadOnePredFile (APredFile,lp,PROGRAM_Halt,TEES_Helper):
    interactions = OrderedDict() 
    lp ("Reading file: " + APredFile)
    tree = TEES_Helper.LoadTEESFile_GetTreeRoot (APredFile) ;
    for sentence in tree.findall('.//sentence'):
        sentence_id = sentence.attrib["id"]
        entities = {}
        
        for entity in sentence.findall ("entity"):
            eid = entity.attrib["id"]
            etp = entity.attrib["type"]
            entities[eid] = etp 
            
        for interaction in sentence.findall('interaction'):
            interaction_id      = interaction.attrib["id"]
            interaction_e1      = interaction.attrib["e1"]
            interaction_e2      = interaction.attrib["e2"]
            interaction_conf    = interaction.attrib["conf"] 
            interaction_conf_np = decode_interaction_conf (interaction_conf,lp,PROGRAM_Halt)
            
            assert interaction_e1 in entities 
            assert interaction_e2 in entities 
            e1_tp = entities[interaction_e1]
            e2_tp = entities[interaction_e2]
            
            assert e1_tp.lower() == "chemical"
            assert e2_tp.lower() == "gene" 
            
            key = (sentence_id, interaction_id,interaction_e1,interaction_e2,e1_tp,e2_tp)
            interactions[key] = [interaction_conf,interaction_conf_np]
    return interactions

def aggregate_results (all_predictions,lp,PROGRAM_Halt):
    aggregated_results = OrderedDict()
    for key in all_predictions[0]:
        aggr_conf_np = [i[key][1] for i in all_predictions]
        confs_npy = np.sum (aggr_conf_np, axis=0)
        interaction_type = confs_reverse_index[np.argmax(confs_npy)]
        confs_str = "" 
        for class_tp in confs_index:
            confs_str+= class_tp + ":" + str(confs_npy[confs_index[class_tp]]) + "," 
        confs_str = confs_str[0:-1] # remove last "," 
        aggregated_results[key] = (confs_str, confs_npy , interaction_type)
    return aggregated_results

def writeback (PRESERVE_ONLY_TARGET_CLASSES_PREDICTIONS, aggregated_results,AggregationFileName,GOLD_XML,TEES_Helper,Configs,lp,PROGRAM_Halt):
    lp ("\nCreating aggregation TEES_XML ... please wait ...")
    tree = TEES_Helper.LoadTEESFile_GetTreeRoot (GOLD_XML, ReturnWholeTree=True)
    TEES_Helper.TREEOPERATION_RemoveAllInteractions(tree)
    TEES_Helper.TREEOPERATION_RemoveAllParseInfo (tree)
    InteractionElementName = Configs["InteractionElementName"];
    InteractionElementClassAttributeName = Configs["InteractionElementClassAttributeName"]; 
            
    for key in aggregated_results:
        s_id, int_id, e1_id, e2_id, e1_tp, e2_tp = key 
        confs_str, confs_npy , interaction_type = aggregated_results[key]
        
        if PRESERVE_ONLY_TARGET_CLASSES_PREDICTIONS and (not interaction_type in ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"]):
            continue 

        sentence = tree.find('./document/sentence[@id="%s"]' % s_id) # This might be slow
        #import pdb ; pdb.set_trace()
        
        interaction = ET.Element(InteractionElementName);
        interaction.set('id',int_id) 
        interaction.set('e1', e1_id) 
        interaction.set('e2', e2_id) 
        interaction.set(InteractionElementClassAttributeName, interaction_type)
        interaction.set('conf', confs_str)
        sentence.append (interaction)
    lp ("Saving final file into: " + AggregationFileName)
    TEES_Helper.Save_TEES_File (tree,AggregationFileName)

def MakeDirIfNotExists(folder):
    if not os.path.exists (folder):
        os.mkdir(folder)
        
if __name__ == "__main__":
    #<<<CRITICAL>>> default is false, because we want the system combination decide on all predictions, not only target-class predictions
    GOLD_FILE_DEVEL = "DATA/BLLIP_BIO-SC-CCprocessed/CP17-devel.xml"    
    GOLD_FILE_TEST  = "DATA/BLLIP_BIO-SC-CCprocessed/CP17-test.xml" 
    
    PRESERVE_ONLY_TARGET_CLASSES_PREDICTIONS = False 
    
    PARAM_ConfigFileAddress  = "CONFIG_ChemProt_11Class_1M.json" 
    PARAM_LogFileAddress     = "LOGS/JE_AggregationEvaluation.log" 
    PRJ = ProjectRunner.TSDP_TSVM_ProjectRunner (PARAM_ConfigFileAddress , PARAM_LogFileAddress) 
    Configs , lp , PROGRAM_Halt = PRJ.Configs  , PRJ.lp , PRJ.PROGRAM_Halt
    
    MAX_EPOCHS = 6
    Architectures = list()
    Architectures+=["Arch_SDP3lstms_FS1lstm_MaxPool(300,200,200,300,25,25,10,300,0.2,'tanh')_1"]
    Architectures+=["Arch_SDP3lstms_FS1lstm_MaxPool(300,200,200,300,25,25,10,500,0.2,'tanh')_1"]
    Architectures+=["Arch_SDP3lstms_FS1lstm_MaxPool(300,200,200,300,25,25,10,1024,0.2,'tanh')_1"]
    
    import collections
    ALL_RESULTS = collections.OrderedDict () 
    
    for architecture in Architectures:
        FINALS = {
            "Devel": {},
            "Test_TrainedOnTrain" : {},
            "Test_TrainedOnTrainAndDevel" : {},
        }
        
        for Experiment in ["Devel","Test_TrainedOnTrain","Test_TrainedOnTrainAndDevel"]:
            for EpochFocus in range(1,MAX_EPOCHS+1):
                if Experiment == "Devel":
                    GOLD_XML              = GOLD_FILE_DEVEL
                    PredictionFolder      = chemprot_tools.this_program_folder_path + "/PREDICTIONS/TrainedOnTrain/DevelPred/"+architecture+"/" 
                    AggregationFolderName = chemprot_tools.this_program_folder_path + "/PREDICTIONS_AGGREGATIONS/TrainedOnTrain/DevelPred/"+architecture+"/" 
                    MakeDirIfNotExists (AggregationFolderName)
                    AggregationFileName   = AggregationFolderName + "DevelAggr_1M_Epoch"+str(EpochFocus)+".xml" 
    
                elif Experiment == "Test_TrainedOnTrain":
                    GOLD_XML              = GOLD_FILE_TEST
                    PredictionFolder      = chemprot_tools.this_program_folder_path + "/PREDICTIONS/TrainedOnTrain/TestPred/"+architecture+"/"
                    AggregationFolderName = chemprot_tools.this_program_folder_path + "/PREDICTIONS_AGGREGATIONS/TrainedOnTrain/TestPred/"+architecture+"/"
                    MakeDirIfNotExists (AggregationFolderName)
                    AggregationFileName   = AggregationFolderName + "TestAggr_1M_Epoch"+str(EpochFocus)+".xml"
    
                elif Experiment == "Test_TrainedOnTrainAndDevel":
                    GOLD_XML              = GOLD_FILE_TEST
                    PredictionFolder      = chemprot_tools.this_program_folder_path + "/PREDICTIONS/TrainedOnTrainAndDevel/TestPred/"+architecture+"/" 
                    AggregationFolderName = chemprot_tools.this_program_folder_path + "/PREDICTIONS_AGGREGATIONS/TrainedOnTrainAndDevel/TestPred/"+architecture+"/" 
                    MakeDirIfNotExists (AggregationFolderName)
                    AggregationFileName   = AggregationFolderName + "TestAggr_1M_Epoch"+str(EpochFocus)+".xml" 
    
                ####################################################################################################
                
                MSG = ["-"*80]
                MSG+= ["Architecture            : " + architecture] 
                MSG+= ["Experiment              : " + Experiment]
                MSG+= ["GOLD_XML                : " + GOLD_XML]
                MSG+= ["PredictionFolder        : " + PredictionFolder]
                MSG+= ["EpochFocus              : " + str(EpochFocus) ]
                MSG+= ["AggregationFileName     : " + AggregationFileName ]
                MSG+= ["-"*30]
                lp (MSG)    
                TEES_Helper = TEESDocumentHelper.FLSTM_TEESXMLHelper(Configs, lp , PROGRAM_Halt)
                FocusFiles = get_focus_pred_files(PredictionFolder, EpochFocus,lp)

                AggregationTSVName  = "/temp/temp_chemprot_pred_aggr.tsv" 
                
                all_predictions = []    
                for fileaddr in FocusFiles:
                    all_predictions.append( ReadOnePredFile (fileaddr,lp,PROGRAM_Halt,TEES_Helper) )
                
                assert len(set([len(i) for i in all_predictions])) == 1
                aggregated_results = aggregate_results (all_predictions,lp,PROGRAM_Halt)
                writeback (PRESERVE_ONLY_TARGET_CLASSES_PREDICTIONS, aggregated_results,AggregationFileName,GOLD_XML,TEES_Helper,Configs,lp,PROGRAM_Halt)
                lp ("-"*80)
                fscore = chemprot_tools.ExternalEvaluation ( {"lp":lp,"PROGRAM_Halt" : PROGRAM_Halt,"PredictedFile": AggregationFileName,"GoldFilesList": [GOLD_XML]} )
                lp ("Final F-Score: " + str(fscore))
                FINALS[Experiment][EpochFocus] = fscore
                lp (["",""])    
        ALL_RESULTS[architecture] = FINALS 

    for architecture in ALL_RESULTS.keys():
        FINALS = ALL_RESULTS[architecture] 
        MSG = ["*"*80, "FINAL RESULTS - ARCHITECTURE: " + architecture , "-"*40]
        for key in ["Devel","Test_TrainedOnTrain","Test_TrainedOnTrainAndDevel"]:
            for EpochFocus in range(1,MAX_EPOCHS+1):
                MSG += ["Experiment: " + key + "   Epoch: " + str(EpochFocus) + "  f-score: " + str(FINALS[key][EpochFocus])]
        MSG += ["*"*80]
        lp (MSG)
    lp ("END.")    