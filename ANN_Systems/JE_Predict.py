import argparse 
from TSDPTSVMCore import FeatureGeneration
from TSDPTSVMCore import REPipeline_ANN

if __name__ == "__main__":
    #example: -experiment 0 -arch_index 2 -seed_index 2
    Archictectures_MaxEpoch = list()

    Archictectures_MaxEpoch+= [("Arch_SDP3lstms_FS1lstm_MaxPool(300,200,200,300,25,25,10,300,0.2,'tanh')" , 6)]     
    Archictectures_MaxEpoch+= [("Arch_SDP3lstms_FS1lstm_MaxPool(300,200,200,300,25,25,10,500,0.2,'tanh')" , 6)]     
    Archictectures_MaxEpoch+= [("Arch_SDP3lstms_FS1lstm_MaxPool(300,200,200,300,25,25,10,1024,0.2,'tanh')", 6)]     
    
    parser = argparse.ArgumentParser(description='predict chemprot on taitto')
    parser.add_argument('-experiment', action="store", dest="e", type=int , choices=[0,1]       , help='0: TrainOnTrain , 1:TrainOnTrainAndDevel')
    parser.add_argument('-arch_index', action="store", dest="a", type=int , choices=range(0,12) , help='architecture index, from 0 to 10',)
    parser.add_argument('-seed_index', action="store", dest="r", type=int , choices=range(0,5)  , help='random seed index , from 0 to 4',)
    
    args = parser.parse_args() 
    
    _Experiment   = ["TrainOnTrain" , "TrainOnTrainAndDevel"][args.e]
    _RandomSeed   = [None,20,1981,2017][args.r]
    _Architecture = [Archictectures_MaxEpoch[args.a]]
    
    #FIXED BEST VALUES ... 
    KTOP          = 1 
    parse_tp      = "CCprocessed"
    WAE_weight    = 5
    SDP_direction = "from_e1value_to_e2value"

    filename   = "NEW_CHEMPROT_PREPROCESSED_DATA" 
    config_file= "CONFIG_ChemProt_11Class_1M.json" 
    
    train_file = "DATA/BLLIP_BIO-SC-CCprocessed/CP17-train.xml"
    devel_file = "DATA/BLLIP_BIO-SC-CCprocessed/CP17-devel.xml"
    test_file  = "DATA/BLLIP_BIO-SC-CCprocessed/CP17-test.xml" 

    FG_PARAMS = FeatureGeneration.TSDP_TSVM_FeatureGenerationParams()
    #1-Feature Generation Parameters ...
    if SDP_direction == "from_e1value_to_e2value":
        FG_PARAMS.SDPDirection = FeatureGeneration.TSDP_TSVM_SDPFeatureGenerationDirection.from_e1value_to_e2value
    
    elif SDP_direction == "from_1stOccurring_to_2nd":
        FG_PARAMS.SDPDirection = FeatureGeneration.TSDP_TSVM_SDPFeatureGenerationDirection.from_1stOccurring_to_2nd
        
    FG_PARAMS.EdgeWeight_WordAdjacency   = WAE_weight
    FG_PARAMS.EdgeWeight_ParseDependency = 1
    FG_PARAMS.TopKShortestPathNumber     = 1
    FG_PARAMS.NumberOfParallelProcesses = -1 #use all core minus 1 , valid values: -1,1,2,3,4,5, ... 
    FG_PARAMS.UsePathLengthCutOffHeuristic = True 
    FG_PARAMS.PathLengthCutOffHeuristicMAXLEN = 20

    #<<<CRITICAL>>>
    FG_PARAMS.ReplaceVectorForEntityTypeIfTokenNotFound = {"gene":"protein" , "chemical" : "chemical"}
    
    
    #2-Relation Extraction Parameters ...
    RE_PARAMS = REPipeline_ANN.TSP_TANN_RE_PARAMS() 
    RE_PARAMS.KERAS_optimizer      = "nadam" 
    
    #<<<CRITICAL>>>            
    RE_PARAMS.KERAS_minibatch_size = 16 #<<<CRITICAL>>> 
    RE_PARAMS.KERAS_optimizer_lr = 0.0002 #<<<CRITICAL>>> 

    #<<<CRITICAL>>> Is set to true, hence not to optimize on a particular order of sentences in the Training set.         
    RE_PARAMS.KERAS_shuffle_training_examples = True  

    RE_PARAMS.TrainingSet_Files_List    = [train_file]
    RE_PARAMS.DevelopmentSet_Files_List = [devel_file]
    RE_PARAMS.TestSet_Files_Lists       = [test_file]
    
    RE_PARAMS.TrainingSet_DuplicationRemovalPolicy    = "DISCARD"
    RE_PARAMS.DevelopmentSet_DuplicationRemovalPolicy = "DISCARD"
    RE_PARAMS.TestSet_DuplicationRemovalPolicy        = "IGNORE" 

    RE_PARAMS.Classification_OptimizationMetric = "total_MINUS_negative_cpr:1_cpr:2_cpr:7_cpr:8_cpr:10_f1_micro" 
    #RE_PARAMS.Classification_ExternalEvaluator = chemprot_tools.ExternalEvaluation

    processed_gold_file = "DATA/" + filename + ".pickle" #new with test set labels, sentences are sorted based on shared task experiments 
    
    DataPreparationSteps = {
        "preprocess": False, 
        "findshortestpaths": False,
        "load_gold_and_data": processed_gold_file, 
        "save_gold_and_data": None, 
    }
    """
    BEST IN THE SHARED TASK WAS: 
    Archictectures_MaxEpoch = [ ("Arch_TopKP_3lstms_WA_wpd(300,200,200,'tanh',25,25,200,'tanh',0.2)" , 5) ] 
    With mini batch size=32 , and default learning rate of keras for nadam. 
    """            
    
    #New Architecture Experiments:
    if _Experiment == "TrainOnTrain":
        log_file = "LOGS/JE_TrainedOnTrain_1M_" 
    else:
        log_file = "LOGS/JE_TrainedOnTrainAndDevel_1M_" 
    
    log_file += filename + "_Seed_" + str(_RandomSeed) + "_ArchIndex_" + str(args.a) + ".log" 

    if _Experiment == "TrainOnTrain":
        DEVEL_PRED_FOLDER = "PREDICTIONS/TrainedOnTrain/DevelPred/" 
        TESTT_PRED_FOLDER = "PREDICTIONS/TrainedOnTrain/TestPred/" 

        RE_PARAMS.PredictDevelSet  = True
        RE_PARAMS.EvaluateDevelSet = True
        RE_PARAMS.WriteBackDevelSetPredictions  = True
        RE_PARAMS.DevelSetPredictionOutputFolder= DEVEL_PRED_FOLDER
        RE_PARAMS.ProcessDevelSetAfterEpochNo   = 0
        
        RE_PARAMS.PredictTestSet  = True
        RE_PARAMS.EvaluateTestSet = True 
        RE_PARAMS.WriteBackTestSetPredictions   = True
        RE_PARAMS.TestSetPredictionOutputFolder = TESTT_PRED_FOLDER
        RE_PARAMS.ProcessTestSetAfterEpochNo  = 0  

        RE_Pipeline = REPipeline_ANN.TSP_TANN_RE_Pipeline (config_file,log_file, RE_PARAMS)
        RE_Pipeline.RunSimplePipeline (DataPreparationSteps, FG_PARAMS, _Architecture, useOnlyXTopPathsForFeatureGeneration=KTOP, RandomSeed=_RandomSeed)
        
    elif _Experiment == "TrainOnTrainAndDevel":
        TESTT_PRED_FOLDER = "PREDICTIONS/TrainedOnTrainAndDevel/TestPred/" 

        RE_PARAMS.PredictDevelSet  = True
        RE_PARAMS.EvaluateDevelSet = True
        RE_PARAMS.WriteBackDevelSetPredictions  = False
        RE_PARAMS.DevelSetPredictionOutputFolder= None
        RE_PARAMS.ProcessDevelSetAfterEpochNo   = 0
        
        RE_PARAMS.PredictTestSet  = True
        RE_PARAMS.EvaluateTestSet = True 
        RE_PARAMS.WriteBackTestSetPredictions   = True
        RE_PARAMS.TestSetPredictionOutputFolder = TESTT_PRED_FOLDER
        RE_PARAMS.ProcessTestSetAfterEpochNo  = 0  

        RE_Pipeline = REPipeline_ANN.TSP_TANN_RE_Pipeline (config_file,log_file, RE_PARAMS)
        RE_Pipeline.RunSimplePipeline_TrainOnTrainAndDevel(DataPreparationSteps, FG_PARAMS, _Architecture, useOnlyXTopPathsForFeatureGeneration=KTOP, RandomSeed=_RandomSeed)
