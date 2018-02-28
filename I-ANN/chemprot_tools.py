import os , subprocess 
this_program_folder_path = os.path.dirname(os.path.realpath(__file__))
TEES_develBranch_folder  = "/home/farmeh/Desktop/PROJECTS/GIT/TEES_DEVELBranch/TEES/" 
EvalKitFolder            = this_program_folder_path + "/ChemProt_Official_EvalKit/" 

def ExternalEvaluation (input_params):
    lp            = input_params["lp"]
    PROGRAM_Halt  = input_params["PROGRAM_Halt"]
    #Set           = input_params["Set"]
    PredictedFile = input_params["PredictedFile"]
    GoldFilesList = input_params["GoldFilesList"]
    GoldFile = GoldFilesList[0]
    
    if not os.path.isfile(PredictedFile):
        PROGRAM_Halt ("Predicted file does not exists: " + PredictedFile)
    
    if (not isinstance(GoldFilesList,list)) or (len(GoldFilesList)<1):
        PROGRAM_Halt ("GoldFilesList should be a list with at least one file:" + str(GoldFilesList))
    
    if not os.path.isfile(GoldFile):
        PROGRAM_Halt ("GoldFile does not exist:" + GoldFile)
    
    
    try:
        f = open("evaluate.sh" , "wt")
        f.write ("rm -rf " + EvalKitFolder +"temp/*" + " \n")
        f.write ("rm -rf " + EvalKitFolder +"out/*"  + " \n")
            
        tsv_fileaddress_pred = EvalKitFolder + "temp/PRED_" + PredictedFile.split("/")[-1].replace(".xml", ".tsv") 
        tsv_fileaddress_gold = EvalKitFolder + "temp/GOLD_" + GoldFile.split("/")[-1].replace(".xml", ".tsv") 
        
        cmd_create_pred = "python " + TEES_develBranch_folder + 'preprocess.py -i "' + PredictedFile + '" -o ' +  tsv_fileaddress_pred  + " --steps EXPORT_CHEMPROT \n" 
        cmd_create_gold = "python " + TEES_develBranch_folder + 'preprocess.py -i "' + GoldFile      + '" -o ' +  tsv_fileaddress_gold  + " --steps EXPORT_CHEMPROT \n" 
        
        f.write (cmd_create_pred)
        f.write (cmd_create_gold) 
        
        f.write ("cd " + EvalKitFolder + " \n")
        cmd_evaluate = "sh eval.sh " + tsv_fileaddress_pred + " " + tsv_fileaddress_gold + " \n"
        f.write (cmd_evaluate) 
        f.close() 
    
        p = subprocess.Popen('sh ' + this_program_folder_path + '/evaluate.sh', stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait() 
        
        process_results = "output : " + str(output) + "\nerr : " + str(err) + "\np_status : " + str(p_status)
        
        if (p_status<>0) or (err <> None) or (len(output.split("\n"))<> 5) or (not 'was successfully created.' in output): 
            PROGRAM_Halt ("problem running external evaluator:\n" + str(input_params) + "\nRES = " + str(process_results)) 
        
        with open(EvalKitFolder+"/out/eval.txt","r") as f:
            lines = ["\t"+line.strip("\n") for line in f if len(line)>1]
        MSG = ["-"*33+"EXTERNAL EVALUATION RESULTS"+"-"*33]  + lines + ["-"*80]
        lp (MSG)
        F_score = float(lines[-1].split("F-score: ")[-1])
        return F_score
        
    except Exception as E:
        PROGRAM_Halt ("problem running external evaluator:\n" + str(input_params) + "\nError: " + E.message + "\n")


def ConvertXMLtoTSV_TEES_Wrapper(Input_TEES_XML_File,Output_TSV_File,lp,PROGRAM_Halt):
    try:
        lp (["Converting TEES_XML into TSV:" , "TEES XML : " + Input_TEES_XML_File , "TSV      : " + Output_TSV_File] )
        
        convert_cmd = "python " + TEES_develBranch_folder + 'preprocess.py -i "' + Input_TEES_XML_File + '" -o ' +  Output_TSV_File  + " --steps EXPORT_CHEMPROT" 
        p = subprocess.Popen(convert_cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait() 
        
        process_results = "output : " + str(output) + "\nerr : " + str(err) + "\np_status : " + str(p_status)
        lp (["" , "process results:" , process_results])
        
        if (p_status<>0) or (err <> None): 
            PROGRAM_Halt ("problem running external evaluator:\n" + str(convert_cmd) + "\nRES = " + str(process_results)) 

        lp ("Conversion successfull.")
        
    except Exception as E:
        PROGRAM_Halt ("problem converting TEES_XML into TSV:\n" + "\nError: " + E.message + "\n")

print "THIS PROGRAM FOLDER            : " , this_program_folder_path
print "TEES DEVELOPMENT BRANCH FOLDER : " , TEES_develBranch_folder
print "CHEMPROT EVALKIT FOLDER        : " , EvalKitFolder 