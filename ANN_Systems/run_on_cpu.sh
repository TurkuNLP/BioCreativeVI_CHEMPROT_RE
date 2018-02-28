#Ensemble of 4 ANNS - Architecture No:1 - Train on: TrainingSet - Predict: DevelopmentSet + Test Set. 
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 0 -arch_index 0
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 1 -arch_index 0
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 2 -arch_index 0
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 3 -arch_index 0

#Ensemble of 4 ANNS - Architecture No:1 - Train on: TrainingSet + DevelopmentSet - Predict: DevelopmentSet + Test Set. 
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 0 -arch_index 0
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 1 -arch_index 0
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 2 -arch_index 0
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 3 -arch_index 0

#Ensemble of 4 ANNS - Architecture No:2 - Train on: TrainingSet - Predict: DevelopmentSet + Test Set. 
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 0 -arch_index 1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 1 -arch_index 1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 2 -arch_index 1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 3 -arch_index 1

#Ensemble of 4 ANNS - Architecture No:2 - Train on: TrainingSet + DevelopmentSet - Predict: DevelopmentSet + Test Set. 
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 0 -arch_index 1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 1 -arch_index 1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 2 -arch_index 1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 3 -arch_index 1

#Ensemble of 4 ANNS - Architecture No:3 - Train on: TrainingSet - Predict: DevelopmentSet + Test Set. 
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 0 -arch_index 2
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 1 -arch_index 2
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 2 -arch_index 2
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 0 -seed_index 3 -arch_index 2

#Ensemble of 4 ANNS - Architecture No:3 - Train on: TrainingSet + DevelopmentSet - Predict: DevelopmentSet + Test Set. 
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 0 -arch_index 2
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 1 -arch_index 2
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 2 -arch_index 2
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python JE_Predict.py -experiment 1 -seed_index 3 -arch_index 2

#Aggregation code ... 
python JE_AggregateEvaluate.py

