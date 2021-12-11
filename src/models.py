from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import CrossValidator,  TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import countDistinct

from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

import numpy as np
from datetime import *
from math import floor
import random
import sys
import time

def spark_nlp_model(spark, df_train, df_val, df_test, model):
    
    print('converting pandas df to spark df...', datetime.now())
    # convert data sets to spark DF
    train_set = spark.createDataFrame(df_train) 
    val_set = spark.createDataFrame(df_val) 
    test_set = spark.createDataFrame(df_test) 

    print('Data set schema')
    print(train_set.printSchema())
    
    print('setting up spark NLP pipeline...', datetime.now())
    # spark NLP pipeline
    tokenizer = Tokenizer(inputCol="TEXT", outputCol="words")
    cv = CountVectorizer(vocabSize=3000, inputCol="words", outputCol='cv')
    idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol = "Label", outputCol = "label")
    pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, model])

    # fit model
    print('Fitting model...', datetime.now())
    pipelineFit = pipeline.fit(train_set)
    
    val_pred = pipelineFit.transform(val_set)
    test_pred = pipelineFit.transform(test_set)

    print('Fitted pipeline Schema')
    print(val_pred.printSchema())
    print('Completed script...', datetime.now())
    
    return pipelineFit, val_pred, test_pred


