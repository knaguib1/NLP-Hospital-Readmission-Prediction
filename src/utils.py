from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import numpy as np
import pandas as pd

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import CrossValidator,  TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import countDistinct

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def combstr(text, s):
    # segment long notes to short chunk by length s
    # s: length of each short chunk
    # t = str.maketrans(dict.fromkeys(string.punctuation, " "))
    # text = text.lower().translate(t)
    x = text.split(' ')
    # print(len(x))
    n = len(x)//s
    res = [' '.join(x[j*s:(j+1)*s ]) for j in range(n)]
    if len(x)%s>10:
        res.append(' '.join(x[-(len(x)%s):]))
    # print(len(res))
    return res
    
def segment_text(spark, sqlContext, psc):

    import numpy as np
    
    # convert spark dataframe to pandas
    dfs = psc.toPandas()
    
    # segment notes into short chunks by length 318
    dfs['tx'] = dfs['TEXT'].apply(lambda r: combstr(r, 318), 1)
    
    newvalues = np.dstack((np.repeat(dfs.SUBJECT_ID.values, list(map(len, dfs.tx.values))), 
                    np.repeat(dfs.HADM_ID.values, list(map(len, dfs.tx.values))),
                    np.repeat(dfs.LABEL.values, list(map(len, dfs.tx.values))),
                    np.concatenate(dfs.tx.values)))

    df = pd.DataFrame(data=newvalues[0],columns=['SUBJECT_ID', 'ID', 'Label', 'TEXT'])
    
    # update datatypes
    df['Label'] = df['Label'].astype('int')
    df['ID'] = df['ID'].astype('int')
    df['SUBJECT_ID'] = df['SUBJECT_ID'].astype('int')
    
    print('Label count...')
    display(df.groupby('Label')['ID'].nunique())
    
    # save dataframe
    df.to_csv('comb_large_cleaned_with_segment.csv', index = None)
    
    return df
    
def class_balance(df, nDays):
#  Reference: https://github.com/kexinhuang12345/clinicalBERT/blob/master/preprocess.py

    adm = df[['ID', 'Label']].drop_duplicates()
    
    # need rewrite
    pos_ind = adm[adm.Label == 1].ID 
    neg_ind = adm[adm.Label == 0].ID 
    neg_ind_use = neg_ind.sample(n=len(pos_ind), random_state=1)
    print('Total Negative, Positive cases')
    print(len(neg_ind_use), len(pos_ind))

    id_val_test_t=pos_ind.sample(frac=0.2,random_state=1)
    id_val_test_f=neg_ind_use.sample(frac=0.2,random_state=1)
    
    print('Validation Set for Negative, Positive cases')
    print(len(set(id_val_test_t)), len(set(id_val_test_f)))

    id_train_t =  pos_ind.drop(id_val_test_t.index)
    id_train_f =  neg_ind_use.drop(id_val_test_f.index)
    
    print('Training Set for Negative, Positive cases')
    print(len(id_train_f), len(id_train_t))

    id_val_t=id_val_test_t.sample(frac=0.5,random_state=1)
    id_test_t=id_val_test_t.drop(id_val_t.index)

    id_val_f=id_val_test_f.sample(frac=0.5,random_state=1)
    id_test_f=id_val_test_f.drop(id_val_f.index)
    
    print('check overlap:', (pd.Index(id_test_t).intersection(pd.Index(id_train_t))).values)

    id_test = pd.concat([id_test_t, id_test_f])
    test_id_label = pd.DataFrame(data = list(zip(id_test, [1]*len(id_test_t)+[0]*len(id_test_f))), columns = ['id','label'])

    id_val = pd.concat([id_val_t, id_val_f])
    val_id_label = pd.DataFrame(data = list(zip(id_val, [1]*len(id_val_t)+[0]*len(id_val_f))), columns = ['id','label'])

    id_train = pd.concat([id_train_t, id_train_f])
    train_id_label = pd.DataFrame(data = list(zip(id_train, [1]*len(id_train_t)+[0]*len(id_train_f))), columns = ['id','label'])

    id_test = pd.concat([id_test_t, id_test_f])
    test_id_label = pd.DataFrame(data = list(zip(id_test, [1]*len(id_test_t)+[0]*len(id_test_f))), columns = ['id','label'])

    id_val = pd.concat([id_val_t, id_val_f])
    val_id_label = pd.DataFrame(data = list(zip(id_val, [1]*len(id_val_t)+[0]*len(id_val_f))), columns = ['id','label'])

    id_train = pd.concat([id_train_t, id_train_f])
    train_id_label = pd.DataFrame(data = list(zip(id_train, [1]*len(id_train_t)+[0]*len(id_train_f))), columns = ['id','label'])

    # get discharge train/val/test
    df_train = df[df.ID.isin(train_id_label.id)]
    df_val = df[df.ID.isin(val_id_label.id)]
    df_test = df[df.ID.isin(test_id_label.id)]

    print('check train/val/test data size:', len(df_train), len(df_val), len(df_test))
    
    print('Training data set...')
    print(df_train.head())

    print('check if balanced:')
    print(df_train['Label'].value_counts())

    x = pd.concat([neg_ind_use, neg_ind]).drop_duplicates(keep=False)
    
    print('check overlap:', (pd.Index(x).intersection(pd.Index(neg_ind_use))).values)
    print('oversample negative cases...')
    
    # get more negative samples
    if nDays == 30:
        nSamples = 500
    else:
        nSamples = 1000
    
    neg_ind_touse = x.sample(n=nSamples, random_state=1)
    
    df_train = pd.concat([df[df.ID.isin(neg_ind_touse)], df_train])

    # shuffle
    df_train = df_train.sample(frac=1, random_state=1).reset_index(drop=True)
    
    # drop column SUBJECT_ID
    df_train.drop(columns='SUBJECT_ID', inplace=True)
    df_val.drop(columns='SUBJECT_ID', inplace=True)
    df_test.drop(columns='SUBJECT_ID', inplace=True)

    # check if balanced
    print('check class balance:')
    print(df_train.Label.value_counts())
        
    return df_train, df_val, df_test

def get_metric(predictions):
    
    # classification metrics
    eva  = BinaryClassificationEvaluator()

    # calculate AUC
    auc = eva.evaluate(predictions, {eva.metricName: 'areaUnderROC'})
    print('AUROC: %0.3f' % auc)
    
    aucpr = eva.evaluate(predictions, {eva.metricName: 'areaUnderPR'})
    print('AUCPR: %0.3f' % aucpr)
    
    # compute TN, TP, FN, and FP
    predictions.groupBy('label', 'prediction').count().show()
    
    # Calculate the elements of the confusion matrix
    TN = predictions.filter('prediction = 0 AND label = prediction').count()
    TP = predictions.filter('prediction = 1 AND label = prediction').count()
    FN = predictions.filter('prediction = 0 AND label <> prediction').count()
    FP = predictions.filter('prediction = 1 AND label <> prediction').count()
    
    # calculate accuracy, precision, recall, and F1-score
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F =  2 * (precision*recall) / (precision + recall)
    
    print('precision: %0.3f' % precision)
    print('recall: %0.3f' % recall)
    print('accuracy: %0.3f' % accuracy)
    print('F1 score: %0.3f' % F)
    
def plot_roc(pipelineFit):
    plt.figure(figsize=(5,5))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(pipelineFit.stages[-1].summary.roc.select('FPR').collect(),
          pipelineFit.stages[-1].summary.roc.select('TPR').collect())
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    
