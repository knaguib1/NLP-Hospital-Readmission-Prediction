import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline 

from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import col, udf, regexp_replace, regexp_extract, lower, col, trim, split, explode, length
from pyspark.sql.types import IntegerType,  StructType, StructField, StringType
from pyspark.sql import functions as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from datetime import *

def extract_readmission(client, nDays=30):
    """MIMIC III readmission data using DIAGNOSES_ICD, NOTEEVENTS, AND PROCEDURES_ICD"""
    
    sSQL = """ 
        with adm as (
        SELECT
            SUBJECT_ID, 
            HADM_ID, 
            FIRST_ADMITTIME, 
            ADMITTIME, 
            DISCHTIME,
            DEATHTIME, 
            ADMISSION_TYPE, -- GENDER,AGE,EXPIRE_FLAG 
            DURATION,
            case 
                when NEXT_ADMITTIME is null then lead(NEXT_ADMITTIME) over (partition by SUBJECT_ID order by ADMITTIME) 
                else NEXT_ADMITTIME 
            end as NEXT_ADMITTIME,
            case
                when NEXT_ADMISSION_TYPE  is null then lead(NEXT_ADMISSION_TYPE) over (partition by SUBJECT_ID order by ADMITTIME) 
                else NEXT_ADMISSION_TYPE  
            end as NEXT_ADMISSION_TYPE 
        from 
            (
                SELECT 
                    SUBJECT_ID, 
                    HADM_ID, 
                    FIRST_ADMITTIME,
                    ADMITTIME,
                    DISCHTIME,
                    DEATHTIME,
                    ADMISSION_TYPE, -- GENDER,AGE,EXPIRE_FLAG
                    DURATION,
                    --   NEXT_ADMITTIME,NEXT_ADMISSION_TYPE
                    case 
                        when NEXT_ADMISSION_TYPE = 'ELECTIVE' then null  
                        else NEXT_ADMISSION_TYPE 
                    end as NEXT_ADMISSION_TYPE,
                    case 
                        when NEXT_ADMISSION_TYPE = 'ELECTIVE' then null  
                        else NEXT_ADMITTIME 
                    end as NEXT_ADMITTIME
                FROM (
                        SELECT  
                            SUBJECT_ID, 
                            HADM_ID, 
                            FIRST_ADMITTIME,
                            ADMITTIME, 
                            DISCHTIME,
                            DEATHTIME,
                            ADMISSION_TYPE, -- GENDER,AGE,EXPIRE_FLAG
                            lead(ADMITTIME) over (partition by SUBJECT_ID order by ADMITTIME) as NEXT_ADMITTIME,
                            lead(admission_type) over (partition by SUBJECT_ID order by ADMITTIME) as NEXT_ADMISSION_TYPE,
                            DATE_DIFF(dischtime, Admittime, DAY)  as DURATION

                        FROM
                            (
                                select distinct 
                                    SUBJECT_ID,
                                    HADM_ID,
                                    FIRST_ADMITTIME,
                                    ADMITTIME,
                                    DISCHTIME,
                                    DEATHTIME,
                                    ADMISSION_TYPE, -- GENDER,AGE,EXPIRE_FLAG
                                from 
                                    `nlp-332020.readmission_dataset.cleaned_dataset`) a
                    ) b
            ) c 
          WHERE 
            admission_type!='NEWBORN' AND deathtime is null 
        ),
            
        notes as (
                    SELECT
                        SUBJECT_ID,
                        HADM_ID, 
                        CHARTDATE, 
                        TEXT
                   FROM 
                        (
                            SELECT
                                SUBJECT_ID,
                                HADM_ID,
                                CHARTDATE, 
                                TEXT,  
                                row_number() over (partition by SUBJECT_ID, HADM_ID order by CHARTDATE DESC) as rk
                            FROM
                              (
                                SELECT DISTINCT 
                                    SUBJECT_ID, 
                                    HADM_ID,  
                                    CHARTDATE, 
                                    TEXT
                               FROM 
                                    `nlp-332020.readmission_dataset.cleaned_dataset`
                                    ) tmp
                            ) tmp1
        WHERE 
            rk=1
            )

        select 
            adm.SUBJECT_ID, 
            adm.HADM_ID, --  GENDER,AGE,
            FIRST_ADMITTIME,
            ADMITTIME,
            DISCHTIME,
            DEATHTIME,
            ADMISSION_TYPE, --EXPIRE_FLAG,
            DURATION,
            NEXT_ADMITTIME,
            NEXT_ADMISSION_TYPE,
            DATE_DIFF( NEXT_ADMITTIME, DISCHTIME, DAY) as DAYS_NEXT_ADMIT,
            case 
                when DATE_DIFF( NEXT_ADMITTIME, DISCHTIME, DAY) < {nDays} then 1 
                else 0 
            end as OUTPUT_LABEL,
            TEXT
        FROM
            adm
        LEFT JOIN 
            notes on adm.SUBJECT_ID = notes.SUBJECT_ID AND adm.HADM_ID = notes.HADM_ID
        where 1=1
            AND TEXT is not null
    """.format(nDays=nDays)
    
    # execute and return big query as dataframe
    return client.query(sSQL).to_dataframe()
    
def comb_labels(df, nDays=30):

    # create labels for readmission dates
    df['LABEL'] = df['DAYS_NEXT_ADMIT'].apply(lambda r: 1 if r <= nDays else 0)
    
    # keep select columns 
    df = df[['SUBJECT_ID', 'HADM_ID',  'LABEL',  'TEXT' ]]
    
    return df
    
def df_to_sparkDF(spark, df, s=11000):
    """
        Convert pandas dataframe to spark dataframe
    """

    # s: chunk size
    # df: pandas df
    sz = len(df)
    
    n = math.ceil(sz/s)
    
    print('will build chunks:', n) 
    
    psc = spark.createDataFrame(df.iloc[:s])
    
    for j in range(1, n):
        chunk = df.iloc[j*s:(j+1)*s]
    
        print(chunk.shape)
        tmp = spark.createDataFrame(chunk)
        psc = psc.union(tmp)
    
    return psc
    
def clean_text(psc):
    
    psc = psc.fillna(' ', subset=['TEXT'])
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], "\\n", " "))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], "\\r", " "))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], "\/", " "))
    psc = psc.withColumn("TEXT", trim(col("TEXT")))
    psc = psc.withColumn("TEXT", lower(col("TEXT")))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], '\\[(.*?)\\]','')) 
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], '[0-9]+\.',''))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], 'dr\.','doctor'))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], 'm\.d\.','md'))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], 'admission date:',''))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], 'discharge date:',''))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], '--|__|==',''))
    psc = psc.withColumn("TEXT", trim(col("TEXT")))
    psc = psc.withColumn("TEXT", regexp_replace(psc['TEXT'], '\\s+',' '))
    
    return psc
    
def readmission_etl(spark, client, nDays=30, s=11000):
    """
        ETL script for extracting MIMIC III readmission data 
        using DIAGNOSES_ICD, NOTEEVENTS, AND PROCEDURES_ICD
    """
    
    print('Extracting readmission data using BigQuery...', datetime.now())
    
    # extract data from big query
    df = extract_readmission(client, nDays=nDays)
    
    print('Completed data extraction... performing etl...', datetime.now())
    
    # create labels for readmission dates
    df['LABEL'] = df['DAYS_NEXT_ADMIT'].apply(lambda r: 1 if r<nDays else 0)
    
    # keep select columns 
    df = df[['SUBJECT_ID', 'HADM_ID',  'LABEL',  'TEXT' ]]
    
    print('Converting pandas to spark DF...', datetime.now())
    
    # pandas to spark df
    psc = df_to_sparkDF(spark, df, s=11000)
    
    print('Cleaning text from Note Events...', datetime.now())
    
    # clean text data
    psc = clean_text(psc)
    
    print('Completed etl script...', datetime.now())
    
    return psc
   
def save_sparkDF(spark, psc, nDays):
    
    newschema = [StructField('SUBJECT_ID', IntegerType(), False), 
             StructField('HADM_ID', IntegerType(), False),
             StructField('LABEL', IntegerType(), False),
             StructField('TEXT', StringType(), False)]
             
    psc = spark.createDataFrame(psc.rdd, StructType(fields=newschema))
    
    file_name = 'readmit_' + str(nDays) + '.parquet'
    
    psc.write.parquet(file_name)

