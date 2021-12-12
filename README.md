# BD4H NLP Final Project
# Hospital readmission prediction from clinical discharge notes
#### Authors: Dan Chen, Kevin Cooper, Kareem Naguib, Patrick O’Brien

### Platform
#### 1. Google Cloud Environment  
Google Big Query and Colab were the technologies used in this project. Below are instructions for setting up your environment.  

Setup instructions:  
1. [Create a new project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) in Google Cloud Platform. You will need to record your project id and project number to connect to Big Query from Colab notebook.
2. [Create a new dataset](https://cloud.google.com/bigquery/docs/quickstarts/quickstart-cloud-console) in Google Big Query - upload relevent MIMIC III datasets. 

#### 2. Spark-NLP for Healthcare (John Snow Labs)
While much of Spark NLP is open source, many of the healthcare-specific
models are only accessible via spark-nlp-jsl, which requires an account and free trial with John Snow Labs. The following will guide you in how to set up your environment to use Spark NLP for Healthcare.   

Spark NLP John Snow Labs (spark-nlp-jsl) step-by-step instructions:
1. Set up an account with [John Snow Labs](https://www.johnsnowlabs.com/)
2. Sign up for a free trial of spark-nlp-jsl
3. On your account page, go to _My Subscription_ where you can download a JSON license file.
4. In the Colab cell that fetches the spark-nlp-jsl license KVP, browse to the location of your JSON license file and select it.
```
# For this step, you need a free trial of the John Snow Labs version (spark-nlp-jsl)
# of Spark NLP. While much of Spark NLP is open source, many of the healthcare-specific
# models are only accessible via spark-nlp-jsl. Sign up for an account and free trial
# at https://www.johnsnowlabs.com/. Step by step instructions found in the README.md
import json
import os
from google.colab import files
license_keys = files.upload()
with open(list(license_keys.keys())[0]) as f:
    license_keys = json.load(f)
# Defining license key-value pairs as local variables
locals().update(license_keys)
# Adding license key-value pairs to environment variables
os.environ.update(license_keys)
```
5. Install spark-nlp public version and spark-nlp-jsl for healthcare-related models
```
# Installing pyspark and spark-nlp
! pip install --upgrade -q pyspark==3.1.2 spark-nlp==$PUBLIC_VERSION
# Installing Spark NLP Healthcare
! pip install --upgrade -q spark-nlp-jsl==$JSL_VERSION  --extra-index-url https://pypi.johnsnowlabs.com/$SECRET
# Installing Spark NLP Display Library for visualization
! pip install -q spark-nlp-display
```

#### 3. Other: Pytorch, PySpark, Pandas, etc.

### Datasets and Cohort  
We use [MIMIC-III](https://mimic.mit.edu/) patient, admission, ICU stay and event notes data.  
Readmission was counted by the day interval between two consecutive admissions. Admission type death was excluded for approach 1 modeling and newborn was also excluded for approach 2 and 3.  


### Approach
#### 1. Utilize SparkNLP streamline  
check this notebook [BD4H_Word_Embeddings_model](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/BD4H_Word_Embeddings_Model.ipynb) for modeling and prediction
 

#### 2. Utilize PySpark Machine Learning Models
#### 2.1 Preprocessing (for both approach 2 and 3)   
Run [src/etl.py](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/src/etl.py) to extract readmission data with ICU notes from MIMIC-III, and clean up the note text by removing special characters, stripping the space, removing numbers and stop-words, etc.   

Run [src/utils.py](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/src/utils.py) to split the notes into 318-word chunks, and subsampled the dataset to create training, validation, and testing data with similar readmission distribution, and balanced the positive and negative chunks.

The final cleaned data file is expected to have columns named as "TEXT", "ID" and "Label", corresponding to note chunks, HADM_ID, readmission label (0 or 1).  

The data folder is expected as following for an example:
```
└───readmit_30
        test.csv
        train.csv
        val.csv
```
#### 2.2 Modeling
Notebook [readmit30_train_pred](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/src/readmit30_train_pred.ipynb) runs Logistic Regression, Random Forest, Naive Bayes and Gradient-Boosted Trees on the data with both TF-IDF and Word2Vec tokens and evaluate the prediction with AUROC, AUCPR and F1-score using PySpark Machine Learning modules.


[This notebook](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/src/notebook.ipynb) runs a spark_nlp_model on the data using spark-nlp-jsl (setup as Approach 1).


#### 3. Utilize pretrained [ClinicalBERT](https://github.com/kexinhuang12345/clinicalBERT.git)[1] weights
#### 3.1 Installation and Requirement
Download the pretrained ClinicalBERT weights from [this link](https://drive.google.com/file/d/1t8L9w-r88Q5-sfC993x2Tjt1pu--A900/view) and unzip it. It's expected to have the following structure:
```
├───model
│   ├───discharge_readmission
│   │       bert_config.json
│   │       pytorch_model.bin
│   │
│   ├───early_readmission
│   │       bert_config.json
│   │       pytorch_model.bin
│   │
│   └───pretraining
│           bert_config.json
│           pytorch_model.bin
│           vocab.txt
```
We will use the pytorch model from the discharge_readmission folder.  
Install corresponding packages:
```
pip install funcsigs
pip install pytorch-pretrained-bert
git clone https://github.com/kexinhuang12345/clinicalBERT.git
```
#### 3.2 Prediction
In order to run the code below please put the cleaned data folder, for example readmit_30, under the src folder  
Run prediction for 30-day readmission using above discharge_readmission model
```
python src/clinicalBERT/run_readmission.py \
  --task_name readmission \
  --readmission_mode discharge \
  --do_eval \
  --data_dir readmit_30/ \
  --bert_model  clinicalBERT/model/discharge_readmission \
  --max_seq_length 512 \
  --output_dir ./result_readmit30
```
Retrain the model for 180-day readmission from pretraining ClinicalBERT and do prediction
```
python src/clinicalBERT/run_readmission.py \
  --task_name readmission \
  --readmission_mode discharge \
  --do_train 12 \
  --do_eval \
  --eval_batch_size 16 \
  --data_dir readmit_180/ \
  --bert_model  discharge180_mymodel/ \
  --max_seq_length 512 \
  --output_dir ./result_discharge180_mymodel
```

### Result
Explainatory analysis on readmission, admission type, age, gender, note length, etc. could be find at this notebook [readmit_explainatory_analysis](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/readmit_explainatory_analysis.ipynb)

Readmission within 30 days prediction and ROC and PR curves can be find at [result_readmit30](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/src/result_readmit30)

| Model	| AUROC |	AUPRC |	F1-Score |
| ------------- |:-------------:|:-------------:|:-------------:|
| TF-IDF with Logistic Regression with ElasticNet|	0.63|	0.629|	0.596|
|TF-IDF with Random Forest|	0.642|	0.642|	0.617|
|ClinicalBERT|	0.813|	0.79||	
|ClinicalBERT (Paper Results[1])|	0.714|	0.701||	
|CNN Exact Text Match (unbalanced data)	0.463	|	0.471|
|Word2Vec with Logistic Regression with L2 Regularization|	0.636|	0.637|	0.61|
|Word2Vec with Random Forest	|0.62|	0.624|	0.606|


Readmission within 6 months(180 days) prediction and ROC and PR curves can be find at [result_discharge180](https://github.gatech.edu/kcooper72/bd4h_nlp_final_project/blob/main/src/discharge180_mymodel)

|Model|	AUROC|	AUPRC|	F1-Score|
| ------------- |:-------------:|:-------------:|:-------------:|
|Clinical Word Embeddings|	0.5|	0.547|	|
|TF-IDF with Logistic Regression with L2 Regularization	|0.672|	0.691|	0.631|
|TF-IDF with Random Forest|	0.702|	0.72|	0.664|
|ClinicalBERT|0.758|	0.75 | |	

Our retrained model for 6 month performs worse than the model in [1] so here we keep the result from original ClinicalBERT model. Reason may due to our limited computation resources, which is only able to run 12 batch size and one iteration takes about 1hr.

### Reference
[1] Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. ClinicalBERT: Modeling CLinical Notes and Predicting Hospital Readmission. arXiv:1904.05342, 2019.