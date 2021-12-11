# BD4H NLP Final Project
# Hospital re-admission prediction from clinical discharge notes
### Dan Chen, Kevin Cooper, Kareem Naguib, Patrick O’Brien

#### 1. Google Cloud Environment Setup
Google Big Query and Colab were the technologies used in this project. Below are instructions for setting up your environment. 
1. [Create a new project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) in Google Cloud Platform. You will need to record your project id and project number to connect to Big Query from Colab notebook.
2. [Create a new dataset](https://cloud.google.com/bigquery/docs/quickstarts/quickstart-cloud-console) in Google Big Query - upload relevent MIMIC III datasets. 

#### 2. Spark-NLP for Healthcare (John Snow Labs) Setup
While much of Spark NLP is open source, many of the healthcare-specific
models are only accessible via spark-nlp-jsl, which requires an account and free trial with John Snow Labs. The following will guide you in how to set up your environment to use
Spark NLP for Healthcare.

Spark NLP John Snow Labs (spark-nlp-jsl) step-by-step instructions
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
