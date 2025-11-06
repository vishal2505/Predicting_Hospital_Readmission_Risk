import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_table(snapshot_date_str,
                         bronze_diabetes_directory,
                         silver_diabetes_monthly_directory,
                         spark):
    
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    target_year = snapshot_date.year
    target_month = snapshot_date.month

    # ---------------- Diabetes Bronze ----------------
    partition_name = f"bronze_diabetes_monthly_{target_year}_{target_month:02d}.csv"
    filepath = bronze_diabetes_directory + partition_name
    # Read the bronze CSV directory (works for local and s3a). Using wildcard ensures we pick up only data files.
    df = spark.read.option("header", True).option("inferSchema", True).csv(filepath + "/*")
    print("loaded from:", filepath, "row count:", df.count())

    column_type_map = {
    "encounter_id": IntegerType(),                     # unique integer ID for encounter
    "patient_nbr": IntegerType(),                      # integer patient number
    "race": StringType(),                              # categorical (Caucasian, AfricanAmerican, etc.)
    "gender": StringType(),                            # categorical (Male/Female)
    "age": StringType(),                               # range category ([0-10), [10-20), etc.)
    "weight": StringType(),                            # mostly "?" or weight ranges → treat as string
    "admission_type_id": IntegerType(),                # numeric code for admission type
    "discharge_disposition_id": IntegerType(),         # numeric code for discharge disposition
    "admission_source_id": IntegerType(),              # numeric code for admission source
    "time_in_hospital": IntegerType(),                 # integer number of days
    "payer_code": StringType(),                        # code for insurance payer (many missing → string)
    "medical_specialty": StringType(),                 # doctor’s specialty or “?” → string
    "num_lab_procedures": IntegerType(),               # count of lab procedures
    "num_procedures": IntegerType(),                   # count of procedures
    "num_medications": IntegerType(),                  # count of medications
    "number_outpatient": IntegerType(),                # number of outpatient visits
    "number_emergency": IntegerType(),                 # number of emergency visits
    "number_inpatient": IntegerType(),                 # number of inpatient visits
    "diag_1": StringType(),                            # diagnosis code (ICD) → string
    "diag_2": StringType(),                            # diagnosis code (ICD) → string
    "diag_3": StringType(),                            # diagnosis code (ICD) → string
    "number_diagnoses": IntegerType(),                 # total diagnosis count
    "max_glu_serum": StringType(),                     # categorical (“None”, “Norm”, “>200”, etc.)
    "A1Cresult": StringType(),                         # categorical (“None”, “Norm”, “>8”, etc.)
    "metformin": StringType(),                         # categorical (No/Steady/Up/Down)
    "repaglinide": StringType(),                       # same categorical pattern
    "nateglinide": StringType(),
    "chlorpropamide": StringType(),
    "glimepiride": StringType(),
    "acetohexamide": StringType(),
    "glipizide": StringType(),
    "glyburide": StringType(),
    "tolbutamide": StringType(),
    "pioglitazone": StringType(),
    "rosiglitazone": StringType(),
    "acarbose": StringType(),
    "miglitol": StringType(),
    "troglitazone": StringType(),
    "tolazamide": StringType(),
    "examide": StringType(),
    "citoglipton": StringType(),
    "insulin": StringType(),
    "glyburide-metformin": StringType(),
    "glipizide-metformin": StringType(),
    "glimepiride-pioglitazone": StringType(),
    "metformin-rosiglitazone": StringType(),
    "metformin-pioglitazone": StringType(),
    "change": StringType(),                            # categorical (Ch/No)
    "diabetesMed": StringType(),                       # categorical (Yes/No)
    "readmitted": StringType(),                        # categorical (>30, <30, NO)
    "snapshot_date": DateType(),                       # date (e.g., 01-01-1999)
    }
    
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast(new_type))

    df = df.withColumn(
        "max_glu_serum",
        F.when(
            (F.col("max_glu_serum").isNull()) | (F.col("max_glu_serum") == "") | (F.col("max_glu_serum") == "?"),
            "None"
        ).otherwise(F.col("max_glu_serum"))
    )

    df = df.withColumn(
        "A1Cresult",
        F.when(
            (F.col("A1Cresult").isNull()) | (F.col("A1Cresult") == "") | (F.col("A1Cresult") == "?"),
            "None"
        ).otherwise(F.col("A1Cresult"))
    )

    # ---------------- Save silver table ----------------
    partition_name = f"silver_diabetes_monthly_{target_year}_{target_month:02d}" + '.parquet'
    filepath = silver_diabetes_monthly_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df













    
