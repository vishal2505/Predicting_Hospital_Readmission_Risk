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


def process_labels_gold_table(snapshot_date_str, silver_diabetes_monthly_directory, gold_label_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    target_year = snapshot_date.year
    target_month = snapshot_date.month
    
    # connect to silver table
    partition_name = f"silver_diabetes_monthly_{target_year}_{target_month:02d}" + '.parquet'
    filepath = silver_diabetes_monthly_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get label
    df = df.withColumn("label", F.when(F.col("readmitted") == "<30", 1).otherwise(0).cast(IntegerType()))
    # df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    # df = df.select("encounter_id", "label", "label_def", "snapshot_date")
    df = df.select("encounter_id", "label", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = f"gold_label_store_{target_year}_{target_month:02d}" + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(snapshot_date_str, silver_diabetes_monthly_directory, gold_feature_store_directory, spark, mob = 0):

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    target_year = snapshot_date.year
    target_month = snapshot_date.month
    
    # connect to silver table
    partition_name = f"silver_diabetes_monthly_{target_year}_{target_month:02d}" + '.parquet'
    filepath = silver_diabetes_monthly_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

     # --- 🔧 Feature Engineering: Race ---
    # Standardize missing or unknown values
    df = df.withColumn(
        "race_cleaned",
        F.when(F.col("race").isin("?", "Unknown", None), "Unknown").otherwise(F.col("race"))
    )

    # One-hot encoding (manual binary flags for interpretability)
    df = (
        df.withColumn("race_AfricanAmerican", F.when(F.col("race_cleaned") == "AfricanAmerican", 1).otherwise(0))
          .withColumn("race_Caucasian", F.when(F.col("race_cleaned") == "Caucasian", 1).otherwise(0))
          .withColumn("race_Asian",F.when((F.col("race_cleaned") == "Asian") | (F.col("race_cleaned") == "Unknown"),1).otherwise(0))
          .withColumn("race_Hispanic", F.when(F.col("race_cleaned") == "Hispanic", 1).otherwise(0))
    )

    # --- 🔧 Feature Engineering: Gender ---
    df = df.withColumn("is_female", F.when(F.col("gender") == "Female", 1).otherwise(0))

    # --- 🔧 Feature Engineering: Age ---
    # Map age ranges to midpoints
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
        "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95
    }

    mapping_expr = F.create_map([F.lit(x) for kv in age_map.items() for x in kv])

    df = df.withColumn("age_midpoint", mapping_expr.getItem(F.col("age")).cast(IntegerType()))

    # Age Group Encoding (Ordinal Category)
    age_order = {
    "[0-10)": 1, "[10-20)": 2, "[20-30)": 3, "[30-40)": 4, "[40-50)": 5,
    "[50-60)": 6, "[60-70)": 7, "[70-80)": 8, "[80-90)": 9, "[90-100)": 10
    }
    age_order_expr = F.create_map([F.lit(x) for kv in age_order.items() for x in kv])
    df = df.withColumn("age_group_index", age_order_expr.getItem(F.col("age")).cast(IntegerType()))

    # Age Group Labeling (Broad Category)
    df = df.withColumn(
    "age_group_label",
    F.when(F.col("age").isin("[0-10)", "[10-20)", "[20-30)"), "Young")
     .when(F.col("age").isin("[30-40)", "[40-50)", "[50-60)", "[60-70)"), "Adult")
     .when(F.col("age").isin("[70-80)", "[80-90)", "[90-100)"), "Senior")
     .otherwise("Unknown")
    )

    # # --- 🔧 Feature Engineering: Weight ---

    # # Map weight ranges to midpoints
    # weight_map = {
    #     "[0-25)": 12.5, "[25-50)": 37.5, "[50-75)": 62.5, "[75-100)": 87.5,
    #     "[100-125)": 112.5, "[125-150)": 137.5, "[150-175)": 162.5,
    #     "[175-200)": 187.5, ">200": 225
    # }
    # mapping_expr = F.create_map([F.lit(x) for kv in weight_map.items() for x in kv])
    # df = df.withColumn("weight_midpoint", mapping_expr.getItem(F.col("weight")).cast(IntegerType()))

    # # Weight Category (Low / Normal / High / Very High)
    # df = df.withColumn(
    # "weight_group_label",
    # F.when(F.col("weight").isin("[0-25)", "[25-50)"), "Underweight")
    #  .when(F.col("weight").isin("[50-75)", "[75-100)"), "Normal")
    #  .when(F.col("weight").isin("[100-125)", "[125-150)"), "Overweight")
    #  .when(F.col("weight").isin("[150-175)", "[175-200)", ">200"), "Obese")
    #  .otherwise("Unknown")
    # )

    # # Health Risk Flag
    # df = df.withColumn(
    #     "is_obese",
    #     F.when(F.col("weight").isin("[150-175)", "[175-200)", ">200"), 1).otherwise(0)
    # )

    # --- 🔧 Feature Engineering: admission_type_id ---
    mapping = {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Not Available",
        6: "Unknown",
        7: "Trauma Center",
        8: "Not Mapped"
    }

    mapping_expr = F.create_map([F.lit(x) for kv in mapping.items() for x in kv])
    df = df.withColumn("admission_type_desc", mapping_expr.getItem(F.col("admission_type_id")).cast(StringType()))

    severity_map = {
        "Emergency": 3,
        "Urgent": 2,
        "Trauma Center": 3,
        "Elective": 1,
        "Newborn": 1,
        "Not Available": 0,
        "Unknown": 0,
        "Not Mapped": 0
    }
    severity_expr = F.create_map([F.lit(x) for kv in severity_map.items() for x in kv])
    df = df.withColumn("admission_severity_score", severity_expr.getItem(F.col("admission_type_desc")).cast(IntegerType()))


    # --- 🔧 Feature Engineering: discharge_disposition_id ---
    # Discharge Severity Score (1=Low Risk, 5=High Risk, 0=Unknown)
    severity_map = {
        1: 1, 6: 1, 8: 1,
        2: 2, 3: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 2, 22: 2, 23: 2, 24: 2, 27: 2, 28: 2, 29: 2, 30: 2, 10: 2, 12: 2,
        13: 3, 14: 3, 9: 3,
        7: 4,
        11: 5, 19: 5, 20: 5, 21: 5,
        18: 0, 25: 0, 26: 0
    }
    severity_expr = F.create_map([F.lit(x) for kv in severity_map.items() for x in kv])
    df = df.withColumn("discharge_severity_score", severity_expr.getItem(F.col("discharge_disposition_id")).cast(IntegerType()))

    # --- 🔧 Feature Engineering: admission_source_id ---
    risk_map = {
        1:1, 2:1, 3:1,
        7:2, 8:2,
        4:3, 5:3, 6:3, 10:3, 18:3, 22:3, 25:3, 26:3,
        11:1, 12:1, 13:1, 14:1, 23:1, 24:1,
        19:4,
        9:0, 15:0, 17:0, 20:0, 21:0
    }
    risk_expr = F.create_map([F.lit(x) for kv in risk_map.items() for x in kv])
    df = df.withColumn("admission_source_risk_score", risk_expr.getItem(F.col("admission_source_id")).cast(IntegerType()))

    # --- 🔧 Feature Engineering: time_in_hospital, num_medications, num_lab_procedures, num_procedures, number_diagnoses ---
    df = df.withColumn(
    "medication_intensity",
    F.when(F.col("num_medications") > 0, 
           (F.col("time_in_hospital") / F.col("num_medications")).cast(FloatType())) 
     .otherwise(None)
   )
    
    df = df.withColumn(
        "diagnosis_density",
        F.when(
            F.col("number_diagnoses") > 0,
            ((F.col("num_lab_procedures") + F.col("num_procedures")) / F.col("number_diagnoses")).cast(FloatType())
        ).otherwise(None)
    )
    
    # --- 🔧 Feature Engineering: payer_code ---
    # risk_map = {
    #     "BC": 4, "CH": 4, "CM": 4, "CP": 4, "HM": 4, "MP": 4, "PO": 4, "SI": 4,
    #     "MC": 3, "MD": 3, "SP": 3, "FR": 3, "OG": 3,
    #     "DM": 3,
    #     "WC": 2,
    #     "UN": 1,
    #     "OT": 0, "?": 0
    # }
    # risk_expr = F.create_map([F.lit(x) for kv in risk_map.items() for x in kv])
    # df = df.withColumn("payer_risk_score", risk_expr.getItem(F.col("payer_code")).cast(IntegerType()))

    # --- 🔧 Feature Engineering: max_glu_serum ---
    max_glu_map = {
        "None": 0,
        "Norm": 1,
        ">200": 2,
        ">300": 3
    }

    # Create a PySpark map expression
    mapping_expr = F.create_map([F.lit(x) for kv in max_glu_map.items() for x in kv])

    # Apply transformation
    df = df.withColumn("max_glu_serum_ord", mapping_expr.getItem(F.col("max_glu_serum")).cast(IntegerType()))

    # --- 🔧 Feature Engineering: A1Cresult ---
    A1Cresult_map = {
        "None": 0,
        "Norm": 1,
        ">7": 2,
        ">8": 3
    }

    # Create a PySpark map expression
    mapping_expr = F.create_map([F.lit(x) for kv in A1Cresult_map.items() for x in kv])

    # Apply transformation
    df = df.withColumn("A1Cresult_ord", mapping_expr.getItem(F.col("A1Cresult")).cast(IntegerType()))

    # --- 🔧 Feature Engineering: Medication Columns ---
    # List of medication columns
    med_cols = [
        "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
        "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
        "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
        "examide","citoglipton","insulin","glyburide-metformin","glipizide-metformin",
        "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone"
    ]

    # Mapping dictionary
    med_map = {"No":0, "Steady":1, "Down":-1, "Up":2}
    mapping_expr = F.create_map([F.lit(x) for kv in med_map.items() for x in kv])

    # Apply mapping to all medication columns
    for col in med_cols:
        df = df.withColumn(f"{col}_ord", mapping_expr.getItem(F.col(col)).cast(IntegerType()))
    
    df = df.withColumn("change", F.when(F.col("change") == "ch", 1).otherwise(0))
    df = df.withColumn("diabetesMed", F.when(F.col("diabetesMed") == "Yes", 1).otherwise(0))

    df = df.withColumn(
        "total_visits",
        F.when(
            F.col("number_diagnoses") > 0,
            ((F.col("number_outpatient") + F.col("number_emergency")) + F.col("number_inpatient")).cast(IntegerType())
        ).otherwise(None)
    )

    df = df.withColumn(
        "poor_glucose_control",
        F.when(
            (F.col("max_glu_serum").isin(">200", ">300")) |
            (F.col("A1Cresult").isin(">7", ">8", ">9")),  # keep ">9" in case it appears
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    df = df.withColumn("severity_x_visits", F.col("discharge_severity_score") * F.col("total_visits"))
    df = df.withColumn("medication_density", F.col("medication_intensity") / (F.col("diagnosis_density") + 1))


    # ,"repaglinide_ord","nateglinide_ord","chlorpropamide_ord","glimepiride_ord","acetohexamide_ord","glipizide_ord","glyburide_ord","tolbutamide_ord","pioglitazone_ord","rosiglitazone_ord","acarbose_ord","miglitol_ord","troglitazone_ord","tolazamide_ord","examide_ord","citoglipton_ord","insulin_ord","glyburide-metformin_ord","glipizide-metformin_ord","glimepiride-pioglitazone_ord","metformin-rosiglitazone_ord","metformin-pioglitazone_ord","change",
    
    # select columns to save
    df = df.select("encounter_id","race_AfricanAmerican","race_Caucasian","race_Asian","race_Hispanic","is_female","age_midpoint","admission_severity_score","admission_source_risk_score", "poor_glucose_control","metformin_ord","insulin_ord","diabetesMed", "severity_x_visits", "medication_density","diag_1", "diag_2", "diag_3" ,"snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = f"gold_feature_store_{target_year}_{target_month:02d}" + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df
    