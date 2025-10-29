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
from pyspark.sql.functions import col, month, year


def process_bronze_table_diabetes(snapshot_date_str, bronze_loan_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    target_year = snapshot_date.year
    target_month = snapshot_date.month
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/diabetic_data_with_snapshot.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter((year(col('snapshot_date')) == target_year) & (month(col('snapshot_date')) == target_month))
    print(snapshot_date_str + ' row count:', df.count())

    # save bronze table to datamart as a single-part CSV (directory with one part file)
    partition_name = f"bronze_diabetes_monthly_{target_year}_{target_month:02d}.csv"
    filepath = bronze_loan_directory + partition_name
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(filepath)
    print('saved to:', filepath)

    return df