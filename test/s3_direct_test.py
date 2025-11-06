import os
import sys
from pyspark.sql import SparkSession
from datetime import datetime

# --- Configuration Constants (Using hardcoded bucket name for this specific test) ---
S3_BUCKET_NAME = "diab-readmit-123456-datamart"
TEST_PATH = f"s3a://{S3_BUCKET_NAME}/_sanity/range_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Retrieve credentials from environment set by docker-compose
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1") # Fallback to a region if not set

# --- Check Prerequisites ---
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    print("FATAL ERROR: AWS credentials not found in environment variables.")
    print("Please ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set on the host machine before running docker-compose up.")
    sys.exit(1)

print(f"--- Starting S3 Connectivity Check for bucket: {S3_BUCKET_NAME} ---")
print(f"Attempting to write to: {TEST_PATH}")

try:
    # 1. Initialize SparkSession using 'spark.jars.packages' to load S3 dependencies
    # We use a stable version of the hadoop-aws package that matches your pyspark version (3.5.5)
    # The 's3a' implementation class is automatically loaded when this package is included.
    spark = (
        SparkSession.builder.appName("S3Check")
        .master("local[*]")
        
        # --- FIX: Use packages to reliably load the S3 dependencies ---
        # The hadoop-aws JAR depends on aws-java-sdk-bundle
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.639")
        
        # Explicitly pass credentials for s3a
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
        
        .getOrCreate()
    )
    
    # Set log level to reduce console spam
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Perform the write operation
    spark.range(5).write.mode("overwrite").parquet(TEST_PATH)
    
    # 3. Verify by reading back
    df_read = spark.read.parquet(TEST_PATH)

    print(f"\n✅ SUCCESS: Wrote and verified {df_read.count()} rows to S3 path:\n{TEST_PATH}")
    print("PySpark S3 connection is working correctly!")

    spark.stop()

except Exception as e:
    print("\n❌ FATAL FAILURE: S3 connection failed.")
    print(f"Error details: {e}")
    print("\nTroubleshooting: 1. Verify credentials (keys, permissions, expiration). 2. Confirm the S3 bucket exists. 3. Ensure the Docker container has network access to Maven repositories (to download packages) and AWS S3.")
    sys.exit(1)
