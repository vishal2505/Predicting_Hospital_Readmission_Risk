# Model Inference Pipeline

**Date:** November 4, 2025  
**Status:** ✅ COMPLETED  

---

## Overview

Automated model inference pipeline that generates predictions for a specific snapshot date using the best trained model. The pipeline matches the notebook workflow (`model_inference.ipynb`) and includes comprehensive prerequisite checks.

**Key Difference from Training:** This pipeline does NOT load a separate scaler - the model artifact already contains the preprocessing transformers (scaler) that were saved during training.

---

## 📋 Architecture

### Pipeline Flow

```
Prerequisites (Sequential Checks)
    ↓
├─→ Check Trained Model Exists
├─→ Check Model Comparison Exists (optional)
├─→ Check Feature Store Data Exists
    ↓
Run Model Inference (with snapshot_date)
    ↓
├─→ Load Model Artifact (contains model + scaler)
├─→ Load Feature Store for Snapshot Date
├─→ Preprocess with Scaler from Artifact
├─→ Generate Predictions
└─→ Save Predictions to S3
```

### Design Principles

1. **Notebook-Based**: Follows exact flow from `model_inference.ipynb`
2. **Snapshot-Based**: Processes data for a specific snapshot date (year/month)
3. **Self-Contained Models**: Model artifacts include preprocessing transformers
4. **Best Model Selection**: Automatically uses recommended model from comparison
5. **Dynamic Date Support**: Snapshot date can be passed per DAG run

---

## 🎯 Components

### 1. Inference Script (`model_inference.py`)

**Purpose:** Generate predictions for a specific snapshot date using trained model artifact

**Key Functions:**

#### `get_best_model_info(config)`
- Loads `latest_model_comparison.json` from S3
- Returns recommended model algorithm name
- Fallback: Uses XGBoost if comparison not found
- Example: `"xgboost"`, `"random_forest"`, `"logistic_regression"`

#### `load_model_from_s3(algorithm, config)`
- Downloads `model.pkl` (model artifact) from S3 model registry
- Model artifact contains:
  - `model_artifact["model"]` - Trained model
  - `model_artifact["preprocessing_transformers"]["stdscaler"]` - Fitted scaler
- Loads metadata (training date, feature count)
- Returns: `(model_artifact, metadata)`

#### `load_feature_store_for_snapshot(snapshot_date)`
- Loads ALL feature store data from `gold/feature_store/`
- Filters by snapshot_date (year and month)
- Example: `"2008-03-01"` → filters for March 2008
- Returns: pandas DataFrame with features for that month
- Columns: `encounter_id`, `snapshot_date`, feature columns

#### `preprocess_and_predict(features_df, model_artifact, algorithm, snapshot_date)`
- Extracts feature columns (excludes: `encounter_id`, `snapshot_date`, `label`, `partition_date`)
- Applies scaler from model artifact: `model_artifact["preprocessing_transformers"]["stdscaler"]`
- Generates predictions using: `model_artifact["model"]`
- Returns: DataFrame with `encounter_id`, `snapshot_date`, `model_algorithm`, `model_predictions`, `prediction_timestamp`

#### `save_predictions_to_s3(predictions_df, algorithm, snapshot_date)`
- Saves predictions as parquet to S3
- Path pattern: `gold/model_predictions/{algorithm}/{algorithm}_predictions_{snapshot_date}.parquet`
- Example: `gold/model_predictions/xgboost/xgboost_predictions_2008_03_01.parquet`
- Saves metadata with statistics (count, avg probability)

---

### 2. Airflow DAG (`diab_model_inference.py`)

**DAG Name:** `diab_model_inference`  
**Schedule:** Manual trigger (or scheduled as needed)  
**Timeout:** 30 minutes

**Configuration:**
- Default snapshot date: `2008-03-01` (can be overridden)
- Can pass custom date via DAG run config: `{"snapshot_date": "2008-06-01"}`

**Prerequisite Checks:**

#### Check 1: Trained Model Exists
- Verifies at least one trained model in model registry
- Checks: `model_registry/{algorithm}/latest/model.pkl`
- **Critical:** Pipeline stops if no model found

#### Check 2: Model Comparison Exists (Optional)
- Checks for `latest_model_comparison.json`
- Used to identify best model
- **Non-blocking:** Pipeline proceeds even if not found (uses default)

#### Check 3: Feature Store Data Exists
- Verifies feature data in `gold/feature_store/`
- **Critical:** Pipeline stops if no data found
- Note: Does NOT check for specific snapshot date, just verifies data exists

**Inference Task:**
- Uses same ECS task definition as training (2vCPU/4GB)
- Runs `model_inference.py` script
- Environment variables:
  - `AWS_REGION`: AWS region
  - `DATAMART_BASE_URI`: S3 datamart location
  - `MODEL_CONFIG_S3_URI`: Model config path
  - `SNAPSHOT_DATE`: Snapshot date for predictions (format: YYYY-MM-DD)
    - Can be passed via DAG config: `{"snapshot_date": "2008-06-01"}`
    - Defaults to `2008-03-01` if not provided

---

## 📊 Data Flow

### Input Data

```
s3://bucket/gold/
├── feature_store/                    # Feature data (ALL snapshots)
│   └── partition_date=YYYY-MM-DD/   # Multiple partitions
│       └── *.parquet                # Filtered by snapshot_date during inference
└── model_registry/                   # Trained models
    ├── {algorithm}/latest/
    │   ├── model.pkl                # Contains model + scaler artifact
    │   └── metadata.json
    └── latest_model_comparison.json
```

**Note:** The scaler is NOT loaded separately - it's included in the model.pkl artifact as:
```python
model_artifact = {
    "model": <trained_model>,
    "preprocessing_transformers": {
        "stdscaler": <fitted_StandardScaler>
    }
}
```

### Output Data

```
s3://bucket/gold/model_predictions/
└── {algorithm}/
    ├── {algorithm}_predictions_2008_03_01.parquet    # Predictions for specific snapshot date
    ├── {algorithm}_predictions_2008_06_01.parquet    # Another snapshot date
    ├── metadata_2008_03_01.json                      # Prediction metadata
    └── metadata_2008_06_01.json
```

**Predictions DataFrame Schema:**
```
encounter_id          | string  | Patient encounter ID
snapshot_date         | date    | Snapshot date from feature store
model_algorithm       | string  | Algorithm used (e.g., "xgboost")
model_predictions     | float   | Prediction probability (0.0 to 1.0)
prediction_timestamp  | string  | When prediction was generated
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
AWS_REGION=ap-southeast-1
DATAMART_BASE_URI=s3://bucket/
MODEL_CONFIG_S3_URI=s3://bucket/config/model_config.json

# Snapshot Date (3 ways to set)
# 1. Via DAG run config (recommended): {"snapshot_date": "2008-06-01"}
# 2. Via environment variable:
DEFAULT_SNAPSHOT_DATE=2008-03-01
# 3. Hardcoded in script (last resort)
SNAPSHOT_DATE=2008-03-01
```

### Model Config (`model_config.json`)

```json
{
  "model_config": {
    "model_registry_bucket": "diab-readmit-123456-datamart",
    "model_registry_prefix": "model_registry/"
  }
}
```

---

## 🚀 Usage

### Option 1: Trigger with Custom Snapshot Date (Recommended)

```bash
# Via Airflow CLI with DAG run config
airflow dags trigger diab_model_inference \
  --conf '{"snapshot_date": "2008-06-01"}'
```

Or via Airflow UI:
1. Navigate to DAGs → `diab_model_inference` → Trigger
2. Add configuration JSON:
```json
{
  "snapshot_date": "2008-06-01"
}
```

### Option 2: Trigger with Default Date

```bash
# Via Airflow UI
# Navigate to DAGs → diab_model_inference → Trigger (no config needed)
# Uses default: 2008-03-01

# Via CLI
airflow dags trigger diab_model_inference
```

### Option 3: Run Inference Locally (for testing)

```bash
# Set environment variables
export AWS_REGION=ap-southeast-1
export DATAMART_BASE_URI=s3://bucket/
export MODEL_CONFIG_S3_URI=s3://bucket/config/model_config.json

# Set snapshot date
export SNAPSHOT_DATE=2008-06-01

# Run inference
python model_inference.py
```

---

## 📈 Predictions Output

### Parquet File Structure

```python
predictions_df.columns = [
    'encounter_id',              # Patient encounter identifier
    'snapshot_date',             # Snapshot date from feature store (YYYY-MM-DD)
    'model_algorithm',           # Algorithm used (e.g., 'xgboost')
    'model_predictions',         # Readmission probability (0-1)
    'prediction_timestamp'       # ISO timestamp when prediction was generated
]
```

### Example Output

**Example Data:**
```
encounter_id  snapshot_date  model_algorithm  model_predictions  prediction_timestamp
-----------  -------------  ---------------  -----------------  ---------------------
12345        2008-03-01     xgboost          0.6842             2025-11-04T10:30:45
23456        2008-03-01     xgboost          0.2153             2025-11-04T10:30:45
34567        2008-03-01     xgboost          0.8291             2025-11-04T10:30:45
```

### Metadata File

```json
{
  "timestamp": "2025-11-04T10:30:45",
  "snapshot_date": "2008-03-01",
  "algorithm": "xgboost",
  "num_predictions": 9882,
  "avg_probability": 0.3456,
  "predictions_path": "s3://bucket/gold/model_predictions/xgboost/xgboost_predictions_2008_03_01.parquet"
}
```

---

## 🔍 Preprocessing Details

### Model Artifact Structure

The `model.pkl` saved during training contains both the model and preprocessing transformers:

```python
model_artifact = {
    "model": <trained_model>,              # XGBoost/RandomForest/LogisticRegression
    "preprocessing_transformers": {
        "stdscaler": <fitted_StandardScaler>  # Already fitted on training data
    }
}
```

### Preprocessing Steps During Inference

```python
# 1. Load model artifact (contains model + scaler)
model_artifact = load_model_from_s3(algorithm, config)

# 2. Extract feature columns
feature_cols = [c for c in df.columns 
                if c not in ["encounter_id", "snapshot_date", "label", "partition_date"]]
X = features_df[feature_cols]

# 3. Apply scaler from model artifact
scaler = model_artifact["preprocessing_transformers"]["stdscaler"]
X_scaled = scaler.transform(X)  # Uses training data's mean/std

# 4. Generate predictions
model = model_artifact["model"]
predictions = model.predict_proba(X_scaled)[:, 1]
```

**Important Notes:**
- The scaler was fitted during training with log1p already applied to the 3 skewed columns
- Inference uses `scaler.transform()` which applies the same transformation
- No need to load a separate scaler file - it's in the model artifact
- Ensures exact same preprocessing as training

---

## 🔄 Snapshot Date Filtering

### How Filtering Works

The inference pipeline filters feature store data by year and month:

```python
snapshot_date = "2008-03-01"  # Input from environment/config

# Parse to get year and month
snapshot_dt = datetime.strptime(snapshot_date, "%Y-%m-%d")
target_year = 2008   # Extract year
target_month = 3     # Extract month

# Filter all feature store data
filtered_df = feature_df[
    (feature_df['snapshot_date'].dt.year == target_year) &
    (feature_df['snapshot_date'].dt.month == target_month)
]
```

**Example:** If snapshot_date is `2008-03-01`, it will include ALL records from March 2008 (2008-03-01, 2008-03-15, 2008-03-31, etc.)

---

## ⚠️ Prerequisites

### Must Run First

1. **Data Processing DAG** (`diab_medallion_ecs`)
   - Creates `gold/feature_store/`
   - Generates inference features with snapshot_date

2. **Model Training DAG** (`diab_model_training`)
   - Creates trained models in `model_registry/`
   - Model artifacts include preprocessing transformers (scaler)
   - Optional: Creates model comparison

### Validation Checks

The DAG automatically validates:
- ✅ At least one trained model exists
- ✅ Feature store data is available
- ℹ️  Model comparison exists (optional - uses default if missing)

**Removed Check:** No longer checks for separate preprocessing artifacts since scaler is in model artifact

---

## 🎯 Best Practices

### 1. Model Selection

**Automatic Best Model:**
```python
# Inference script automatically:
# 1. Loads latest_model_comparison.json
# 2. Uses recommended model
# 3. Falls back to XGBoost if comparison missing
```

**Manual Model Selection:**

### 1. Model Selection

**Automatic (Recommended):**
```python
# Uses model comparison to select best model
best_algorithm = get_best_model_info(config)
```

**Manual Override:**
```python
# Modify model_inference.py if needed:
best_algorithm = "random_forest"  # Override automatic selection
```

### 2. Snapshot Date Selection

**Via DAG Config (Recommended):**
```json
{"snapshot_date": "2008-06-01"}
```

**Via Environment Variable:**
```bash
export DEFAULT_SNAPSHOT_DATE=2008-06-01
```

**In Code:**
```python
snapshot_date = os.environ.get("SNAPSHOT_DATE", "2008-03-01")
```

### 3. Monitoring

**Check Prediction Quality:**
```python
# Review metadata file for:
- snapshot_date: Correct date processed?
- num_predictions: Expected number of records?
- avg_probability: Reasonable average score (typically 0.2-0.4)?
```

**CloudWatch Logs:**
- Monitor ECS task logs: `/ecs/diab-readmit-demo-model-training`
- Check for errors or warnings
- Verify snapshot date filtering worked correctly

### 4. Error Handling

**Missing Model:**
```
✗ No trained models found.
→ Action: Run diab_model_training DAG first
```

**Missing Data:**
```
✗ No feature data found
→ Action: Run diab_medallion_ecs DAG first
```

**No Data for Snapshot Date:**
```
✗ No data found for snapshot date 2008-06-01 (year=2008, month=6)
→ Action: Verify feature store has data for that month
→ Check: s3://bucket/gold/feature_store/
```
→ Action: Run diab_model_training DAG (creates scaler)
```

---

## 📝 Example Workflow

### Complete Inference Workflow

```bash
# Step 1: Ensure data is processed
# Trigger: diab_medallion_ecs
# Output: gold/feature_store/ (with snapshot_date column)

# Step 2: Ensure models are trained
# Trigger: diab_model_training
# Output: model_registry/{algorithm}/latest/model.pkl (contains model + scaler)

# Step 3: Run inference with specific snapshot date
# Trigger: diab_model_inference with config {"snapshot_date": "2008-03-01"}
# Output: gold/model_predictions/{algorithm}/{algorithm}_predictions_2008_03_01.parquet

# Step 4: Access predictions
aws s3 cp s3://bucket/gold/model_predictions/xgboost/xgboost_predictions_2008_03_01.parquet ./
```

### Local Testing

```python
import pandas as pd

# Load predictions for specific snapshot date
predictions = pd.read_parquet('xgboost_predictions_2008_03_01.parquet')

# Analyze results
print(f"Snapshot date: {predictions['snapshot_date'].iloc[0]}")
print(f"Total predictions: {len(predictions)}")
print(f"Avg probability: {predictions['model_predictions'].mean():.4f}")
print(f"High risk (>70%): {(predictions['model_predictions'] > 0.7).sum()}")

# Distribution of predictions
print("\nPrediction distribution:")
print(predictions['model_predictions'].describe())

# Filter high-risk patients
high_risk = predictions[predictions['model_predictions'] > 0.7]
print(f"\nHigh-risk encounter IDs:\n{high_risk['encounter_id'].tolist()}")
```

### Running Multiple Snapshots

```bash
# Generate predictions for different months
for date in "2008-03-01" "2008-06-01" "2008-09-01" "2008-12-01"; do
    airflow dags trigger diab_model_inference \
        --conf "{\"snapshot_date\": \"$date\"}"
    sleep 300  # Wait 5 minutes between runs
done
```

---

## 🔄 Scheduling

### Manual Trigger (Default)

```python
schedule_interval=None  # Manual trigger only
```

### Daily Inference

```python
schedule_interval='0 2 * * *'  # Daily at 2 AM
```

### After Training Completes

```python
# Use ExternalTaskSensor to wait for training
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_training = ExternalTaskSensor(
---

## 📊 Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| **Execution Time** | 5-15 minutes |
| **Data Volume** | ~10,000 records/snapshot month |
| **Prediction Rate** | ~1000 predictions/second |
| **Memory Usage** | ~1-2 GB |
| **CPU Usage** | ~50-70% (2vCPU) |

---

## 🐛 Troubleshooting

### Issue: "No trained models found"

**Cause:** Model training hasn't completed  
**Solution:** Run `diab_model_training` DAG first

### Issue: "No data found for snapshot date 2008-06-01"

**Cause:** Feature store doesn't have data for that year/month  
**Solution:** 
- Check available snapshot dates: `aws s3 ls s3://bucket/gold/feature_store/`
- Verify data processing DAG created features for that period
- Try a different snapshot date

### Issue: "Can't get attribute 'log1p_transform'"

**Cause:** Pickle serialization issue with custom transformer  
**Solution:** 
- Ensure `log1p_transform` function exists at module level in `model_inference.py`
- Function must match exactly with training script
- Re-run if code was updated

### Issue: "Feature columns mismatch"

**Cause:** Inference data has different features than training  
**Solution:** 
- Check feature engineering consistency
- Verify same data processing pipeline version
- Ensure feature store schema matches training expectations

### Issue: "Predictions seem unrealistic"

**Cause:** Preprocessing not applied correctly  
**Solution:**
- Verify model artifact loaded correctly (contains scaler)
- Check that scaler.transform() is being called
- Compare feature distributions with training data
- Verify snapshot_date filtering worked correctly

---

## 📚 Related Documentation

- **Model Training Setup**: `docs/MODEL_TRAINING_SETUP.md`
- **Model Comparison**: `docs/MODEL_COMPARISON_FEATURE.md`
- **Notebook Reference**: `notebooks/model_inference.ipynb`

---

## ✅ Summary

**Inference Pipeline Features:**
- ✅ Notebook-based workflow (matches `model_inference.ipynb`)
- ✅ Snapshot date filtering (by year/month)
- ✅ Self-contained model artifacts (model + scaler)
- ✅ Automatic best model selection
- ✅ Dynamic snapshot date configuration
- ✅ Comprehensive prerequisite validation
- ✅ Self-contained model artifacts (no separate scaler loading)
- ✅ Snapshot-based predictions (year/month filtering)
- ✅ Versioned predictions with metadata
- ✅ Production-ready error handling
- ✅ CloudWatch logging integration

**Key Differences from Training:**
- Uses model artifact (model + scaler combined)
- Filters by snapshot_date instead of train/test/oot splits
- No separate preprocessing step needed
- Simpler pipeline (fewer dependencies)

**Output:**
- Predictions saved to S3 in parquet format
- Metadata includes statistics and provenance
- Easy to load and analyze downstream

---

**Created:** November 3, 2025  
**Status:** ✅ Production Ready
