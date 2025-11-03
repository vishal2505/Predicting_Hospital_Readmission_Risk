# Model Inference Pipeline

**Date:** November 3, 2025  
**Status:** ✅ COMPLETED  

---

## Overview

Automated model inference pipeline that loads the best trained model and generates predictions for new data. The pipeline includes comprehensive prerequisite checks and follows production best practices.

---

## 📋 Architecture

### Pipeline Flow

```
Prerequisites (Sequential Checks)
    ↓
├─→ Check Trained Model Exists
├─→ Check Model Comparison Exists (optional)
├─→ Check Inference Data Exists
├─→ Check Preprocessing Artifacts Exist
    ↓
Run Model Inference
```

### Design Principles

1. **Best Model Selection**: Automatically uses the recommended model from comparison
2. **Fallback Strategy**: Uses XGBoost if comparison not available
3. **Preprocessing Consistency**: Uses same scaler and transformations as training
4. **Comprehensive Checks**: Validates all prerequisites before running
5. **Flexible Data Loading**: Can specify inference date or use latest data

---

## 🎯 Components

### 1. Inference Script (`model_inference.py`)

**Purpose:** Generate predictions using trained model

**Key Functions:**

#### `get_best_model_info(config)`
- Loads `latest_model_comparison.json` from S3
- Returns recommended model algorithm
- Fallback: Uses XGBoost if comparison not found

#### `load_model_from_s3(algorithm, config)`
- Downloads model.pkl from S3 model registry
- Loads model metadata (training date, feature count)
- Returns trained model and metadata

#### `load_preprocessed_scaler_from_s3()`
- Loads fitted StandardScaler from training preprocessing
- Ensures same transformation as training data
- Critical for prediction accuracy

#### `load_inference_data_from_s3(inference_date=None)`
- Loads features from `gold/feature_store/`
- Can specify date or use latest partition
- Returns pandas DataFrame with encounter_id

#### `preprocess_inference_data(df, scaler, feature_cols)`
- Applies log1p transformation to numeric features
- Applies StandardScaler (fitted during training)
- Returns preprocessed feature matrix

#### `generate_predictions(model, X, entity_ids, algorithm)`
- Generates probability predictions (0-1 score)
- Generates class predictions (0 or 1)
- Returns DataFrame with encounter_id and predictions

#### `save_predictions_to_s3(predictions_df, algorithm)`
- Saves predictions as parquet to S3
- Creates versioned files: `predictions_{timestamp}.parquet`
- Updates `latest_predictions.parquet` pointer
- Saves metadata (timestamp, counts, statistics)

---

### 2. Airflow DAG (`diab_model_inference.py`)

**DAG Name:** `diab_model_inference`  
**Schedule:** Manual trigger (or scheduled as needed)  
**Timeout:** 30 minutes

**Prerequisite Checks:**

#### Check 1: Trained Model Exists
- Verifies at least one trained model in model registry
- Checks: `model_registry/{algorithm}/latest/model.pkl`
- **Critical:** Pipeline stops if no model found

#### Check 2: Model Comparison Exists (Optional)
- Checks for `latest_model_comparison.json`
- Used to identify best model
- **Non-blocking:** Pipeline proceeds even if not found (uses default)

#### Check 3: Inference Data Exists
- Verifies feature data in `gold/feature_store/`
- **Critical:** Pipeline stops if no data found

#### Check 4: Preprocessing Artifacts Exist
- Checks for fitted scaler in `gold/preprocessed/`
- **Critical:** Required for consistent preprocessing

**Inference Task:**
- Uses same ECS task definition as training (2vCPU/4GB)
- Runs `model_inference.py` script
- Environment variables:
  - `AWS_REGION`: AWS region
  - `DATAMART_BASE_URI`: S3 datamart location
  - `MODEL_CONFIG_S3_URI`: Model config path
  - `INFERENCE_DATE` (optional): Specific date for predictions

---

## 📊 Data Flow

### Input Data

```
s3://bucket/gold/
├── feature_store/                    # Inference features
│   └── partition_date=YYYY-MM-DD/
│       └── *.parquet
├── preprocessed/                     # Scaler from training
│   └── train_data_YYYYMMDD_HHMMSS/
│       └── scaler.pkl
└── model_registry/                   # Trained models
    ├── {algorithm}/latest/
    │   ├── model.pkl
    │   └── metadata.json
    └── latest_model_comparison.json
```

### Output Data

```
s3://bucket/gold/model_predictions/
└── {algorithm}/
    ├── predictions_{timestamp}.parquet    # Versioned predictions
    ├── latest_predictions.parquet         # Latest pointer
    └── metadata_{timestamp}.json          # Prediction metadata
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
AWS_REGION=ap-southeast-1
DATAMART_BASE_URI=s3://bucket/
MODEL_CONFIG_S3_URI=s3://bucket/config/model_config.json

# Optional
INFERENCE_DATE=2008-12-01  # Specific date for predictions (YYYY-MM-DD)
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

### Trigger DAG Manually

```bash
# Via Airflow UI
# Navigate to DAGs → diab_model_inference → Trigger

# Via CLI
airflow dags trigger diab_model_inference
```

### Run Inference Locally

```bash
# Set environment variables
export AWS_REGION=ap-southeast-1
export DATAMART_BASE_URI=s3://bucket/
export MODEL_CONFIG_S3_URI=s3://bucket/config/model_config.json

# Optional: Specify inference date
export INFERENCE_DATE=2008-12-01

# Run inference
python model_inference.py
```

### Specify Inference Date in DAG

Edit `diab_model_inference.py` to set specific date:

```python
overrides={
    'containerOverrides': [
        {
            'environment': [
                {'name': 'INFERENCE_DATE', 'value': '2008-12-01'},  # Add this line
            ],
        },
    ],
}
```

---

## 📈 Predictions Output

### Parquet File Structure

```python
predictions_df.columns = [
    'encounter_id',              # Patient encounter identifier
    'prediction_probability',    # Readmission probability (0-1)
    'prediction_class',          # Binary prediction (0 or 1)
    'model_algorithm',           # Algorithm used (e.g., 'xgboost')
    'prediction_timestamp'       # When prediction was generated
]
```

### Example Output

```
encounter_id  prediction_probability  prediction_class  model_algorithm  prediction_timestamp
-----------  ----------------------  ----------------  ---------------  ---------------------
12345        0.6842                  1                 xgboost          2025-11-03T15:30:00
23456        0.2153                  0                 xgboost          2025-11-03T15:30:00
34567        0.8291                  1                 xgboost          2025-11-03T15:30:00
```

### Metadata File

```json
{
  "timestamp": "2025-11-03T15:30:00",
  "algorithm": "xgboost",
  "num_predictions": 9882,
  "positive_rate": 0.1124,
  "avg_probability": 0.3456,
  "predictions_path": "s3://bucket/gold/model_predictions/xgboost/predictions_20251103_153000.parquet"
}
```

---

## 🔍 Preprocessing Details

### Log1p Transformation

Applied to **only 3 columns** with highly skewed distributions (same as training):
- `age_midpoint`
- `severity_x_visits`
- `medication_density`

**NOT applied to:**
- `admission_severity_score` (already normalized)
- `admission_source_risk_score` (not highly skewed)
- `metformin_ord` (discrete ordinal values)
- `insulin_ord` (discrete ordinal values)

**Why Only 3 Columns?**
- These 3 features have extreme right-skewness
- Other numeric features are well-distributed or discrete
- Applying log to normalized/discrete features causes numerical issues
- Matches validated approach from training notebook

### Scaling

Uses **fitted StandardScaler from training**:
- Loaded from `gold/preprocessed/{run}/scaler.pkl`
- Same mean and std as training data
- Ensures feature ranges match training

**Pipeline:**
```python
# For age_midpoint, severity_x_visits, medication_density:
X → log1p transform → StandardScaler → Model

# For other numeric features:
X → StandardScaler → Model

# For categorical features:
X → passthrough (no transformation) → Model
```

---

## ⚠️ Prerequisites

### Must Run First

1. **Data Processing DAG** (`diab_medallion_ecs`)
   - Creates `gold/feature_store/`
   - Generates inference features

2. **Model Training DAG** (`diab_model_training`)
   - Creates trained models in `model_registry/`
   - Generates preprocessing artifacts (`scaler.pkl`)
   - Optional: Creates model comparison

### Validation Checks

The DAG automatically validates:
- ✅ At least one trained model exists
- ✅ Feature data is available
- ✅ Preprocessing scaler exists
- ℹ️  Model comparison exists (optional)

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
```python
# Modify model_inference.py if needed:
best_algorithm = "random_forest"  # Override automatic selection
```

### 2. Data Freshness

**Latest Data:**
```python
# Default: Uses most recent partition
inference_df = load_inference_data_from_s3()
```

**Specific Date:**
```python
# For specific date:
inference_df = load_inference_data_from_s3(inference_date="2008-12-01")
```

### 3. Monitoring

**Check Prediction Quality:**
```python
# Review metadata file for:
- num_predictions: Expected number of records?
- positive_rate: Within expected range (10-15%)?
- avg_probability: Reasonable average score?
```

**CloudWatch Logs:**
- Monitor ECS task logs: `/ecs/diab-readmit-demo-model-training`
- Check for errors or warnings
- Verify preprocessing steps completed

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

**Missing Scaler:**
```
✗ No preprocessing artifacts found
→ Action: Run diab_model_training DAG (creates scaler)
```

---

## 📝 Example Workflow

### Complete Inference Workflow

```bash
# Step 1: Ensure data is processed
# Trigger: diab_medallion_ecs
# Output: gold/feature_store/, gold/label_store/

# Step 2: Ensure models are trained
# Trigger: diab_model_training
# Output: model_registry/{algorithm}/, gold/preprocessed/scaler.pkl

# Step 3: Run inference
# Trigger: diab_model_inference
# Output: gold/model_predictions/{algorithm}/

# Step 4: Access predictions
aws s3 cp s3://bucket/gold/model_predictions/xgboost/latest_predictions.parquet ./
```

### Local Testing

```python
import pandas as pd

# Load predictions
predictions = pd.read_parquet('latest_predictions.parquet')

# Analyze results
print(f"Total predictions: {len(predictions)}")
print(f"Positive rate: {predictions['prediction_class'].mean():.2%}")
print(f"High risk (>70%): {(predictions['prediction_probability'] > 0.7).sum()}")

# Filter high-risk patients
high_risk = predictions[predictions['prediction_probability'] > 0.7]
print(f"\nHigh-risk encounter IDs:\n{high_risk['encounter_id'].tolist()}")
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
    task_id='wait_for_training',
    external_dag_id='diab_model_training',
    external_task_id='generate_model_comparison',
    mode='poke',
)
```

---

## 📊 Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| **Execution Time** | 5-15 minutes |
| **Data Volume** | ~10,000 records/partition |
| **Prediction Rate** | ~1000 predictions/second |
| **Memory Usage** | ~1-2 GB |
| **CPU Usage** | ~50-70% (2vCPU) |

---

## 🐛 Troubleshooting

### Issue: "No trained models found"

**Cause:** Model training hasn't completed  
**Solution:** Run `diab_model_training` DAG first

### Issue: "No preprocessing artifacts found"

**Cause:** Training preprocessing step not completed  
**Solution:** Ensure `diab_model_training` DAG completed successfully

### Issue: "Feature columns mismatch"

**Cause:** Inference data has different features than training  
**Solution:** 
- Check feature engineering consistency
- Verify same data processing pipeline
- Review feature_cols in model metadata

### Issue: "Predictions seem unrealistic"

**Cause:** Preprocessing not applied correctly  
**Solution:**
- Verify scaler loaded from correct training run
- Check log1p transformation applied
- Compare feature distributions with training

---

## 📚 Related Documentation

- **Model Training Setup**: `docs/MODEL_TRAINING_SETUP.md`
- **Model Comparison**: `docs/MODEL_COMPARISON_FEATURE.md`
- **DAG Architecture**: `docs/DAG_ARCHITECTURE.md`

---

## Summary

**Inference Pipeline Features:**
- ✅ Automatic best model selection
- ✅ Comprehensive prerequisite validation
- ✅ Consistent preprocessing (log1p + scaling)
- ✅ Flexible data loading (latest or specific date)
- ✅ Versioned predictions with metadata
- ✅ Production-ready error handling
- ✅ CloudWatch logging integration

**Output:**
- Predictions saved to S3 in parquet format
- Metadata includes statistics and provenance
- Easy to load and analyze downstream

---

**Created:** November 3, 2025  
**Status:** ✅ Production Ready
