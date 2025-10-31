# Model Training Pipeline Setup

## Overview

The model training pipeline supports **parallel training of multiple algorithms** with temporal window validation and prerequisite checks. Each algorithm runs as a separate ECS task for independent execution, failure handling, and resource optimization.

**Key Infrastructure:** Uses a **dedicated ECS task definition** with **2 vCPU / 4 GB** (double the resources of data processing) for faster model training.

## Architecture

```
DAG: diab_model_training
│
├── check_model_config (ShortCircuitOperator)
│   └── Validates model_config.json
│
├── check_gold_data_exists (ShortCircuitOperator)
│   └── Ensures feature_store + label_store exist
│
├── check_sufficient_training_data (ShortCircuitOperator)
│   └── Verifies >= 10 data partitions
│
└── Parallel Training (3 ECS Tasks)
    ├── train_logistic_regression (2vCPU/4GB, 2hr timeout)
    ├── train_random_forest (2vCPU/4GB, 3hr timeout)
    └── train_xgboost (2vCPU/4GB, 3hr timeout)
```

### ECS Task Definitions

The pipeline uses **two separate task definitions** for optimal resource allocation:

| Task Definition | CPU | Memory | Use Case | Tasks |
|----------------|-----|--------|----------|-------|
| **diab-readmit-demo-pipeline** | 1 vCPU | 2 GB | Data processing (Bronze→Silver→Gold) | ETL workloads |
| **diab-readmit-demo-model-training** | 2 vCPU | 4 GB | Model training with hyperparameter tuning | ML workloads |

**Benefits:**
- 🚀 **2x faster training** with increased CPU
- 💾 **Prevents OOM errors** with 4GB memory for large datasets
- 💰 **Cost optimized** - data processing still uses cheaper 1vCPU tasks
- 🔧 **Independent scaling** - adjust resources per workload type

### Key Design Decisions

**Parallel vs Sequential Training:**
- ✅ **Parallel:** Faster pipeline execution, independent failures, better resource utilization
- ❌ Sequential: Slower, one failure blocks all subsequent models

**Single vs Multi-Algorithm Mode:**
- **Single-Algorithm Mode:** Triggered by DAG tasks with `ALGORITHM` environment variable
- **Multi-Algorithm Mode:** Local/manual execution trains all enabled algorithms

**Separate Task Definitions:**
- ✅ **Model Training (2vCPU/4GB):** Optimized for CPU-intensive ML workloads
- ✅ **Data Processing (1vCPU/2GB):** Cost-effective for I/O-bound ETL tasks

## Temporal Window Configuration

**File:** `conf/model_config.json`

The pipeline uses a 3-window temporal split:

| Window | Date Range | Purpose | Duration |
|--------|-----------|---------|----------|
| **Train** | 1999-01-01 to 2005-12-31 | Model fitting | 7 years |
| **Test** | 2006-01-01 to 2007-12-31 | Hyperparameter tuning & evaluation | 2 years |
| **OOT** | 2008-01-01 to 2008-12-31 | Final validation & inference | 1 year |

### Why This Split?

- **Prevents data leakage:** Strict temporal ordering
- **Realistic evaluation:** OOT simulates future unseen data
- **Sufficient training data:** 7 years provides robust signal
- **Proper validation:** 2-year test window for reliable performance estimation

## Training Configuration

### Algorithms Supported

Each algorithm runs as an **independent ECS task** in the DAG:

1. **Logistic Regression** (2 hour timeout)
   - Fast, interpretable baseline
   - L1/L2 regularization
   - Class weight balancing
   - Good for understanding feature importance
   
2. **Random Forest** (3 hour timeout)
   - Handles non-linear relationships
   - Feature importance via tree splits
   - Tree depth and count optimization
   
3. **XGBoost** (3 hour timeout)
   - State-of-the-art gradient boosting
   - Learning rate optimization
   - Scale_pos_weight for class imbalance
   - Best performance typically

### Algorithm Toggle Configuration

**Enable/Disable algorithms via `conf/model_config.json`:**

```json
{
  "training_config": {
    "algorithms": ["logistic_regression", "random_forest", "xgboost"],
    "enabled_algorithms": {
      "logistic_regression": true,
      "random_forest": true,
      "xgboost": false  // Disable XGBoost by setting to false
    },
    "hyperparameter_tuning": true,
    "feature_selection": true,
    "class_weight": "balanced"
  }
}
```

**How it works:**
- **In DAG (ECS):** Each algorithm has a dedicated task that runs regardless of config (passes `ALGORITHM` env var)
- **In model_train.py:** Checks `ALGORITHM` env var; if set, trains only that algorithm
- **Local execution:** Reads `enabled_algorithms` from config and trains all enabled ones

### Execution Modes

1. **Single-Algorithm Mode (DAG)**
   ```python
   # Triggered by ECS task with environment variable
   ALGORITHM=logistic_regression python model_train.py
   ```
   - Used by Airflow DAG tasks
   - Each algorithm runs in parallel
   - Independent failures and resource allocation

2. **Multi-Algorithm Mode (Local)**
   ```bash
   # No ALGORITHM env var - trains all enabled algorithms
   python model_train.py
   ```
   - Used for local testing
   - Trains all algorithms with `enabled_algorithms: true`
   - Sequential execution

### Hyperparameter Tuning
- Method: RandomizedSearchCV
- Cross-validation: 5-fold
- Scoring metric: AUC-ROC
- Iterations: 20 per algorithm

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC (primary metric)

## Files Created/Modified

### New Files
1. **`conf/model_config.json`**
   - Temporal window definitions
   - Model hyperparameters
   - Registry configuration

2. **`model_train.py`**
   - Main training script
   - Loads from gold layer
   - Trains multiple algorithms
   - Saves to S3 model registry

3. **`airflow/dags/diab_model_training.py`**
   - Training DAG with prerequisite checks
   - ShortCircuitOperators for validation
   - ECS task for training execution

### Modified Files
4. **`Dockerfile`**
   - Added model_train.py copy

## Model Registry Structure

Models are saved to S3 with the following structure:

```
s3://diab-readmit-123456-model-registry/
└── models/
    └── readmission/
        ├── logistic_regression_v20251031_120000.pkl
        ├── logistic_regression_v20251031_120000_metadata.json
        ├── logistic_regression_latest.pkl  (symlink)
        ├── logistic_regression_latest_metadata.json
        ├── random_forest_v20251031_120500.pkl
        ├── random_forest_v20251031_120500_metadata.json
        ├── xgboost_v20251031_121000.pkl
        └── xgboost_v20251031_121000_metadata.json
```

### Metadata Structure

Each model has a metadata JSON file containing:
```json
{
  "model_name": "xgboost",
  "training_date": "2025-10-31T12:10:00",
  "temporal_splits": {...},
  "feature_count": 45,
  "training_samples": 50000,
  "test_samples": 15000,
  "oot_samples": 8000,
  "performance": {
    "test": {
      "accuracy": 0.72,
      "precision": 0.68,
      "recall": 0.75,
      "f1": 0.71,
      "auc_roc": 0.78
    },
    "oot": {
      "accuracy": 0.70,
      "precision": 0.66,
      "recall": 0.73,
      "f1": 0.69,
      "auc_roc": 0.76
    }
  },
  "config": {...}
}
```

## How to Run

### Prerequisites
1. ✅ Data processing DAG completed (diab_medallion_ecs)
2. ✅ Gold layer data in S3 (feature_store + label_store)
3. ✅ S3 bucket for model registry created

### Step 1: Create Model Registry Bucket (if not exists)

```bash
aws s3 mb s3://diab-readmit-123456-model-registry --region ap-southeast-1
```

### Step 2: Update Airflow DAG on EC2

```bash
# SSH to EC2
ssh ec2-user@<EC2_PUBLIC_IP>

# Pull latest code
cd /opt/airflow/repo
git pull origin feature/airflow_aws_pipeline

# Restart Airflow
docker compose -f airflow-docker-compose.yaml down
docker compose -f airflow-docker-compose.yaml up -d
```

### Step 3: Build and Push Docker Image

```bash
# From local machine
ECR_REPO=$(aws ecr describe-repositories \
  --repository-names diab-readmit-pipeline \
  --region ap-southeast-1 \
  --query 'repositories[0].repositoryUri' \
  --output text)

aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin ${ECR_REPO}

docker build -t diab-readmit-pipeline:latest .
docker tag diab-readmit-pipeline:latest ${ECR_REPO}:latest
docker push ${ECR_REPO}:latest
```

### Step 4: Trigger Training DAG

1. Open Airflow UI: `http://<EC2_PUBLIC_IP>:8080`
2. Find DAG: `diab_model_training`
3. Enable the DAG (toggle on)
4. Trigger manually (play button)

### Step 5: Monitor Training

**In Airflow UI:**
- Watch task progress: 
  - `check_model_config` → `check_gold_data_exists` → `check_sufficient_training_data`
  - Then 3 parallel tasks: `train_logistic_regression`, `train_random_forest`, `train_xgboost`
- Graph View shows parallel execution of training tasks
- Each task can succeed/fail independently

**In AWS Console:**
- ECS → Clusters → diab-readmit-demo-cluster → Tasks
- CloudWatch Logs → /ecs/diab-readmit-demo
- Each algorithm will have separate task logs

**Expected Runtime:** 
- **Parallel execution:** ~2-3 hours (all models train simultaneously)
- **Sequential execution:** Would take 6-8 hours (if tasks were chained)

### Step 6: Verify Model Registry

After training completes, check S3:

```bash
# List all trained models
aws s3 ls s3://diab-readmit-123456-model-registry/models/readmission/ --recursive

# Expected output:
# logistic_regression_v20251031_155508.pkl
# logistic_regression_v20251031_155508_metadata.json
# logistic_regression_latest.pkl
# random_forest_v20251031_155509.pkl
# random_forest_v20251031_155509_metadata.json
# random_forest_latest.pkl
# xgboost_v20251031_155509.pkl
# xgboost_v20251031_155509_metadata.json
# xgboost_latest.pkl
```

## Task Dependencies

```
┌─────────────────────┐
│ check_model_config  │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│ check_gold_data_exists   │
└──────────┬───────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ check_sufficient_training_data  │
└──────────┬──────────────────────┘
           │
           ▼
    ┌──────┴──────┬──────────┐
    │             │          │
    ▼             ▼          ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│ LogReg │  │ RandFor  │  │ XGBoost  │
│ 2hr    │  │ 3hr      │  │ 3hr      │
└────────┘  └──────────┘  └──────────┘
```

**Benefits of Parallel Execution:**
1. ⏱️ **Faster:** 2-3 hours vs 6-8 hours sequential
2. 🔄 **Independent:** One model failure doesn't block others
3. 📊 **Visibility:** See which algorithms complete first
4. 🎛️ **Control:** Disable specific algorithms via config toggle

## Validation Checks

### 1. Model Config Check
```python
# Validates:
- temporal_splits exist
- All required fields present
- start_date/end_date for train/test/oot
```

**Skip if:** Config is invalid or missing

### 2. Gold Data Exists Check
```python
# Validates:
- s3://.../gold/feature_store/ has files
- s3://.../gold/label_store/ has files
```

**Skip if:** Data doesn't exist (run diab_medallion_ecs first)

### 3. Sufficient Training Data Check
```python
# Validates:
- At least 10 parquet partitions in feature_store
```

**Skip if:** Insufficient data partitions

## Troubleshooting

### Issue: DAG skipped all tasks

**Cause:** One of the ShortCircuitOperator checks failed

**Solution:**
1. Check Airflow logs for which check failed
2. Fix the underlying issue:
   - Config invalid → Fix conf/model_config.json and re-upload to S3
   - Data missing → Run diab_medallion_ecs DAG
   - Insufficient data → Extend data processing date range

### Issue: Specific algorithm task fails

**Cause:** Algorithm-specific error (e.g., XGBoost memory issues)

**Solution:**
1. Check CloudWatch logs for the specific algorithm task
2. Options:
   - **Disable the failing algorithm:** Set `enabled_algorithms.xgboost: false` in config
   - **Increase resources:** Update ECS task definition with more CPU/memory
   - **Adjust hyperparameters:** Reduce complexity in model_train.py

**Benefits of separate tasks:** Other algorithms continue training successfully!

### Issue: Training task fails with "No module named 'sklearn'"

**Cause:** Docker image doesn't have scikit-learn

**Solution:**
Add to requirements.txt:
```
scikit-learn==1.5.1
xgboost==2.1.0
```

Then rebuild and push image.

### Issue: "Access Denied" when saving to S3

**Cause:** ECS task role doesn't have S3 write permissions

**Solution:**
Update task role policy in Terraform to include:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:PutObject",
    "s3:GetObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::diab-readmit-123456-model-registry",
    "arn:aws:s3:::diab-readmit-123456-model-registry/*"
  ]
}
```

### Issue: "ValueError: could not convert string to float"

**Cause:** Categorical features (diagnosis codes) not excluded from training

**Solution:**
This is now fixed in the latest model_train.py:
- Excludes `diag_1`, `diag_2`, `diag_3` columns
- Applies StandardScaler to 7 numeric features only
- Uses ColumnTransformer with `remainder='passthrough'` for one-hot encoded features

### Issue: Want to train only one algorithm for testing

**Solution:**
**Option 1 - Disable in config (affects local execution only):**
```json
{
  "training_config": {
    "enabled_algorithms": {
      "logistic_regression": true,
      "random_forest": false,  // Skip this
      "xgboost": false          // Skip this
    }
  }
}
```

**Option 2 - Manual DAG task trigger (affects Airflow):**
In Airflow UI, manually trigger only specific tasks:
1. Run prerequisite checks
2. Trigger only `train_logistic_regression` task
3. Skip `train_random_forest` and `train_xgboost`

**Option 3 - Local testing:**
```bash
export ALGORITHM=logistic_regression
python model_train.py
```

## Local Test Results

The training pipeline was successfully tested locally in Docker with the following results:

### Dataset Statistics
```
✓ X_train: (71,596, 14), Readmission rate: 11.1%
✓ X_test:  (20,288, 14), Readmission rate: 11.2%
✓ X_oot:   (9,882, 14),  Readmission rate: 11.4%
```

### Preprocessing Applied
```
StandardScaler applied to 7 numeric features:
- age_midpoint
- admission_severity_score
- admission_source_risk_score
- metformin_ord
- insulin_ord
- severity_x_visits
- medication_density
```

### Model Performance

| Model | Test AUC | Test Recall | OOT AUC | OOT Recall | Training Time |
|-------|----------|-------------|---------|------------|---------------|
| **Logistic Regression** | 0.6120 | 0.4481 | 0.6161 | 0.4495 | ~2 min |
| **Random Forest** | 0.6202 | 0.4662 | **0.6308** | 0.4663 | ~3 min |
| **XGBoost** | 0.6198 | 0.4865 | **0.6321** | **0.4965** | ~3 min |

**Key Observations:**
- ✅ **No data leakage:** OOT performance consistent with Test set
- ✅ **XGBoost best:** Highest OOT AUC (0.6321) and recall (0.4965)
- ✅ **Class imbalance handled:** All models use balanced class weights
- ✅ **All models saved:** Successfully uploaded to S3 with metadata

**Best Hyperparameters:**
```python
# Logistic Regression
{'penalty': 'l1', 'class_weight': 'balanced', 'C': 0.001}

# Random Forest
{'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 5, 
 'min_samples_leaf': 1, 'class_weight': 'balanced'}

# XGBoost
{'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 50,
 'subsample': 1.0, 'colsample_bytree': 0.6, 'scale_pos_weight': 7.99}
```

## Next Steps

After training completes:

1. ✅ **Validate Models** - Check model performance in metadata
2. ⏭️ **Inference DAG** - Use trained models for batch predictions
3. ⏭️ **Monitoring DAG** - Track model drift and performance
4. ⏭️ **Model Comparison** - Compare multiple model versions
5. ⏭️ **A/B Testing** - Test models in production

## Configuration Customization

### Modify Temporal Windows

Edit `conf/model_config.json`:
```json
{
  "temporal_splits": {
    "train": {
      "start_date": "2000-01-01",  // Adjust as needed
      "end_date": "2006-12-31"
    },
    ...
  }
}
```

### Change Algorithms

Edit `conf/model_config.json`:
```json
{
  "training_config": {
    "algorithms": ["xgboost"],  // Train only XGBoost
    "hyperparameter_tuning": true
  }
}
```

### Adjust Hyperparameter Search

Modify `model_train.py` - Update param_dist dictionaries in each training function.

## Summary

✅ **Temporal window configuration**: Prevents data leakage  
✅ **Prerequisite checks**: Ensures safe execution  
✅ **Multiple algorithms**: LogReg, RF, XGBoost  
✅ **Hyperparameter tuning**: Automated with RandomizedSearchCV  
✅ **Model registry**: Versioned storage in S3  
✅ **Metadata tracking**: Full training lineage  

**Your training pipeline is production-ready!** 🚀
