# DAG Architecture for ML Pipeline

## Overview

This document explains the architectural decision for organizing the hospital readmission prediction ML pipeline into **4 separate DAGs** instead of combining data processing with model training into a single DAG.

## Architecture Decision

### Recommended: 4 Separate DAGs

1. **`diab_medallion_ecs`** - Data Processing (Bronze → Silver → Gold)
2. **`diab_model_training`** - Model Training
3. **`diab_model_inference`** - Batch Predictions
4. **`diab_model_monitoring`** - Drift Detection & Performance Monitoring

### Alternative Considered: 3 Combined DAGs

- Data Processing + Training (combined)
- Inference
- Monitoring

**Decision: The 4-DAG architecture is recommended as the production-ready approach.**

---

## Rationale for Separate DAGs

### 1. Different Scheduling Requirements

**Data Processing:**
- Runs frequently (daily or weekly)
- Processes new patient data as it arrives
- Keeps gold layer up-to-date for analysis

**Model Training:**
- Runs infrequently (monthly or quarterly)
- Only when sufficient new data accumulated
- Triggered manually or by performance degradation
- Should NOT run every time data is processed

**Problem with Combined Approach:**
If data processing and training are in the same DAG:
- Training runs unnecessarily every time new data arrives (wasteful)
- Cannot run data processing without triggering training
- Difficult to tune two different schedules in one DAG

### 2. Resource Management

**Data Processing:**
- Lightweight Spark jobs
- 1 vCPU / 2 GB memory sufficient
- Fast execution (minutes)

**Model Training:**
- Heavy compute requirements
- **2 vCPU / 4 GB** for hyperparameter tuning (implemented in our architecture)
- May need 4+ vCPUs / 8-16 GB for larger datasets
- Hyperparameter tuning can take hours
- Cross-validation multiplies compute time

**Problem with Combined Approach:**
- ECS task must be sized for training (expensive resources)
- Data processing wastes resources when provisioned for training
- Cannot optimize task definitions separately
- Higher AWS costs for every data processing run

**Our Solution:**
- ✅ **Separate task definitions:** `diab-readmit-demo-pipeline` (1vCPU/2GB) and `diab-readmit-demo-model-training` (2vCPU/4GB)
- ✅ **Cost optimized:** Data processing uses cheaper resources
- ✅ **Performance optimized:** Model training gets 2x more CPU for faster hyperparameter search

### 3. Failure Handling & Monitoring

**Data Processing Failures:**
- Usually data quality issues
- Quick to debug and retry
- Impacts downstream analytics immediately
- Need fast alerts and resolution

**Model Training Failures:**
- Can be hyperparameter issues, convergence problems, or data distribution changes
- Require ML expertise to debug
- Don't block daily operations
- Acceptable to fail and investigate later

**Problem with Combined Approach:**
- Training failure blocks data processing retry
- Cannot monitor SLAs separately
- Alerting becomes ambiguous (which part failed?)
- Rollback complexity increases

### 4. Development & Testing

**Data Processing:**
- Changes affect data schema and transformations
- Need to test with production data samples
- Deployed frequently (feature engineering updates)

**Model Training:**
- Changes affect algorithms and hyperparameters
- Need to test with historical data windows
- Deployed less frequently (algorithm updates)

**Problem with Combined Approach:**
- Cannot deploy data processing updates without redeploying training code
- Testing becomes complex (must validate both components together)
- Harder to A/B test different models
- Version control becomes messy

### 5. Scalability & Flexibility

**Separate DAGs Enable:**
- Training multiple models in parallel (different algorithms)
- Champion/challenger model patterns
- Re-training on different temporal windows
- Running inference without re-training
- Hot-swapping models without changing data pipeline

**Combined DAGs Limit:**
- Tightly coupled components
- Difficult to add new model variants
- Cannot experiment with training schedules
- Hard to implement rollback strategies

---

## Production MLOps Patterns

### Industry Examples

**Netflix:**
- Separate DAGs for feature engineering and model training
- Training triggered by data quality checks
- Independent inference pipelines

**Uber:**
- Data pipelines run continuously
- Model training scheduled separately
- Monitoring DAGs alert on drift

**Airbnb:**
- ETL DAGs independent of ML training
- Feature stores decouple data from models
- Training orchestrated by separate workflows

### Best Practices from MLOps Community

1. **Separation of Concerns**: Data engineering ≠ model training
2. **Independent Scaling**: Size resources appropriately
3. **Failure Isolation**: Don't cascade failures
4. **Clear Ownership**: Data team owns processing, ML team owns training
5. **Flexible Scheduling**: Each component has its own SLA

---

## Orchestration Between DAGs

### Using TriggerDagRunOperator

DAGs can be connected while remaining independent:

```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# In data processing DAG
trigger_training = TriggerDagRunOperator(
    task_id='trigger_model_training',
    trigger_dag_id='diab_model_training',
    wait_for_completion=False,
    conf={'triggered_by': 'data_processing'},
)

# Add conditional trigger (only if enough new data)
check_new_data = ShortCircuitOperator(
    task_id='check_sufficient_new_data',
    python_callable=lambda: check_data_volume_increased()
)

check_new_data >> trigger_training
```

### Example Flow

```
Data Processing DAG (Daily)
    └─> Bronze → Silver → Gold → Check New Data?
                                      ├─> Yes: Trigger Training DAG (Monthly)
                                      └─> No: Skip

Training DAG (On-Demand)
    └─> Validate Config → Check Data → Train Models → Save to Registry
                                                           └─> Trigger Inference DAG

Inference DAG (Daily)
    └─> Load Latest Model → Generate Predictions → Save Results
                                                       └─> Trigger Monitoring DAG

Monitoring DAG (Weekly)
    └─> Check Performance → Detect Drift → Alert if Needed
                                              └─> Re-trigger Training if Drift Detected
```

---

## DAG Responsibilities

### 1. Data Processing DAG (`diab_medallion_ecs`)

**Purpose:** Transform raw hospital data into ML-ready features

**Tasks:**
- Bronze: Raw CSV to Parquet
- Silver: Clean, deduplicate, standardize
- Gold: Feature engineering, create feature/label stores

**Triggers:**
- Scheduled: Daily at 2 AM
- Manual: When new data files uploaded

**Outputs:**
- `s3://diab-readmit-123456-datamart/gold/features/`
- `s3://diab-readmit-123456-datamart/gold/labels/`

**SLA:** 30 minutes

---

### 2. Model Training DAG (`diab_model_training`)

**Purpose:** Train and validate ML models using temporal windows with optimized preprocessing

**Tasks:**
1. **Prerequisite Checks:**
   - Check model config exists and is valid
   - Check gold data exists in S3
   - Check sufficient training data available

2. **Data Preprocessing (runs ONCE):**
   - Load gold layer data (feature_store + label_store)
   - Apply temporal window splits (train/test/oot)
   - Apply StandardScaler preprocessing
   - Save preprocessed data to `s3://bucket/gold/preprocessed/`
   - Creates `latest.txt` pointer for downstream tasks

3. **Model Training (runs in PARALLEL):**
   - Train Logistic Regression (loads preprocessed data)
   - Train Random Forest (loads preprocessed data)
   - Train XGBoost (loads preprocessed data)
   - Each evaluates on test and OOT windows
   - Each saves to separate algorithm folder in model registry

**Triggers:**
- Scheduled: Monthly (first Sunday at 3 AM)
- Manual: When retraining needed
- Conditional: Triggered by data processing if significant new data
- Alert-based: Triggered by monitoring if drift detected

**Outputs:**
- **Preprocessed Data:** `s3://bucket/gold/preprocessed/train_data_<timestamp>/`
  - `train_processed.parquet`
  - `test_processed.parquet`
  - `oot_processed.parquet`
  - `scaler.pkl`
  - `metadata.json`
- **Models (organized by algorithm):**
  - `s3://bucket/model_registry/logistic_regression/v<timestamp>/model.pkl`
  - `s3://bucket/model_registry/random_forest/v<timestamp>/model.pkl`
  - `s3://bucket/model_registry/xgboost/v<timestamp>/model.pkl`
  - Each with `latest/` symlink and `versions.json` index

**Architecture Benefits:**
- ✅ **No redundant preprocessing** - Preprocessing runs once, shared by all models
- ✅ **Parallel training** - All 3 algorithms train simultaneously
- ✅ **Organized model registry** - Clear folder structure per algorithm
- ✅ **Version tracking** - Complete history with `versions.json`

**SLA:** 2.5-3 hours (preprocessing ~5 min + parallel training ~2.5 hours)

**Configuration:** `conf/model_config.json`

---

### 3. Model Inference DAG (`diab_model_inference`)

**Purpose:** Generate batch predictions for new patients

**Tasks:**
1. Check latest model exists in registry
2. Check new data in gold layer (not already predicted)
3. Load model and generate predictions
4. Save predictions with confidence scores
5. Optionally trigger monitoring

**Triggers:**
- Scheduled: Daily at 6 AM (after data processing)
- Manual: On-demand prediction requests

**Outputs:**
- `s3://diab-readmit-123456-predictions/predictions_<date>.parquet`

**SLA:** 15 minutes

---

### 4. Model Monitoring DAG (`diab_model_monitoring`)

**Purpose:** Track model performance and detect drift

**Tasks:**
1. Check predictions exist
2. Check actual labels available (for performance metrics)
3. Calculate performance metrics (accuracy, precision, recall, F1, AUC)
4. Detect data drift (feature distribution changes)
5. Detect concept drift (performance degradation)
6. Alert if thresholds breached
7. Trigger retraining if needed

**Triggers:**
- Scheduled: Weekly (Sunday at 10 AM)
- Manual: Ad-hoc analysis
- Event-based: After inference completes

**Outputs:**
- `s3://diab-readmit-123456-monitoring/reports/report_<date>.json`
- Alerts via email/Slack

**SLA:** 30 minutes

**Alert Thresholds:**
- Accuracy drop > 5%
- Data drift PSI > 0.2
- Feature missing rate > 10%

---

## Configuration Management

### Centralized Config: `conf/model_config.json`

```json
{
  "temporal_splits": {
    "train": {"start_year": 1999, "end_year": 2005},
    "test": {"start_year": 2006, "end_year": 2007},
    "oot": {"start_year": 2008, "end_year": 2008}
  },
  "model_config": {
    "model_registry_bucket": "diab-readmit-123456-model-registry",
    "random_state": 42,
    "cv_folds": 5,
    "n_iter_search": 20
  },
  "training_config": {
    "algorithms": ["logistic_regression", "random_forest", "xgboost"],
    "hyperparameter_tuning": true,
    "feature_selection": false
  },
  "inference_config": {
    "batch_size": 10000,
    "prediction_threshold": 0.5,
    "confidence_intervals": true
  },
  "monitoring_config": {
    "drift_threshold_psi": 0.2,
    "performance_threshold_drop": 0.05,
    "min_samples_for_metrics": 100
  }
}
```

### Environment-Specific Configs

- **Development:** `conf/model_config.dev.json`
- **Staging:** `conf/model_config.staging.json`
- **Production:** `conf/model_config.prod.json`

Use environment variables in DAGs to select appropriate config:

```python
import os
env = os.getenv('ENVIRONMENT', 'dev')
config_path = f'conf/model_config.{env}.json'
```

---

## Deployment Workflow

### 1. Initial Setup

```bash
# Create model registry bucket
aws s3 mb s3://diab-readmit-123456-model-registry --region ap-southeast-1

# Upload model configuration
aws s3 cp conf/model_config.json s3://diab-readmit-123456-datamart/config/

# Build and push Docker image with training code
docker build -t hospital-readmission-pipeline .
aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com
docker tag hospital-readmission-pipeline:latest <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/hospital-readmission-pipeline:latest
docker push <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/hospital-readmission-pipeline:latest
```

### 2. Deploy DAGs

```bash
# Copy DAG files to Airflow instance
scp -i <key>.pem airflow/dags/*.py ec2-user@<ec2-ip>:~/airflow/dags/

# Or use git on EC2
ssh -i <key>.pem ec2-user@<ec2-ip>
cd ~/Predicting_Hospital_Readmission_Risk
git pull origin feature/airflow_aws_pipeline
docker-compose restart airflow-scheduler
```

### 3. First Run Sequence

```
Day 1: Run data processing DAG manually
    └─> Validates bronze/silver/gold pipeline works

Day 2: Run training DAG manually
    └─> Trains initial models on historical data (1999-2008)
    └─> Saves models to registry

Day 3: Run inference DAG manually
    └─> Generates predictions using latest model
    └─> Validates end-to-end prediction flow

Day 4: Run monitoring DAG manually
    └─> Establishes baseline metrics
    └─> Sets up alerting

Week 2: Enable scheduled runs
    └─> Data processing: daily
    └─> Inference: daily
    └─> Monitoring: weekly
    └─> Training: monthly (conditional on new data)
```

---

## Monitoring & Alerting

### DAG-Level Metrics

| DAG | Success Rate | Avg Duration | SLA |
|-----|--------------|--------------|-----|
| Data Processing | > 95% | 15 min | 30 min |
| Model Training | > 90% | 2 hours | 4 hours |
| Model Inference | > 98% | 5 min | 15 min |
| Model Monitoring | > 95% | 10 min | 30 min |

### Alert Channels

1. **Critical Alerts** (PagerDuty/Phone):
   - Data processing failed 3 times consecutively
   - Inference DAG down > 2 hours
   - Model accuracy dropped > 10%

2. **Warning Alerts** (Slack):
   - Training DAG failed
   - Data drift detected
   - New data volume unusual

3. **Info Alerts** (Email):
   - Training completed successfully
   - New model deployed
   - Weekly monitoring report

---

## Common Operations

### Retrain Model Manually

```bash
# Trigger training DAG via Airflow UI
# Or use CLI
airflow dags trigger diab_model_training \
  --conf '{"reason": "manual_retrain", "triggered_by": "ml_team"}'
```

### Deploy New Model Version

```bash
# Models are auto-deployed when training completes
# Inference DAG automatically picks latest model from registry
# No manual deployment needed

# To rollback to previous model, update metadata:
aws s3 cp s3://diab-readmit-123456-model-registry/metadata/metadata_<old_timestamp>.json \
          s3://diab-readmit-123456-model-registry/metadata/latest.json
```

### Check Model Performance

```bash
# Run monitoring DAG
airflow dags trigger diab_model_monitoring

# View latest report
aws s3 cp s3://diab-readmit-123456-monitoring/reports/latest.json - | jq .
```

### Update Training Configuration

```bash
# Edit config file
vim conf/model_config.json

# Upload to S3
aws s3 cp conf/model_config.json s3://diab-readmit-123456-datamart/config/

# Training DAG will use updated config on next run
```

---

## Troubleshooting

### Training DAG Stuck at "check_sufficient_training_data"

**Symptom:** ShortCircuitOperator marks task as skipped

**Cause:** Fewer than 10 partitions in gold layer

**Solution:**
```bash
# Check partition count
aws s3 ls s3://diab-readmit-123456-datamart/gold/features/ | wc -l

# If < 10, run data processing DAG to load more data
airflow dags trigger diab_medallion_ecs
```

### Model Training Timeout After 4 Hours

**Symptom:** ECS task killed by timeout

**Cause:** Hyperparameter tuning taking too long

**Solutions:**
1. Reduce `n_iter_search` in model_config.json (from 20 to 10)
2. Reduce `cv_folds` (from 5 to 3)
3. Train fewer algorithms (comment out XGBoost for faster runs)
4. Increase ECS task timeout in DAG (from 4 hours to 6 hours)

### Inference DAG: "No new data to predict"

**Symptom:** ShortCircuitOperator skips inference task

**Cause:** All gold layer data already predicted

**Expected Behavior:** This is normal if data processing hasn't added new records

**Action:** Wait for new data or run data processing with updated date range

### Monitoring DAG: "Actual labels not available"

**Symptom:** Performance metrics cannot be calculated

**Cause:** Readmission labels require 30-day follow-up period

**Expected Behavior:** Recent predictions won't have labels yet

**Solution:** Monitoring should only calculate performance metrics for predictions > 30 days old

---

## Future Enhancements

### Short-Term (Next 3 Months)

1. **Real-Time Inference:**
   - Add streaming inference DAG using Kinesis
   - Lambda function for single-patient predictions
   - API Gateway endpoint for external requests

2. **Advanced Monitoring:**
   - Explainability reports (SHAP values)
   - Feature importance tracking
   - Prediction distribution analysis

3. **A/B Testing:**
   - Deploy champion/challenger models
   - Split traffic for comparison
   - Automated winner selection

### Long-Term (6-12 Months)

1. **Auto-Retraining:**
   - Trigger training based on drift severity
   - Automatic hyperparameter optimization
   - Continual learning pipeline

2. **Multi-Model Ensemble:**
   - Combine predictions from multiple algorithms
   - Weighted voting or stacking
   - Confidence calibration

3. **Model Serving Infrastructure:**
   - SageMaker endpoints for low-latency inference
   - Model versioning with rollback
   - Canary deployments

---

## Conclusion

The **4-DAG architecture** provides:

✅ **Flexibility:** Independent scheduling and scaling  
✅ **Reliability:** Isolated failures and clear SLAs  
✅ **Maintainability:** Separation of concerns and ownership  
✅ **Scalability:** Easy to add new models or pipelines  
✅ **Cost-Efficiency:** Right-sized resources for each task  

While it may seem more complex initially, this architecture is the **production-ready approach** used by leading companies and recommended by MLOps best practices.

---

## References

- [Airflow Best Practices - DAG Design](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Netflix ML Infrastructure](https://netflixtechblog.com/introducing-metaflow-a-framework-for-data-scientists-a85f8c9a5c1a)
- [Uber Michelangelo](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- [Production ML Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/)

---

**Document Version:** 1.0  
**Last Updated:** October 31, 2025  
**Author:** ML Engineering Team  
**Reviewers:** Data Engineering Team, DevOps Team
