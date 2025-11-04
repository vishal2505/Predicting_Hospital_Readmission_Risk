# Model Monitoring and Governance Pipeline

## Overview

The Model Monitoring and Governance pipeline is the final component of the MLOps lifecycle, closing the loop with automated model health tracking and retraining decisions.

**Pipeline Flow:**
```
Inference DAG Completes
         ↓
Monitoring & Governance DAG Triggers
         ↓
Load Predictions + Labels
         ↓
Calculate Metrics (AUC, GINI, PSI)
         ↓
Make Governance Decision
         ↓
(Optional) Trigger Retraining
```

## Architecture

### Components

1. **Airflow DAG** (`diab_model_monitoring_governance.py`)
   - Orchestrates monitoring and governance tasks
   - Runs as ECS tasks in Fargate
   - Can trigger automatically after inference or on schedule

2. **Monitoring Script** (`model_monitoring.py`)
   - Compares predictions vs actual labels
   - Calculates performance metrics per month
   - Detects model drift

3. **Governance Script** (`model_governance.py`)
   - Reviews monitoring results
   - Applies threshold-based decision logic
   - Optionally triggers retraining

### Data Flow

```
S3 Inputs:
├── gold/model_predictions/{algorithm}/
│   └── {algorithm}_predictions_{date}.parquet
└── gold/label_store/
    └── labels.parquet

S3 Outputs:
├── gold/model_monitoring/
│   └── {algorithm}_monitoring.parquet
└── gold/model_governance/
    └── {algorithm}_governance.parquet
```

## Metrics Explained

### Performance Metrics

#### AUC (Area Under ROC Curve)
- Range: 0.5 to 1.0
- Measures model's ability to distinguish classes
- **Threshold: < 0.70** triggers retraining
- Higher is better

#### GINI Coefficient
- Formula: `2 * AUC - 1`
- Range: 0 to 1
- Normalized version of AUC
- Higher indicates better discrimination

### Drift Metrics

#### PSI (Population Stability Index)
Measures shift in prediction score distribution between reference (first month) and current month.

**Interpretation:**
- **PSI < 0.10**: No significant change ✅
- **PSI 0.10 - 0.25**: Moderate drift ⚠️ (schedule retrain)
- **PSI > 0.25**: Significant drift 🚨 (retrain now)

**Calculation:**
```python
PSI = Σ (% reference - % current) × ln(% reference / % current)
```

Where:
- Reference: First month's prediction distribution
- Current: Current month's prediction distribution
- Predictions are bucketed into 10 bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)

#### CSI (Characteristic Stability Index)
Similar to PSI but for categorical features. Can be used to detect drift in input feature distributions.

## Governance Decision Logic

### Thresholds

```python
AUC_THRESHOLD = 0.70      # Performance threshold
PSI_WARNING = 0.10        # Moderate drift
PSI_CRITICAL = 0.25       # Significant drift
```

### Decision Matrix

| Condition | AUC | PSI | Decision | Action |
|-----------|-----|-----|----------|--------|
| Critical | < 0.70 | Any | **Retrain** | Immediate retraining |
| Critical | Any | > 0.25 | **Retrain** | Immediate retraining |
| Warning | ≥ 0.70 | 0.10-0.25 | **Schedule Retrain** | Plan retraining |
| Healthy | ≥ 0.70 | < 0.10 | **No Action** | Continue monitoring |

### Exit Codes

Scripts return exit codes for programmatic handling:

- `0`: No Action (healthy)
- `1`: Schedule Retrain (warning)
- `2`: Retrain (critical)
- `3`: Error

## Usage

### Running the DAG

#### Method 1: Manual Trigger (Default Settings)

```bash
# Via Airflow UI
# Navigate to DAGs → diab_model_monitoring_governance → Trigger DAG

# Via CLI
airflow dags trigger diab_model_monitoring_governance
```

#### Method 2: With Custom Configuration

```bash
# Monitor specific algorithm
airflow dags trigger diab_model_monitoring_governance \
  --conf '{"model_algorithm": "logistic_regression"}'

# Enable auto-retrain
airflow dags trigger diab_model_monitoring_governance \
  --conf '{"model_algorithm": "xgboost", "auto_retrain": "true"}'
```

#### Method 3: Scheduled Run

Edit DAG file to set schedule:

```python
# Run weekly on Monday
schedule_interval='0 9 * * 1'

# Run monthly on 1st
schedule_interval='0 9 1 * *'
```

### Running Scripts Locally

#### Monitoring Script

```bash
# Set environment
export AWS_REGION=ap-southeast-1
export DATAMART_BASE_URI=s3a://diab-readmit-123456-datamart/
export MODEL_ALGORITHM=xgboost

# Run monitoring
python model_monitoring.py
```

**Output:**
```
================================================================================
Model Monitoring
================================================================================
Model Algorithm: xgboost
Datamart URI: s3a://diab-readmit-123456-datamart/

Initializing Spark...
✓ Spark initialized

Loading predictions...
✓ Loaded 10,000 prediction records

Loading labels...
✓ Loaded 50,000 label records

Merging predictions with labels...
✓ Merged 9,800 matching records

Found 3 unique months: 2008-01, 2008-02, 2008-03

Using 2008-01 as reference month (n=3,200 records)

================================================================================
Calculating Metrics by Month
================================================================================

2008-01:
  Records: 3,200
  AUC: 0.7850
  GINI: 0.5700
  PSI: 0.0000
  PSI Status: ✓ No significant change

2008-02:
  Records: 3,300
  AUC: 0.7820
  GINI: 0.5640
  PSI: 0.0520
  PSI Status: ✓ No significant change

2008-03:
  Records: 3,300
  AUC: 0.7650
  GINI: 0.5300
  PSI: 0.1850
  PSI Status: ⚠️ Moderate drift

✓ Monitoring results saved to:
  s3a://diab-readmit-123456-datamart/gold/model_monitoring/xgboost_monitoring.parquet
```

#### Governance Script

```bash
# Set environment
export AWS_REGION=ap-southeast-1
export DATAMART_BASE_URI=s3a://diab-readmit-123456-datamart/
export MODEL_ALGORITHM=xgboost
export AUTO_RETRAIN=false

# Run governance
python model_governance.py
```

**Output:**
```
================================================================================
Model Governance
================================================================================
Model Algorithm: xgboost
Datamart URI: s3a://diab-readmit-123456-datamart/
Auto-Retrain: False

Initializing Spark...
✓ Spark initialized

Loading monitoring results...
✓ Loaded 3 monitoring records

Finding latest snapshot...
✓ Latest snapshot: 2008-03

================================================================================
Latest Metrics
================================================================================
📅 Snapshot Month: 2008-03
📊 Records: 3,300
📈 AUC: 0.7650 (threshold: 0.70)
📉 GINI: 0.5300
🔄 PSI: 0.1850 (warning: 0.10, critical: 0.25)

================================================================================
Governance Decision
================================================================================
⚠️ Warning: Moderate model drift detected

Decision: Schedule Retrain
Reason: PSI (0.1850) exceeds warning threshold (0.10)

✓ Governance decision saved to:
  s3a://diab-readmit-123456-datamart/gold/model_governance/xgboost_governance.parquet
```

## DAG Configuration

### Environment Variables

```yaml
# Required
AWS_REGION: ap-southeast-1
ECS_CLUSTER: my-ecs-cluster
ECS_CONTAINER_NAME: app
DATAMART_BASE_URI: s3a://diab-readmit-123456-datamart/

# ECS Networking
ECS_SUBNETS: subnet-xxx,subnet-yyy
ECS_SECURITY_GROUPS: sg-xxx

# Task Definition
ECS_MODEL_TRAINING_TASK_DEF: my-task-definition:1

# Optional
DEFAULT_MODEL_ALGORITHM: xgboost
```

### DAG Runtime Configuration

```json
{
  "model_algorithm": "xgboost",
  "auto_retrain": "false"
}
```

**Parameters:**
- `model_algorithm`: Which model to monitor (xgboost, logistic_regression, random_forest)
- `auto_retrain`: Enable automatic retraining trigger (true/false)

## Task Breakdown

### Task 1: Check Predictions Exist

**Purpose:** Verify predictions are available for monitoring

**Logic:**
- Checks `s3://{bucket}/gold/model_predictions/{algorithm}/`
- Returns True if files exist, False otherwise
- Short-circuits DAG if predictions missing

**Example Output:**
```
Checking s3://bucket/gold/model_predictions/xgboost/
✓ Predictions found for xgboost
  Proceeding with monitoring.
```

### Task 2: Check Labels Exist

**Purpose:** Verify label store is available

**Logic:**
- Checks `s3://{bucket}/gold/label_store/`
- Returns True if files exist, False otherwise
- Short-circuits DAG if labels missing

**Example Output:**
```
Checking s3://bucket/gold/label_store/
✓ Label store is available
  Proceeding with monitoring.
```

### Task 3: Run Model Monitoring

**Purpose:** Calculate performance and drift metrics

**Process:**
1. Load predictions from S3
2. Load labels from S3
3. Merge on encounter_id + snapshot_date
4. Group by snapshot_month (YYYY-MM)
5. Calculate per month:
   - AUC (Area Under Curve)
   - GINI coefficient
   - PSI (Population Stability Index)
6. Save to `gold/model_monitoring/`

**Runtime:** ~5-20 minutes (depends on data volume)

**Output Schema:**
```
snapshot_month: string (e.g., "2008-03")
row_count: long
auc: double
gini: double
psi: double
```

### Task 4: Run Model Governance

**Purpose:** Make retraining decision

**Process:**
1. Read monitoring results
2. Get latest snapshot metrics
3. Apply governance thresholds
4. Make decision (Retrain / Schedule Retrain / No Action)
5. Save to `gold/model_governance/`
6. Optionally trigger retraining

**Runtime:** ~2-5 minutes

**Output Schema:**
```
model_name: string
latest_snapshot: string
auc: double
psi: double
decision: string
reason: string
```

## Monitoring Best Practices

### 1. Establish Baseline

Run monitoring for at least 3 months before enabling auto-retrain to:
- Understand normal drift patterns
- Set appropriate thresholds
- Avoid false positives

### 2. Regular Cadence

- **Daily predictions** → **Weekly monitoring**
- **Weekly predictions** → **Monthly monitoring**
- **Monthly predictions** → **Quarterly monitoring**

### 3. Threshold Tuning

Default thresholds are conservative. Adjust based on:
- Business impact of false positives/negatives
- Model criticality
- Available retraining resources

```python
# Conservative (more retraining)
AUC_THRESHOLD = 0.75
PSI_CRITICAL = 0.20

# Aggressive (less retraining)
AUC_THRESHOLD = 0.65
PSI_CRITICAL = 0.30
```

### 4. Multi-Algorithm Monitoring

Monitor all algorithms in parallel:

```bash
# Monitor all models
for algo in xgboost logistic_regression random_forest; do
  airflow dags trigger diab_model_monitoring_governance \
    --conf "{\"model_algorithm\": \"$algo\"}"
done
```

## Troubleshooting

### No Matching Records

**Problem:**
```
⚠️ Warning: No matching records found!
```

**Causes:**
1. Predictions and labels have different encounter_ids
2. snapshot_date values don't align
3. Inference ran on wrong date

**Solutions:**
```bash
# Check prediction dates
aws s3 ls s3://bucket/gold/model_predictions/xgboost/ --recursive

# Check label dates
aws s3 ls s3://bucket/gold/label_store/ --recursive

# Re-run inference with correct snapshot_date
airflow dags trigger diab_model_inference \
  --conf '{"snapshot_date": "2008-03-01"}'
```

### AUC Calculation Failed

**Problem:**
```
Warning: Could not calculate AUC for 2008-03: Only one class present
```

**Causes:**
- All predictions are same class (0 or 1)
- Not enough label diversity in month

**Solutions:**
- Check label distribution in data
- May need longer time period
- Verify model is actually predicting probabilities

### PSI Shows High Drift

**Problem:**
```
PSI: 0.3500
PSI Status: 🚨 Significant drift
Decision: Retrain
```

**Possible Causes:**
1. **Data distribution changed** (expected - retrain needed)
2. **Different data source** (check data processing)
3. **Wrong reference month** (review monitoring logic)

**Investigation:**
```python
# Analyze prediction distributions
predictions = spark.read.parquet("s3a://bucket/gold/model_predictions/xgboost/")
predictions.groupBy("snapshot_month").agg(
    avg("model_predictions"),
    stddev("model_predictions")
).show()
```

### Governance File Not Found

**Problem:**
```
✗ Error loading monitoring results: File not found
```

**Causes:**
- Monitoring task failed
- Wrong model_algorithm specified

**Solutions:**
1. Check monitoring task logs
2. Verify model_algorithm matches inference
3. Run monitoring manually first

## Integration Patterns

### Pattern 1: Sequential After Inference

Most common - monitoring runs after each inference:

```python
# In inference DAG, add callback
def trigger_monitoring(**context):
    from airflow.api.common.trigger_dag import trigger_dag
    trigger_dag(
        dag_id='diab_model_monitoring_governance',
        conf={'model_algorithm': MODEL_ALGORITHM}
    )

# Add to inference DAG
on_success_callback = trigger_monitoring
```

### Pattern 2: Scheduled Independent

Monitoring runs on schedule, processes all available data:

```python
# In monitoring DAG
schedule_interval = '0 9 * * 1'  # Monday 9 AM
```

### Pattern 3: Event-Driven

Monitoring triggered by S3 events:

```python
# Lambda function on S3 prediction upload
def lambda_handler(event, context):
    # Trigger Airflow DAG via API
    airflow_api.trigger_dag(
        'diab_model_monitoring_governance',
        conf={'model_algorithm': extract_algorithm(event)}
    )
```

## Performance Optimization

### For Large Datasets

#### 1. Incremental Monitoring

Only monitor new months:

```python
# In model_monitoring.py
# Filter to recent months only
merged_pdf = merged_pdf[
    merged_pdf["snapshot_date"] >= last_monitored_date
]
```

#### 2. Sampling

Use sampling for very large datasets:

```python
# Sample 10% for monitoring
merged_pdf = merged_pdf.sample(frac=0.1, random_state=42)
```

#### 3. Partition Pruning

Partition data by month for faster reads:

```python
# Save with partitioning
predictions_df.write.partitionBy("snapshot_month").parquet(path)

# Read specific partition
spark.read.parquet(f"{path}/snapshot_month=2008-03")
```

## Security Considerations

### IAM Permissions

ECS task role needs:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::bucket-name/gold/model_predictions/*",
    "arn:aws:s3:::bucket-name/gold/label_store/*",
    "arn:aws:s3:::bucket-name/gold/model_monitoring/*",
    "arn:aws:s3:::bucket-name/gold/model_governance/*"
  ]
}
```

### Data Privacy

Label data contains ground truth - ensure:
- Encrypted at rest (S3 server-side encryption)
- Encrypted in transit (HTTPS)
- Access logged (S3 access logs)
- PII/PHI properly handled

## Monitoring the Monitor

Track monitoring pipeline health:

### CloudWatch Metrics

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

# Log monitoring completion
cloudwatch.put_metric_data(
    Namespace='MLOps/Monitoring',
    MetricData=[{
        'MetricName': 'MonitoringSuccess',
        'Value': 1,
        'Unit': 'Count'
    }]
)
```

### Alerting

Set up CloudWatch alarms:

```bash
# Alert if monitoring fails
aws cloudwatch put-metric-alarm \
  --alarm-name monitoring-failure \
  --metric-name MonitoringSuccess \
  --namespace MLOps/Monitoring \
  --statistic Sum \
  --period 86400 \
  --threshold 0 \
  --comparison-operator LessThanThreshold

# Alert if retrain decision made
aws cloudwatch put-metric-alarm \
  --alarm-name retrain-required \
  --metric-name RetrainDecision \
  --namespace MLOps/Monitoring \
  --statistic Sum \
  --period 3600 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold
```

## Future Enhancements

### 1. Feature Drift Monitoring

Track individual feature distributions:

```python
# Monitor drift in each feature
for feature in FEATURES:
    feature_psi = psi(
        reference_df[feature],
        current_df[feature]
    )
```

### 2. Prediction Confidence

Track model confidence over time:

```python
# Monitor prediction uncertainty
avg_confidence = predictions.mean()
confidence_std = predictions.std()
```

### 3. Business Metrics

Link model performance to business outcomes:

```python
# Track business impact
cost_avoided = calculate_cost_savings(predictions, actuals)
roi = cost_avoided / retraining_cost
```

### 4. Automated A/B Testing

Compare challenger model against champion:

```python
# Deploy both models
# Route traffic 50/50
# Monitor comparative performance
```

## Summary

The monitoring and governance pipeline completes the MLOps lifecycle:

**Train** → **Deploy** → **Monitor** → **Govern** → **Retrain**

Key benefits:
- ✅ Automated performance tracking
- ✅ Early drift detection
- ✅ Data-driven retraining decisions
- ✅ Reduced manual intervention
- ✅ Consistent model quality

This ensures models remain accurate and reliable in production without manual oversight.
