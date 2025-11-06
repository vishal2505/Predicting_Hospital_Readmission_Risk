# Hospital Readmission Risk Prediction

ML pipeline for predicting 30-day hospital readmission risk for diabetes patients using AWS cloud infrastructure.

## 🎯 Project Overview

This project implements a production-ready MLOps pipeline that:
- Processes hospital patient data through bronze → silver → gold medallion architecture
- Trains ML models using temporal window split to prevent data leakage
- Generates batch predictions for readmission risk
- Monitors model performance and data drift

**Tech Stack:** Apache Airflow, Apache Spark, AWS ECS/Fargate, S3, Docker, Terraform

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Raw Data  │ --> │ Bronze      │ --> │   Silver    │ --> │    Gold     │
│   (S3)      │     │ (Parquet)   │     │  (Cleaned)  │     │ (Features)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Monitoring  │ <-- │ Predictions │ <-- │   Models    │ <-- │  Training   │
│   (Drift)   │     │   (Batch)   │     │  Registry   │     │  Pipeline   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**4 Airflow DAGs:**
1. **Data Processing** (`diab_medallion_ecs`) - ETL pipeline
2. **Model Training** (`diab_model_training`) - Train & validate models
3. **Model Inference** (`diab_model_inference`) - Batch predictions *(coming soon)*
4. **Model Monitoring** (`diab_model_monitoring`) - Performance tracking *(coming soon)*

## 🚀 Quick Start

### Prerequisites
- AWS Account with admin access
- AWS CLI configured (`aws configure`)
- Terraform v1.x
- Docker v20.x
- Git

### 1. Initial Setup (First Time)

Follow the complete setup guide: **[📖 How to Start Guide](docs/HOW_TO_START.md)**

Quick summary:
```bash
# Clone repository
git clone https://github.com/vishal2505/Predicting_Hospital_Readmission_Risk.git
cd Predicting_Hospital_Readmission_Risk

# Deploy AWS infrastructure
cd infra/terraform/aws-ec2-airflow-ecs
terraform init
terraform apply

# Build and push Docker image
docker build -t hospital-readmission-pipeline .
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin <ECR_URL>
docker tag hospital-readmission-pipeline:latest <ECR_URL>:latest
docker push <ECR_URL>:latest

# Upload configuration
aws s3 cp conf/model_config.json s3://diab-readmit-123456-datamart/config/ --region ap-southeast-1

# Upload raw data
aws s3 cp data/diabetic_data.csv s3://diab-readmit-123456-datamart/bronze/ --region ap-southeast-1
```

### 2. Access Airflow UI

```bash
# Get EC2 public IP from Terraform output
cd infra/terraform/aws-ec2-airflow-ecs
terraform output ec2_public_ip

# Open in browser
http://<EC2_PUBLIC_IP>:8080

# Login: admin / admin123
```

### 3. Run Pipelines

**Step 1:** Run data processing DAG (`diab_medallion_ecs`)  
**Step 2:** Run model training DAG (`diab_model_training`)  
**Step 3:** Monitor progress in Airflow UI

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[How to Start Guide](docs/HOW_TO_START.md)** | Complete initial setup and configuration |
| **[DAG Architecture](docs/DAG_ARCHITECTURE.md)** | MLOps design decisions and best practices |
| **[Model Training Setup](docs/MODEL_TRAINING_SETUP.md)** | Training pipeline details and configuration |
| **[Deployment Guide](DEPLOYMENT.md)** | Docker build/push workflow |

## 🧪 ML Models

**Algorithms Trained:**
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

**Training Strategy:**
- **Temporal Window Split** (prevents data leakage):
  - Train: 1999-2005 (7 years)
  - Test: 2006-2007 (2 years)
  - OOT: 2008 (1 year)
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold CV
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC

**Model Registry:** S3-based versioned storage with metadata tracking

## 🗂️ Project Structure

```
.
├── airflow/
│   └── dags/              # Airflow DAG definitions
├── conf/                  # Configuration files
│   └── model_config.json  # Model training configuration
├── data/                  # Raw CSV data files
├── docs/                  # Documentation
├── infra/
│   └── terraform/         # Infrastructure as Code
├── utils/                 # Python utility modules
│   ├── data_processing_bronze_table.py
│   ├── data_processing_silver_table.py
│   └── data_processing_gold_table.py
├── main.py               # PySpark ETL entry point
├── model_train.py        # Model training script
├── Dockerfile            # Container image definition
├── docker-compose.yaml   # Airflow local setup
└── requirements.txt      # Python dependencies
```

## 🔧 Technology Stack

**Data Processing:**
- Apache Spark 3.x (PySpark)
- AWS S3 (data lake)
- Parquet (storage format)

**ML Training:**
- scikit-learn
- XGBoost
- pandas, numpy

**Orchestration:**
- Apache Airflow 2.9.3
- AWS ECS Fargate (compute)
- Docker containers

**Infrastructure:**
- Terraform (IaC)
- AWS EC2 (Airflow)
- AWS ECR (image registry)
- AWS CloudWatch (logging)

## 📊 Data Pipeline

### Bronze Layer (Raw)
- Source: CSV files from hospital systems
- Storage: S3 Parquet
- Transformations: Schema enforcement, basic validation

### Silver Layer (Cleaned)
- Deduplication, data quality checks
- Standardization of values
- Missing value handling

### Gold Layer (Features)
- Feature engineering for ML
- Label creation (30-day readmission)
- Separate feature_store and label_store

## ⚙️ Configuration

**Model Config** (`conf/model_config.json`):
```json
{
  "temporal_splits": {
    "train": {"start_date": "1999-01-01", "end_date": "2005-12-31"},
    "test": {"start_date": "2006-01-01", "end_date": "2007-12-31"},
    "oot": {"start_date": "2008-01-01", "end_date": "2008-12-31"}
  },
  "model_config": {
    "model_registry_bucket": "diab-readmit-123456-model-registry",
    "cv_folds": 5,
    "n_iter_search": 20
  },
  "training_config": {
    "algorithms": ["logistic_regression", "random_forest", "xgboost"]
  }
}
```

**Airflow Environment** (`airflow/airflow.env`):
- AWS region and credentials
- ECS cluster name
- **Two ECS task definitions:**
  - `ECS_TASK_DEF` - Data processing (1vCPU/2GB)
  - `ECS_MODEL_TRAINING_TASK_DEF` - Model training (2vCPU/4GB)
- S3 bucket paths

## 🐛 Troubleshooting

**Common Issues:**

| Issue | Solution |
|-------|----------|
| ECS task fails with exit code 1 | Check CloudWatch logs, verify config in S3 |
| Model training slow | Verify using model training task def (2vCPU/4GB) |
| Airflow UI not accessible | Check EC2 security group allows port 8080 |
| DAG not visible | Restart scheduler: `docker-compose restart airflow-scheduler` |
| S3 Access Denied | Verify ECS task role has S3 permissions |

See **[Troubleshooting Section](docs/HOW_TO_START.md#troubleshooting)** for detailed solutions.

## 📈 Model Performance Tracking

Models are evaluated on:
- **Test Set** (2006-2007): Temporal validation
- **Out-of-Time (OOT) Set** (2008): Production-like evaluation

Metrics saved to S3 model registry with each trained model.

## 🔒 Security & Best Practices

- ✅ IAM roles for service-to-service authentication (no hardcoded credentials)
- ✅ VPC with security groups for network isolation
- ✅ S3 bucket policies for access control
- ✅ CloudWatch for centralized logging
- ✅ Terraform state management
- ✅ Temporal data split prevents data leakage

## 🚧 Roadmap

- [x] Data processing pipeline (bronze → silver → gold)
- [x] Model training with temporal windows
- [x] S3-based model registry
- [ ] Batch inference pipeline
- [ ] Model monitoring and drift detection
- [ ] Real-time inference API
- [ ] A/B testing framework
- [ ] Automated retraining triggers

## 📝 Contributing

1. Create feature branch from `feature/airflow_aws_pipeline`
2. Make changes and test locally
3. Update documentation if needed
4. Submit pull request

## 📧 Contact

**Project Team:** ML Engineering Team  
**Course:** SMU MITB Term-4 MLE  
**Repository:** [github.com/vishal2505/Predicting_Hospital_Readmission_Risk](https://github.com/vishal2505/Predicting_Hospital_Readmission_Risk)

---

**Last Updated:** October 31, 2025  
**Version:** 1.0.0  
**License:** MIT
