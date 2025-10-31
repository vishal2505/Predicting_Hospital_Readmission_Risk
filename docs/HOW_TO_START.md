# How to Start Guide

## Complete Setup Guide for Hospital Readmission ML Pipeline

This guide walks you through the complete initial setup required before running any DAG in the hospital readmission prediction pipeline.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Infrastructure Setup](#aws-infrastructure-setup)
3. [S3 Buckets Setup](#s3-buckets-setup)
4. [Docker Image Build & Push](#docker-image-build--push)
5. [Airflow Setup on EC2](#airflow-setup-on-ec2)
6. [Configuration Files Upload](#configuration-files-upload)
7. [Initial Data Upload](#initial-data-upload)
8. [Verify Setup](#verify-setup)
9. [Running Your First DAG](#running-your-first-dag)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

Install the following on your local machine:

- **AWS CLI** (v2.x or higher)
  ```bash
  aws --version
  ```

- **Terraform** (v1.x or higher)
  ```bash
  terraform --version
  ```

- **Docker** (v20.x or higher)
  ```bash
  docker --version
  ```

- **Git**
  ```bash
  git --version
  ```

- **jq** (for JSON processing)
  ```bash
  # macOS
  brew install jq
  
  # Linux
  sudo apt-get install jq
  ```

### AWS Account Setup

1. **AWS Account Access**
   - Ensure you have an AWS account with admin permissions
   - Your IAM user should have permissions for: EC2, ECS, S3, VPC, IAM, CloudWatch

2. **AWS CLI Configuration**
   ```bash
   aws configure
   ```
   
   Provide:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region: `ap-southeast-1`
   - Default output format: `json`

3. **Verify AWS CLI Access**
   ```bash
   aws sts get-caller-identity
   ```
   
   Expected output:
   ```json
   {
       "UserId": "AIDAXXXXXXXXXXXXX",
       "Account": "503382476502",
       "Arn": "arn:aws:iam::503382476502:user/your-username"
   }
   ```

### Repository Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/vishal2505/Predicting_Hospital_Readmission_Risk.git
   cd Predicting_Hospital_Readmission_Risk
   git checkout feature/airflow_aws_pipeline
   ```

2. **Verify Directory Structure**
   ```bash
   ls -la
   ```
   
   Expected key directories:
   - `airflow/` - Airflow DAG files
   - `conf/` - Configuration files
   - `data/` - Raw data files
   - `infra/terraform/` - Terraform infrastructure code
   - `utils/` - Python utility scripts
   - `docs/` - Documentation

---

## AWS Infrastructure Setup

### Step 1: Navigate to Terraform Directory

```bash
cd infra/terraform/aws-ec2-airflow-ecs
```

### Step 2: Review Terraform Variables

Edit `terraform.tfvars` (or create if it doesn't exist):

```hcl
# terraform.tfvars
aws_region     = "ap-southeast-1"
project_name   = "diab-readmit"
environment    = "demo"

# EC2 Configuration
ec2_instance_type = "t3.medium"  # Minimum recommended for Airflow
ec2_key_name      = "your-key-pair-name"  # Your existing EC2 key pair

# ECS Configuration
ecs_task_cpu    = "1024"  # 1 vCPU
ecs_task_memory = "2048"  # 2 GB

# S3 Buckets
datamart_bucket_suffix = "123456"  # Will create: diab-readmit-123456-datamart
model_registry_bucket_suffix = "123456"  # Will create: diab-readmit-123456-model-registry

# VPC Configuration (optional - will create new VPC if not specified)
# vpc_id = "vpc-xxxxx"
# subnet_ids = ["subnet-xxxxx", "subnet-yyyyy"]
```

### Step 3: Initialize Terraform

```bash
terraform init
```

Expected output:
```
Initializing the backend...
Initializing provider plugins...
Terraform has been successfully initialized!
```

### Step 4: Review Infrastructure Plan

```bash
terraform plan
```

Review the resources that will be created:
- VPC, Subnets, Security Groups (if not provided)
- EC2 instance for Airflow
- ECS Cluster
- ECS Task Definition
- IAM Roles and Policies
- S3 Buckets (datamart and model registry)
- CloudWatch Log Groups

### Step 5: Apply Terraform Configuration

```bash
terraform apply
```

Type `yes` when prompted.

Expected duration: 5-10 minutes

### Step 6: Capture Terraform Outputs

```bash
terraform output -json > outputs.json
cat outputs.json | jq
```

**Important outputs to save:**
- `ec2_public_ip` - Airflow UI access
- `ec2_instance_id` - For SSH access
- `ecs_cluster_name` - ECS cluster name
- `ecs_task_definition_arn` - Task definition ARN
- `datamart_bucket_name` - S3 datamart bucket
- `model_registry_bucket_name` - S3 model registry bucket
- `ecr_repository_url` - ECR repository for Docker images

Save these values - you'll need them later:

```bash
# Extract key values
export EC2_PUBLIC_IP=$(terraform output -raw ec2_public_ip)
export ECS_CLUSTER=$(terraform output -raw ecs_cluster_name)
export ECS_TASK_DEF=$(terraform output -raw ecs_task_definition_arn)
export DATAMART_BUCKET=$(terraform output -raw datamart_bucket_name)
export MODEL_REGISTRY_BUCKET=$(terraform output -raw model_registry_bucket_name)
export ECR_REPO=$(terraform output -raw ecr_repository_url)

# Display values
echo "EC2 IP: $EC2_PUBLIC_IP"
echo "ECS Cluster: $ECS_CLUSTER"
echo "ECS Task: $ECS_TASK_DEF"
echo "Datamart Bucket: $DATAMART_BUCKET"
echo "Model Registry Bucket: $MODEL_REGISTRY_BUCKET"
echo "ECR Repository: $ECR_REPO"
```

---

## S3 Buckets Setup

### Verify S3 Buckets Created by Terraform

```bash
# List buckets
aws s3 ls | grep diab-readmit

# Should show:
# diab-readmit-123456-datamart
# diab-readmit-123456-model-registry
```

### Create Required S3 Directory Structure

```bash
# Navigate back to project root
cd /Users/vishalmishra/MyDocuments/SMU_MITB/Term-4/MLE/Project/Predicting_Hospital_Readmission_Risk

# Create directory structure in datamart bucket
aws s3api put-object \
  --bucket diab-readmit-123456-datamart \
  --key bronze/ \
  --region ap-southeast-1

aws s3api put-object \
  --bucket diab-readmit-123456-datamart \
  --key silver/ \
  --region ap-southeast-1

aws s3api put-object \
  --bucket diab-readmit-123456-datamart \
  --key gold/feature_store/ \
  --region ap-southeast-1

aws s3api put-object \
  --bucket diab-readmit-123456-datamart \
  --key gold/label_store/ \
  --region ap-southeast-1

aws s3api put-object \
  --bucket diab-readmit-123456-datamart \
  --key config/ \
  --region ap-southeast-1

# Create directory structure in model registry bucket
aws s3api put-object \
  --bucket diab-readmit-123456-model-registry \
  --key models/ \
  --region ap-southeast-1

aws s3api put-object \
  --bucket diab-readmit-123456-model-registry \
  --key metadata/ \
  --region ap-southeast-1
```

### Verify S3 Structure

```bash
# Check datamart bucket
aws s3 ls s3://diab-readmit-123456-datamart/ --region ap-southeast-1

# Expected output:
#                            PRE bronze/
#                            PRE config/
#                            PRE gold/
#                            PRE silver/

# Check model registry bucket
aws s3 ls s3://diab-readmit-123456-model-registry/ --region ap-southeast-1

# Expected output:
#                            PRE metadata/
#                            PRE models/
```

---

## Docker Image Build & Push

### Step 1: Authenticate Docker with ECR

```bash
# Get ECR repository URL from terraform output
export ECR_REPO=$(cd infra/terraform/aws-ec2-airflow-ecs && terraform output -raw ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin ${ECR_REPO}
```

Expected output: `Login Succeeded`

### Step 2: Build Docker Image

```bash
# From project root
docker build -t hospital-readmission-pipeline .
```

Expected duration: 5-10 minutes (first build)

Verify build success:
```bash
docker images | grep hospital-readmission-pipeline
```

### Step 3: Tag and Push to ECR

```bash
# Tag image
docker tag hospital-readmission-pipeline:latest ${ECR_REPO}:latest

# Push to ECR
docker push ${ECR_REPO}:latest
```

Expected duration: 2-5 minutes

### Step 4: Verify Image in ECR

```bash
aws ecr describe-images \
  --repository-name diab-readmit-pipeline \
  --region ap-southeast-1
```

Expected output should show at least one image with tag `latest`.

---

## Airflow Setup on EC2

### Step 1: SSH into EC2 Instance

```bash
# Get EC2 IP from terraform
export EC2_IP=$(cd infra/terraform/aws-ec2-airflow-ecs && terraform output -raw ec2_public_ip)

# SSH into instance (replace with your key pair)
ssh -i ~/.ssh/your-key-pair.pem ec2-user@${EC2_IP}
```

### Step 2: Clone Repository on EC2

```bash
# On EC2 instance
cd ~
git clone https://github.com/vishal2505/Predicting_Hospital_Readmission_Risk.git
cd Predicting_Hospital_Readmission_Risk
git checkout feature/airflow_aws_pipeline
```

### Step 3: Configure Airflow Environment Variables

Create or update `airflow/.env` file:

```bash
# On EC2 instance
cd ~/Predicting_Hospital_Readmission_Risk

cat > airflow/airflow.env << 'EOF'
# AWS Configuration
AWS_REGION=ap-southeast-1
AWS_DEFAULT_REGION=ap-southeast-1

# ECS Configuration
ECS_CLUSTER=diab-readmit-demo-cluster
ECS_TASK_DEF=diab-readmit-demo-pipeline
ECS_CONTAINER_NAME=app
ECS_SUBNETS=subnet-016ccc26ac506b800,subnet-0a8b7c9d0e1f2g3h4  # Replace with your subnets
ECS_SECURITY_GROUPS=sg-0123456789abcdef0  # Replace with your security group

# S3 Configuration
DATAMART_BASE_URI=s3a://diab-readmit-123456-datamart/
MODEL_CONFIG_S3_PATH=s3://diab-readmit-123456-datamart/config/model_config.json

# Airflow Configuration
AIRFLOW_UID=1000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin123  # Change this!
EOF
```

**Update these values with your actual values from terraform outputs:**
- `ECS_CLUSTER` - from terraform output `ecs_cluster_name`
- `ECS_TASK_DEF` - from terraform output `ecs_task_definition_arn` (just the family name, not the full ARN)
- `ECS_SUBNETS` - from terraform output `ecs_subnets`
- `ECS_SECURITY_GROUPS` - from terraform output `ecs_security_groups`

### Step 4: Update docker-compose.yaml

Ensure `docker-compose.yaml` references the `.env` file:

```bash
# On EC2 instance
cd ~/Predicting_Hospital_Readmission_Risk

# Check if docker-compose.yaml has env_file configured
grep -A 2 "env_file:" docker-compose.yaml
```

### Step 5: Start Airflow

```bash
# On EC2 instance
cd ~/Predicting_Hospital_Readmission_Risk

# Initialize Airflow database (first time only)
docker-compose up airflow-init

# Start Airflow services
docker-compose up -d

# Verify containers are running
docker-compose ps
```

Expected output - all services should be "Up":
```
NAME                  STATUS
airflow-scheduler     Up
airflow-webserver     Up
postgres              Up
redis                 Up
```

### Step 6: Access Airflow UI

Open browser and navigate to:
```
http://<EC2_PUBLIC_IP>:8080
```

Login credentials (from airflow.env):
- Username: `admin`
- Password: `admin123`

---

## Configuration Files Upload

### Step 1: Upload Model Configuration to S3

```bash
# From your local machine (project root)
cd /Users/vishalmishra/MyDocuments/SMU_MITB/Term-4/MLE/Project/Predicting_Hospital_Readmission_Risk

# Upload model config
aws s3 cp conf/model_config.json \
  s3://diab-readmit-123456-datamart/config/model_config.json \
  --region ap-southeast-1
```

### Step 2: Verify Config Upload

```bash
# Check file exists
aws s3 ls s3://diab-readmit-123456-datamart/config/ --region ap-southeast-1

# Download and verify content
aws s3 cp s3://diab-readmit-123456-datamart/config/model_config.json - | jq
```

Expected output should show the model configuration JSON with temporal_splits, model_config, etc.

### Step 3: Review Model Configuration

Ensure `conf/model_config.json` has correct values:

```json
{
  "temporal_splits": {
    "train": {
      "start_date": "1999-01-01",
      "end_date": "2005-12-31"
    },
    "test": {
      "start_date": "2006-01-01",
      "end_date": "2007-12-31"
    },
    "oot": {
      "start_date": "2008-01-01",
      "end_date": "2008-12-31"
    }
  },
  "model_config": {
    "model_registry_bucket": "diab-readmit-123456-model-registry",
    "model_registry_prefix": "models/",
    "random_state": 42,
    "cv_folds": 5,
    "n_iter_search": 20
  },
  "training_config": {
    "algorithms": ["logistic_regression", "random_forest", "xgboost"],
    "hyperparameter_tuning": true,
    "feature_selection": false
  },
  "monitoring_config": {
    "drift_threshold_psi": 0.2,
    "performance_threshold_drop": 0.05,
    "min_samples_for_metrics": 100
  }
}
```

**Update `model_registry_bucket` to match your actual bucket name!**

---

## Initial Data Upload

### Step 1: Prepare Raw Data Files

Ensure your raw diabetes dataset CSV files are in the `data/` directory:

```bash
ls -lh data/

# Expected files:
# diabetic_data.csv
# IDs_mapping.csv (optional)
```

### Step 2: Upload Raw Data to S3 Bronze Layer

```bash
# Upload raw CSV files to bronze layer
aws s3 cp data/diabetic_data.csv \
  s3://diab-readmit-123456-datamart/bronze/diabetic_data.csv \
  --region ap-southeast-1

# If you have additional files
aws s3 cp data/IDs_mapping.csv \
  s3://diab-readmit-123456-datamart/bronze/IDs_mapping.csv \
  --region ap-southeast-1
```

### Step 3: Verify Data Upload

```bash
# Check bronze layer
aws s3 ls s3://diab-readmit-123456-datamart/bronze/ --region ap-southeast-1 --human-readable

# Expected output:
# 2025-10-31 14:30:00   10.5 MiB diabetic_data.csv
```

---

## Verify Setup

### Checklist

Run through this checklist to ensure everything is set up correctly:

#### AWS Infrastructure
- [ ] Terraform applied successfully
- [ ] EC2 instance running and accessible
- [ ] ECS cluster created
- [ ] ECS task definition exists
- [ ] IAM roles created with correct permissions

```bash
# Verify EC2
aws ec2 describe-instances \
  --instance-ids $(cd infra/terraform/aws-ec2-airflow-ecs && terraform output -raw ec2_instance_id) \
  --region ap-southeast-1 \
  --query 'Reservations[0].Instances[0].State.Name'

# Expected: "running"

# Verify ECS cluster
aws ecs describe-clusters \
  --clusters diab-readmit-demo-cluster \
  --region ap-southeast-1 \
  --query 'clusters[0].status'

# Expected: "ACTIVE"
```

#### S3 Buckets
- [ ] Datamart bucket exists with bronze/silver/gold/config directories
- [ ] Model registry bucket exists with models/metadata directories
- [ ] Raw data uploaded to bronze layer
- [ ] model_config.json uploaded to config folder

```bash
# Verify buckets and structure
aws s3 ls s3://diab-readmit-123456-datamart/ --region ap-southeast-1
aws s3 ls s3://diab-readmit-123456-model-registry/ --region ap-southeast-1

# Verify config file
aws s3 ls s3://diab-readmit-123456-datamart/config/ --region ap-southeast-1
```

#### Docker & ECR
- [ ] Docker image built successfully
- [ ] Image pushed to ECR with `:latest` tag
- [ ] ECR repository accessible

```bash
# Verify ECR image
aws ecr describe-images \
  --repository-name diab-readmit-pipeline \
  --region ap-southeast-1 \
  --query 'imageDetails[?imageTags[?@==`latest`]]'
```

#### Airflow
- [ ] Airflow containers running on EC2
- [ ] Airflow UI accessible via browser
- [ ] DAG files visible in Airflow UI
- [ ] Environment variables configured in airflow.env

```bash
# SSH to EC2 and check
ssh -i ~/.ssh/your-key.pem ec2-user@${EC2_IP}

# On EC2:
cd ~/Predicting_Hospital_Readmission_Risk
docker-compose ps

# All services should show "Up"
```

#### Airflow DAGs Loaded
- [ ] Access Airflow UI at `http://<EC2_IP>:8080`
- [ ] Login with admin credentials
- [ ] Verify DAGs are visible:
  - `diab_medallion_ecs` (Data Processing)
  - `diab_model_training` (Model Training)

```bash
# On EC2, check DAG files
ls -la ~/Predicting_Hospital_Readmission_Risk/airflow/dags/

# Expected files:
# diab_pipeline.py (or diab_medallion_ecs.py)
# diab_model_training.py
```

---

## Running Your First DAG

### Step 1: Run Data Processing Pipeline

This DAG processes raw data from bronze → silver → gold layers.

1. **In Airflow UI:**
   - Navigate to DAGs page
   - Find `diab_medallion_ecs` DAG
   - Click the toggle to "Unpause" the DAG
   - Click the "▶" (Play) button to trigger manual run

2. **Monitor Progress:**
   - Click on the DAG name to see the graph view
   - Watch tasks turn green as they complete
   - Expected duration: 15-30 minutes

3. **Check Outputs:**
   ```bash
   # Verify silver layer data
   aws s3 ls s3://diab-readmit-123456-datamart/silver/ --recursive --human-readable | head

   # Verify gold layer data
   aws s3 ls s3://diab-readmit-123456-datamart/gold/feature_store/ --recursive --human-readable | head
   aws s3 ls s3://diab-readmit-123456-datamart/gold/label_store/ --recursive --human-readable | head
   ```

### Step 2: Run Model Training Pipeline

After data processing completes successfully:

1. **In Airflow UI:**
   - Navigate to DAGs page
   - Find `diab_model_training` DAG
   - Click the toggle to "Unpause" the DAG
   - Click the "▶" (Play) button to trigger manual run

2. **Monitor Progress:**
   - Watch prerequisite checks (config validation, data existence, data sufficiency)
   - If all checks pass, training task will start on ECS
   - Expected duration: 2-4 hours (with hyperparameter tuning)

3. **Check Training Logs:**
   - In Airflow UI, click on the `train_models` task
   - Click "Log" to see Airflow logs
   - For detailed Python logs, check CloudWatch:
   
   ```bash
   # Get log group name from task definition
   aws ecs describe-task-definition \
     --task-definition diab-readmit-demo-pipeline \
     --region ap-southeast-1 \
     --query 'taskDefinition.containerDefinitions[0].logConfiguration.options.awslogs-group'
   
   # Tail logs (replace LOG_GROUP with actual value)
   aws logs tail /aws/ecs/diab-readmit-demo-cluster \
     --since 1h \
     --follow \
     --region ap-southeast-1
   ```

4. **Verify Model Registry:**
   ```bash
   # Check trained models
   aws s3 ls s3://diab-readmit-123456-model-registry/models/ --recursive --human-readable

   # Expected output:
   # logistic_regression_v20251031_140530.pkl
   # random_forest_v20251031_143020.pkl
   # xgboost_v20251031_145510.pkl
   # metadata files...
   ```

### Step 3: Verify End-to-End Pipeline

1. **Check Data Flow:**
   ```bash
   # Bronze → Silver → Gold → Models
   
   # Count records at each layer (example commands)
   aws s3 ls s3://diab-readmit-123456-datamart/bronze/ --recursive | wc -l
   aws s3 ls s3://diab-readmit-123456-datamart/silver/ --recursive | wc -l
   aws s3 ls s3://diab-readmit-123456-datamart/gold/feature_store/ --recursive | wc -l
   aws s3 ls s3://diab-readmit-123456-model-registry/models/ --recursive | wc -l
   ```

2. **Download and Inspect Model Metadata:**
   ```bash
   # Download latest model metadata
   aws s3 cp s3://diab-readmit-123456-model-registry/models/logistic_regression_latest_metadata.json - | jq
   ```

3. **Expected Metadata Structure:**
   ```json
   {
     "model_name": "logistic_regression",
     "training_date": "2025-10-31T14:05:30.123456",
     "temporal_splits": {...},
     "feature_count": 50,
     "training_samples": 50000,
     "test_samples": 15000,
     "oot_samples": 7500,
     "performance": {
       "test": {
         "accuracy": 0.85,
         "auc_roc": 0.89,
         "precision": 0.82,
         "recall": 0.78,
         "f1": 0.80
       },
       "oot": {
         "accuracy": 0.83,
         "auc_roc": 0.87,
         ...
       }
     }
   }
   ```

---

## Troubleshooting

### Issue: Terraform Apply Fails

**Error:** "Error creating EC2 instance" or "Invalid key pair"

**Solution:**
1. Ensure your EC2 key pair exists in the region:
   ```bash
   aws ec2 describe-key-pairs --region ap-southeast-1
   ```

2. Create key pair if missing:
   ```bash
   aws ec2 create-key-pair \
     --key-name my-airflow-key \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/my-airflow-key.pem
   
   chmod 400 ~/.ssh/my-airflow-key.pem
   ```

3. Update `terraform.tfvars` with correct key name

---

### Issue: Cannot Access Airflow UI

**Error:** Browser shows "Connection refused" or timeout

**Solution:**
1. Check EC2 security group allows inbound traffic on port 8080:
   ```bash
   # Get security group ID
   aws ec2 describe-instances \
     --instance-ids $(cd infra/terraform/aws-ec2-airflow-ecs && terraform output -raw ec2_instance_id) \
     --region ap-southeast-1 \
     --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId'
   
   # Check inbound rules
   aws ec2 describe-security-groups \
     --group-ids sg-xxxxx \
     --region ap-southeast-1
   ```

2. Add rule to allow port 8080 from your IP:
   ```bash
   aws ec2 authorize-security-group-ingress \
     --group-id sg-xxxxx \
     --protocol tcp \
     --port 8080 \
     --cidr $(curl -s https://checkip.amazonaws.com)/32 \
     --region ap-southeast-1
   ```

3. Verify Airflow containers are running on EC2:
   ```bash
   ssh -i ~/.ssh/your-key.pem ec2-user@${EC2_IP}
   docker-compose ps
   ```

---

### Issue: DAG Not Visible in Airflow UI

**Error:** DAG files not showing up

**Solution:**
1. SSH to EC2 and check DAG files exist:
   ```bash
   ls -la ~/Predicting_Hospital_Readmission_Risk/airflow/dags/
   ```

2. Check docker-compose.yaml mounts the dags directory:
   ```bash
   grep -A 5 "volumes:" docker-compose.yaml | grep dags
   ```

3. Restart Airflow scheduler:
   ```bash
   docker-compose restart airflow-scheduler
   ```

4. Check scheduler logs for errors:
   ```bash
   docker-compose logs airflow-scheduler | tail -50
   ```

---

### Issue: ECS Task Fails Immediately

**Error:** Task exits with code 1, "Essential container exited"

**Common Causes & Solutions:**

1. **Missing model_config.json in S3:**
   ```bash
   # Verify config exists
   aws s3 ls s3://diab-readmit-123456-datamart/config/model_config.json
   
   # Upload if missing
   aws s3 cp conf/model_config.json \
     s3://diab-readmit-123456-datamart/config/model_config.json \
     --region ap-southeast-1
   ```

2. **IAM Permission Issues:**
   - ECS task role must have S3 read/write permissions
   - Check CloudWatch logs for "Access Denied" errors
   
   ```bash
   # View CloudWatch logs
   aws logs tail /aws/ecs/diab-readmit-demo-cluster \
     --since 30m \
     --region ap-southeast-1
   ```

3. **Docker Image Not Updated:**
   - After code changes, rebuild and push image
   
   ```bash
   # Rebuild and push
   docker build -t hospital-readmission-pipeline .
   docker tag hospital-readmission-pipeline:latest ${ECR_REPO}:latest
   docker push ${ECR_REPO}:latest
   ```

4. **Missing Environment Variables:**
   - Check `airflow/airflow.env` has all required variables
   - Verify DAG passes environment variables to ECS task

---

### Issue: Model Training Takes Too Long

**Error:** Task timeout after 4 hours

**Solutions:**

1. **Reduce Hyperparameter Search Space:**
   Edit `conf/model_config.json`:
   ```json
   {
     "model_config": {
       "cv_folds": 3,        // Reduce from 5 to 3
       "n_iter_search": 10   // Reduce from 20 to 10
     }
   }
   ```

2. **Train Fewer Algorithms:**
   ```json
   {
     "training_config": {
       "algorithms": ["logistic_regression"]  // Start with just one
     }
   }
   ```

3. **Increase ECS Task Resources:**
   - Update task definition to use 2 vCPU / 4 GB memory
   - Rebuild terraform with updated variables

---

### Issue: S3 Access Denied Errors

**Error:** "An error occurred (AccessDenied) when calling..."

**Solution:**
1. Verify IAM role attached to ECS task has S3 permissions:
   ```bash
   # Get task role ARN
   aws ecs describe-task-definition \
     --task-definition diab-readmit-demo-pipeline \
     --region ap-southeast-1 \
     --query 'taskDefinition.taskRoleArn'
   
   # List attached policies
   aws iam list-attached-role-policies \
     --role-name <role-name-from-arn>
   ```

2. Ensure policy includes:
   ```json
   {
     "Effect": "Allow",
     "Action": [
       "s3:GetObject",
       "s3:PutObject",
       "s3:ListBucket"
     ],
     "Resource": [
       "arn:aws:s3:::diab-readmit-123456-datamart/*",
       "arn:aws:s3:::diab-readmit-123456-model-registry/*"
     ]
   }
   ```

---

## Next Steps

After successful setup and first DAG runs:

1. **Schedule Regular Runs:**
   - Update `schedule_interval` in DAG definitions
   - Data processing: `@daily` or `@weekly`
   - Model training: `@monthly` or triggered by data processing

2. **Set Up Monitoring:**
   - Configure CloudWatch alarms for DAG failures
   - Set up email/Slack notifications in Airflow
   - Review model performance metrics

3. **Implement Inference Pipeline:**
   - Create `diab_model_inference` DAG
   - Generate batch predictions on new data
   - Save predictions to S3

4. **Implement Monitoring Pipeline:**
   - Create `diab_model_monitoring` DAG
   - Track model performance over time
   - Detect data drift and concept drift
   - Trigger retraining when needed

5. **Production Hardening:**
   - Enable Airflow authentication (LDAP/OAuth)
   - Set up HTTPS for Airflow UI
   - Configure database backups
   - Implement CI/CD for DAG deployments

---

## Useful Commands Reference

### AWS CLI

```bash
# List S3 buckets
aws s3 ls

# List bucket contents
aws s3 ls s3://bucket-name/prefix/ --recursive --human-readable

# Copy file to S3
aws s3 cp local-file.txt s3://bucket-name/path/

# Copy file from S3
aws s3 cp s3://bucket-name/path/file.txt ./

# Describe EC2 instance
aws ec2 describe-instances --instance-ids i-xxxxx

# Describe ECS task
aws ecs describe-tasks --cluster cluster-name --tasks task-id

# View CloudWatch logs
aws logs tail /aws/ecs/log-group --since 1h --follow
```

### Docker

```bash
# Build image
docker build -t image-name .

# List images
docker images

# Tag image
docker tag source-image:tag target-image:tag

# Push to ECR
docker push ecr-url:tag

# Run container locally
docker run -it --rm image-name bash
```

### Terraform

```bash
# Initialize
terraform init

# Plan changes
terraform plan

# Apply changes
terraform apply

# Show outputs
terraform output

# Destroy resources
terraform destroy
```

### Airflow (on EC2)

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f service-name

# Restart service
docker-compose restart service-name

# List running containers
docker-compose ps
```

---

## Additional Resources

- **Architecture Documentation:** [DAG_ARCHITECTURE.md](./DAG_ARCHITECTURE.md)
- **Model Training Setup:** [MODEL_TRAINING_SETUP.md](./MODEL_TRAINING_SETUP.md)
- **Deployment Guide:** [DEPLOYMENT.md](../DEPLOYMENT.md)
- **Airflow Documentation:** https://airflow.apache.org/docs/
- **AWS ECS Documentation:** https://docs.aws.amazon.com/ecs/
- **Terraform AWS Provider:** https://registry.terraform.io/providers/hashicorp/aws/

---

**Document Version:** 1.0  
**Last Updated:** October 31, 2025  
**Maintained By:** ML Engineering Team

**For Issues/Questions:**
- Create GitHub issue in repository
- Contact: ml-team@yourcompany.com
