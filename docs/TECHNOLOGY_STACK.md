# TECHNOLOGY STACK

Our comprehensive MLOps platform leverages a carefully curated technology stack designed for enterprise-grade machine learning operations. This architecture combines industry-leading frameworks for data processing, model training, deployment orchestration, and monitoring, ensuring scalability, reliability, and performance at every layer.

---

## Core Framework & Runtime

- **Python 3.12** runtime environment
- **OpenJDK 21** for Apache Spark execution

---

## Data Processing & Analytics

- **Apache Spark 3.5.5** (PySpark) for distributed data processing
- **Pandas 2.2.3** & **NumPy 2.2.2** for data manipulation
- **Apache Parquet** via PyArrow 14.0.1 for efficient columnar storage
- **Medallion Architecture** (Bronze-Silver-Gold) for data lakehouse pattern
- **S3A FileSystem** integration for cloud-native data access

---

## Machine Learning

- **scikit-learn 1.6.1** framework for traditional ML algorithms
- **XGBoost 3.0.0** gradient boosting framework
- **LightGBM 4.6.0** gradient boosting framework
- **Pickle** model serialization for deployment
- **GridSearchCV** & **RandomizedSearchCV** for hyperparameter tuning
- **Cross-validation** (5-fold) for robust model evaluation

---

## Cloud & Storage Infrastructure

- **Amazon S3** for scalable cloud storage
  - Datamart bucket (Bronze/Silver/Gold layers)
  - Model Registry bucket (trained models, metadata)
- **boto3 1.34.0** AWS SDK integration
- **Parquet format** for efficient data storage
- **Partitioned storage** optimized for distributed processing
- **S3 versioning** for data lineage and audit trails

---

## Orchestration & Workflow

- **Apache Airflow 2.10.3** for workflow management
- **Amazon ECS (Fargate)** for containerized task execution
  - Data Processing tasks: 1 vCPU / 2 GB memory
  - Model Training tasks: 2 vCPU / 4 GB memory
- **Custom DAG-based** job scheduling system
- **Automated pipeline** scheduling and dependencies
- **Real-time job** monitoring and alerts via Airflow UI
- **ECS task definitions** with auto-scaling capabilities

---

## Development & Containerization

- **Docker 27.x** and **Docker Compose 2.x** enable consistent, reproducible environments across development and production
- **Amazon ECR** for Docker image registry
- **Multi-stage builds** for optimized image sizes
- **JupyterLab 4.3.5** provides interactive notebooks for experimentation and analysis
- **Git version control** with feature branch workflow
- **Terraform 1.x** for Infrastructure as Code (IaC)

---

## Model Monitoring & Governance

- **Custom monitoring framework** with drift detection
  - **PSI (Population Stability Index)** for distribution drift
  - **CSI (Characteristic Stability Index)** for categorical features
  - **AUC-ROC** and **GINI** coefficient tracking
- **Automated governance** decision engine
  - Threshold-based retraining triggers
  - Configurable warning/critical levels
- **JSON-based reporting** stored in Model Registry bucket
- **Time-series metrics** tracking for trend analysis

---

## Model Inference & Serving

- **Batch inference** pipeline for scalable predictions
- **Automated model selection** (best model from comparison)
- **Preprocessing artifact** management (scalers, encoders)
- **Snapshot-based** temporal predictions
- **Manual upload** capability for ad-hoc predictions via DAG configuration

---

## Data Visualization & Reporting

- **Matplotlib 3.10.0** and **Seaborn 0.13.2** deliver publication-quality visualizations for model performance analysis, data exploration, and stakeholder reporting
- **JSON format** for structured monitoring/governance reports
- **Airflow UI** for DAG execution visualization and monitoring

---

## Testing & Quality Assurance

- **pytest** for unit and integration testing
- **pytest-cov** for code coverage analysis
- **Mock/MagicMock** for isolated unit testing
- **Integration tests** for end-to-end pipeline validation
- **CloudWatch Logs** for runtime monitoring and debugging

---

## Security & Compliance

- **AWS IAM** role-based access control
- **Security Groups** for network isolation
- **Encrypted S3 buckets** with server-side encryption
- **VPC isolation** for ECS tasks
- **Secrets Manager** for credential management
- **Audit logging** via CloudWatch

---

## Monitoring & Observability

- **Amazon CloudWatch** for centralized logging
  - Separate log groups for data processing and model training
  - Structured JSON logging for analysis
- **CloudWatch Metrics** for resource utilization
- **CloudWatch Alarms** for automated alerting
- **Airflow UI (port 8080)** for:
  - DAG execution monitoring and visualization
  - Task logs and debugging
  - Pipeline orchestration management
  - Model inference/monitoring/governance dashboards
- **ECS Task monitoring** for container health
- **S3-based monitoring reports** (JSON) for model performance tracking

---

## Infrastructure as Code

- **Terraform 1.x** for AWS infrastructure provisioning
  - VPC, Subnets, Security Groups
  - EC2 instance for Airflow
  - ECS cluster and task definitions
  - S3 buckets with lifecycle policies
  - IAM roles and policies
- **Modular Terraform** configuration for reusability
- **State management** with remote backend
- **Version-controlled** infrastructure definitions

---

## Configuration Management

- **JSON-based** model configuration (`model_config.json`)
- **Environment variables** for runtime configuration
- **Docker Compose** environment files (`.env`)
- **Airflow Variables** for DAG configuration
- **S3-stored configuration** for centralized management

---

## Key Architectural Patterns

### Medallion Architecture (Bronze-Silver-Gold)
- **Bronze Layer**: Raw data ingestion from source systems
- **Silver Layer**: Cleaned, validated, and standardized data
- **Gold Layer**: Feature store and label store for ML consumption

### Model Registry Pattern
- **Versioned models** with timestamp-based naming
- **Metadata tracking** (metrics, hyperparameters, training config)
- **Model comparison** framework for best model selection
- **Separate bucket** for model artifacts isolation

### Monitoring & Governance Pattern
- **Continuous monitoring** of deployed model performance
- **Drift detection** (data distribution and concept drift)
- **Automated governance** decisions (retrain/schedule/no-action)
- **JSON-based reporting** for easy consumption

### Microservices Architecture
- **Containerized services** for isolation and scalability
- **Separate task definitions** for different workloads
  - Data processing: 1 vCPU / 2 GB
  - Model training: 2 vCPU / 4 GB
- **Independent scaling** of components
- **Airflow orchestration** for service coordination

---

## Performance Optimization

- **Parquet columnar format** for 10x compression vs CSV
- **S3 partition pruning** for faster data access
- **Spark DataFrame** optimization with catalyst optimizer
- **Broadcast joins** for small dimension tables
- **Lazy evaluation** for efficient query planning
- **Resource-specific task definitions** (1vCPU vs 2vCPU)

---

## Development Workflow

1. **Local Development**: Docker Compose for consistent environment
2. **Version Control**: Git with feature branch workflow
3. **CI/CD**: Automated Docker builds and ECR pushes
4. **Testing**: pytest suite before deployment
5. **Infrastructure**: Terraform apply for AWS resources
6. **Deployment**: Docker image push to ECR
7. **Monitoring**: CloudWatch and Airflow UI observation

---

## Scalability Features

- **Horizontal scaling** with ECS Fargate auto-scaling
- **S3 unlimited storage** capacity
- **Spark distributed processing** for large datasets
- **Stateless containers** for easy replication
- **Load balancing** ready architecture
- **Multi-AZ deployment** capability

---

## Future Enhancements

- **Real-time streaming** with Kinesis Data Streams
- **Feature store** with Feast or SageMaker Feature Store
- **Real-time model serving** with SageMaker endpoints or FastAPI
- **Interactive UI** for inference and monitoring (Flask/Streamlit)
- **A/B testing** framework for model comparison
- **MLflow** integration for experiment tracking
- **Kubernetes** migration for advanced orchestration

---

**Technology Stack Version**: 2.0  
**Last Updated**: November 5, 2025  
**Project**: Hospital Readmission Risk Prediction - MLOps Platform  
**Maintained By**: ML Engineering Team
