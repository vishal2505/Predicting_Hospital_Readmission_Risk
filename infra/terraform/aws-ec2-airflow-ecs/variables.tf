variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

variable "project_name" {
  description = "Prefix for resource names"
  type        = string
  default     = "diab-readmit-demo"
}

variable "allowed_cidr" {
  description = "CIDR allowed to access Airflow UI (8080) and SSH (22). Default opens to all (demo only)."
  type        = string
  default     = "0.0.0.0/0"
}

variable "ec2_instance_type" {
  description = "EC2 instance type for Airflow"
  type        = string
  default     = "t3.medium"
}

variable "key_pair_name" {
  description = "Name of the key pair to use for SSH access"
  type        = string
  default     = "diab_readmit_airflow"
}

variable "ecr_repo_name" {
  description = "ECR repository name for pipeline image"
  type        = string
  default     = "diab-readmit-pipeline"
}

variable "container_name" {
  description = "Container name defined in ECS task definition"
  type        = string
  default     = "app"
}

variable "datamart_base_uri" {
  description = "Base URI for datamart (local path or s3a://bucket/path)"
  type        = string
  default     = "s3a://diab-readmit-123456-datamart/"
}

variable "start_date" {
  description = "Backfill start date"
  type        = string
  default     = "1999-01-01"
}

variable "end_date" {
  description = "Backfill end date"
  type        = string
  default     = "2008-12-31"
}

variable "project_root" {
  description = "Absolute path to project root directory (where Dockerfile, main.py are located)"
  type        = string
  default     = ""  # Will be auto-detected using path.cwd if not provided
}
