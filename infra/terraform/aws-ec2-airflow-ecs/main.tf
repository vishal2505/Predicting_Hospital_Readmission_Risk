terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# Use default VPC and its subnets for simplicity
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  name_prefix = var.project_name
  # Auto-detect project root: if not provided, assume terraform is in infra/terraform/aws-ec2-airflow-ecs
  project_root = var.project_root != "" ? var.project_root : abspath("${path.module}/../../..")
}

# Security Group for Airflow EC2
resource "aws_security_group" "airflow_sg" {
  name        = "${local.name_prefix}-airflow-sg"
  description = "Allow SSH and Airflow UI"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  ingress {
    description = "Airflow UI"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security Group for ECS Tasks
resource "aws_security_group" "ecs_tasks_sg" {
  name        = "${local.name_prefix}-ecs-tasks-sg"
  description = "Allow ECS tasks to access AWS services"
  vpc_id      = data.aws_vpc.default.id

  # Allow all outbound traffic (needed for S3, ECR, CloudWatch)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${local.name_prefix}-ecs-tasks-sg"
  }
}

# IAM role for EC2 to call ECS RunTask and read S3/ECR if needed
resource "aws_iam_role" "airflow_ec2_role" {
  name               = "${local.name_prefix}-airflow-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "airflow_ec2_inline" {
  name = "${local.name_prefix}-airflow-ec2-inline"
  role = aws_iam_role.airflow_ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecs:RunTask",
          "ecs:Describe*",
          "ec2:Describe*",
          "iam:PassRole"
        ],
        Resource = "*"
      },
      {
        Effect   = "Allow",
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Resource = "*"
      },
      {
        Effect   = "Allow",
        Action   = ["s3:*"],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "airflow_profile" {
  name = "${local.name_prefix}-airflow-profile"
  role = aws_iam_role.airflow_ec2_role.name
}

# Find Amazon Linux 2023 AMI
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["137112412989"] # Amazon

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_instance" "airflow" {
  ami                    = "ami-0d678b26d3e31e016"
  instance_type          = var.ec2_instance_type
  subnet_id              = element(data.aws_subnets.default.ids, 0)
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.airflow_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.airflow_profile.name
  associate_public_ip_address = true
  tags = {
    Name = "${local.name_prefix}-airflow"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    
    # Install git to clone repo (needed for bootstrap)
    dnf install -y git

    # Clone repo to get setup script and env template
    mkdir -p /opt/airflow
    git clone -b feature/airflow_aws_pipeline https://github.com/vishal2505/Predicting_Hospital_Readmission_Risk.git /opt/airflow/repo
    
    # Populate airflow.env from template with Terraform values
    sed -e 's|__AWS_REGION__|${var.aws_region}|g' \
        -e 's|__ECS_CLUSTER__|${aws_ecs_cluster.pipeline.name}|g' \
        -e 's|__ECS_TASK_DEF__|${aws_ecs_task_definition.pipeline.family}:${aws_ecs_task_definition.pipeline.revision}|g' \
        -e 's|__ECS_SUBNETS__|${join(",", data.aws_subnets.default.ids)}|g' \
        -e 's|__ECS_SECURITY_GROUPS__|${aws_security_group.airflow_sg.id}|g' \
        -e 's|__DATAMART_BASE_URI__|${var.datamart_base_uri}|g' \
        -e 's|__ECS_CONTAINER_NAME__|${var.container_name}|g' \
        -e 's|__START_DATE__|${var.start_date}|g' \
        -e 's|__END_DATE__|${var.end_date}|g' \
        /opt/airflow/repo/ops/airflow/airflow.env.template > /opt/airflow/airflow.env
    
    # Run setup script (handles all package installation, Docker setup, and Airflow startup)
    chmod +x /opt/airflow/repo/ops/airflow/setup_airflow_ec2.sh
    /opt/airflow/repo/ops/airflow/setup_airflow_ec2.sh
  EOF
}

# ECR repository
resource "aws_ecr_repository" "repo" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# Build and push Docker image to ECR
resource "null_resource" "docker_image" {
  depends_on = [aws_ecr_repository.repo]

  triggers = {
    # Rebuild if any Python files, Dockerfile, or requirements change
    dockerfile_hash = filemd5("${local.project_root}/Dockerfile")
    requirements_hash = filemd5("${local.project_root}/requirements.txt")
    main_py_hash = filemd5("${local.project_root}/main.py")
    ecr_repo_url = aws_ecr_repository.repo.repository_url
  }

  provisioner "local-exec" {
    working_dir = local.project_root
    command     = <<-EOT
      echo "Building and pushing Docker image to ECR..."
      
      # Login to ECR
      aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.repo.repository_url}
      
      # Build the image
      docker build -t ${var.ecr_repo_name}:latest .
      
      # Tag the image
      docker tag ${var.ecr_repo_name}:latest ${aws_ecr_repository.repo.repository_url}:latest
      
      # Push to ECR
      docker push ${aws_ecr_repository.repo.repository_url}:latest
      
      echo "Docker image pushed successfully!"
    EOT
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "pipeline" {
  name = "${local.name_prefix}-cluster"
}

# CloudWatch Logs for ECS
resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 7
}

# Task roles
resource "aws_iam_role" "ecs_task_execution" {
  name               = "${local.name_prefix}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_exec_attach" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow SSM Session Manager access to the EC2 instance (so you can disable SSH if desired)
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.airflow_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Allow EC2 Instance Connect (browser-based SSH) to push ephemeral keys
// Note: EC2 Instance Connect policy must be attached to the IAM user/role initiating the
// browser-based SSH session, not to the instance role. Ensure your IAM identity has
// AmazonEC2InstanceConnect if you want to use the "EC2 Instance Connect" option in the console.

resource "aws_iam_role" "ecs_task_role" {
  name               = "${local.name_prefix}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "ecs_task_inline" {
  name = "${local.name_prefix}-ecs-task-inline"
  role = aws_iam_role.ecs_task_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ],
        Resource = [
          "arn:aws:s3:::diab-readmit-*",
          "arn:aws:s3:::diab-readmit-*/*"
        ]
      },
      {
        Effect = "Allow",
        Action = ["logs:CreateLogStream", "logs:PutLogEvents"],
        Resource = "*"
      }
    ]
  })
}

# ECS Task Definition
resource "aws_ecs_task_definition" "pipeline" {
  depends_on = [null_resource.docker_image]  # Wait for Docker image to be pushed
  
  family                   = "${local.name_prefix}-pipeline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = var.container_name,
      image     = "${aws_ecr_repository.repo.repository_url}:latest",
      essential = true,
      command   = ["python", "main.py"],
      environment = [
        { name = "AWS_REGION", value = var.aws_region },
        { name = "DATAMART_BASE_URI", value = var.datamart_base_uri },
        { name = "START_DATE", value = var.start_date },
        { name = "END_DATE", value = var.end_date },
        { name = "RUN_BRONZE", value = "true" },
        { name = "RUN_SILVER", value = "true" },
        { name = "RUN_GOLD", value = "true" }
      ],
      logConfiguration = {
        logDriver = "awslogs",
        options   = {
          awslogs-group         = aws_cloudwatch_log_group.pipeline.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}
