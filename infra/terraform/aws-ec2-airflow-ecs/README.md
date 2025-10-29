# EC2 Airflow (Open Source) + ECS Fargate (Pipeline)

This Terraform deploys:
- A small EC2 instance running open-source Airflow (standalone, SQLite) via Docker.
- An ECR repository for your container image.
- An ECS cluster and a Fargate task definition pointing to your ECR image.

The Airflow UI runs on http://<public-ip>:8080 and triggers ECS tasks via AWS APIs.

## Prerequisites
- Terraform >= 1.5, AWS credentials configured.
- Your repo contains the Airflow DAG at `airflow/dags/diab_pipeline.py`.
- You will build and push your container image to ECR (instructions below).

## Usage

1) Initialize and deploy

```bash
cd infra/terraform/aws-ec2-airflow-ecs
terraform init
terraform apply -auto-approve \
  -var "datamart_base_uri=s3a://diab-readmit-123456-datamart/"
```

Outputs will include:
- `airflow_url` – open in your browser
- `ecr_repository_url` – use for docker push
- `ecs_cluster_name`, `ecs_task_definition` – used by the DAG automatically via env

2) Build and push your image (from project root)

```bash
# Get ECR URL from terraform output
ECR_URL=<account-id>.dkr.ecr.<region>.amazonaws.com/diab-readmit-pipeline
REGION=<region>

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URL"

docker build -t diab-readmit-pipeline:latest .

docker tag diab-readmit-pipeline:latest "$ECR_URL":latest

docker push "$ECR_URL":latest
```

3) Open Airflow UI
- Visit `airflow_url` in outputs. Airflow standalone may display an initial password in logs; default user is `airflow`.
- The DAG `diab_medallion_ecs` should appear (the instance clones your repo to mount DAGs – update the repo URL in `user_data` if needed).

4) Trigger the DAG
- Trigger manually, or add a schedule if desired.

5) Destroy when done

```bash
terraform destroy -auto-approve
```

## Notes and knobs
- Instance type: `t3.small` by default; change via `-var ec2_instance_type=t3.medium` if needed.
- No NAT costs: ECS tasks are launched with public IPs.
- Logs: CloudWatch Logs group `/ecs/<project>` with 7-day retention.
- Airflow provider: AWS provider is installed at container start.
- Security: By default port 8080 and 22 are open to the internet for demo. Override with `-var "allowed_cidr=<YOUR_IP/32>"` to restrict, or disable SSH and use AWS Systems Manager Session Manager.

## Wiring your repo for DAGs
The EC2 `user_data` clones a placeholder repo. Edit the URL in `main.tf` (user_data section) to your GitHub repo so the DAGs mount from `/opt/airflow/repo/airflow/dags`.

Alternatively, SSH to the instance and `git clone` your repo into `/opt/airflow/repo` and restart the container:

```bash
ssh ec2-user@<public-ip>
cd /opt/airflow
sudo git clone https://github.com/<owner>/<repo>.git repo
sudo docker compose restart
```

## Manual bootstrap on EC2 via SSH (if user_data didn’t run)
If your `/opt` directory is empty on the instance, run the provided setup script manually:

```bash
# On your local machine (copy script path)
# Then on the EC2 instance after SSH:
sudo bash /bin/bash -lc 'curl -fsSL https://raw.githubusercontent.com/vishal2505/Predicting_Hospital_Readmission_Risk/main/ops/airflow/setup_airflow_ec2.sh -o /tmp/setup_airflow_ec2.sh && chmod +x /tmp/setup_airflow_ec2.sh && /tmp/setup_airflow_ec2.sh'
```

Alternatively, run these commands step-by-step after SSH:

```bash
sudo dnf update -y
sudo dnf install -y docker docker-compose-plugin git curl
sudo systemctl enable docker && sudo systemctl start docker

sudo mkdir -p /opt/airflow && cd /opt/airflow
sudo git clone https://github.com/vishal2505/Predicting_Hospital_Readmission_Risk.git repo

cd /opt/airflow/repo
sudo docker compose -f airflow-docker-compose.yaml up -d
sudo docker compose -f airflow-docker-compose.yaml ps
```

Open: `http://<instance-public-ip>:8080`
