#!/usr/bin/env bash
set -euo pipefail

# Complete bootstrap for EC2 (Amazon Linux 2023) to run Airflow via Docker Compose
# This script handles everything: package installation, Docker setup, repo cloning, and Airflow startup

echo "[1/5] Installing system packages (git, amazon-ssm-agent, ec2-instance-connect)" >&2
sudo dnf install -y git amazon-ssm-agent ec2-instance-connect
sudo systemctl enable amazon-ssm-agent && sudo systemctl start amazon-ssm-agent
sudo systemctl enable sshd && sudo systemctl start sshd

echo "[2/5] Installing and starting Docker" >&2
sudo dnf install -y docker
sudo systemctl enable docker
sudo systemctl start docker

# Verify docker is running
sleep 2
if ! sudo docker info &>/dev/null; then
  echo "ERROR: Docker is not running properly" >&2
  sudo systemctl status docker
  exit 1
fi

# Check if docker compose command is available (Docker v2 includes compose)
if ! sudo docker compose version &>/dev/null; then
  echo "Docker Compose v2 not available, installing docker-compose v1..." >&2
  sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  COMPOSE_CMD="docker-compose"
else
  echo "Docker Compose v2 detected" >&2
  COMPOSE_CMD="docker compose"
fi

echo "[3/5] Cloning repo if missing"
if [ ! -d /opt/airflow/repo/.git ]; then
  sudo mkdir -p /opt/airflow
  cd /opt/airflow
  sudo rm -rf /opt/airflow/repo
  sudo git clone -b feature/airflow_aws_pipeline https://github.com/vishal2505/Predicting_Hospital_Readmission_Risk.git /opt/airflow/repo
fi

echo "[4/5] Starting Airflow services (init, webserver, scheduler)"
cd /opt/airflow/repo

sudo $COMPOSE_CMD -f airflow-docker-compose.yaml up -d airflow-init
sudo $COMPOSE_CMD -f airflow-docker-compose.yaml up -d airflow-webserver airflow-scheduler

echo "[5/5] Checking container status and logs"
sudo $COMPOSE_CMD -f airflow-docker-compose.yaml ps
CID=$(sudo docker ps --filter name=airflow --format '{{.ID}}' | head -n1 || true)
if [ -n "${CID:-}" ]; then
  sudo docker logs "$CID" --tail 200 || true
fi

echo "Done. Open Airflow on: http://<this-instance-public-ip>:8080" >&2
