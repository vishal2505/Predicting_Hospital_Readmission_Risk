#!/bin/bash
# Open interactive shell in Docker container for debugging
# You can run commands manually to test each step

set -e

echo "=========================================="
echo "INTERACTIVE DOCKER SHELL"
echo "=========================================="

# Get AWS credentials
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-$(aws configure get aws_access_key_id)}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-$(aws configure get aws_secret_access_key)}
AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN:-$(aws configure get aws_session_token)}

if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "Error: AWS credentials not found. Run 'aws configure' first."
    exit 1
fi

echo "Building Docker image..."
docker build -t hospital-readmission-pipeline:local .

echo ""
echo "Starting interactive shell in container..."
echo ""
echo "You can now run:"
echo "  python model_train.py          # Run full training"
echo "  python -c 'from model_train import load_config; print(load_config())'  # Test config loading"
echo "  python -c 'from model_train import init_spark; spark = init_spark(); print(spark)'  # Test Spark"
echo ""
echo "Press Ctrl+D or type 'exit' to quit"
echo "=========================================="

docker run -it --rm \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  -e AWS_REGION="ap-southeast-1" \
  -e DATAMART_BASE_URI="s3a://diab-readmit-123456-datamart/" \
  -e MODEL_CONFIG_S3_URI="s3://diab-readmit-123456-datamart/config/model_config.json" \
  503382476502.dkr.ecr.ap-southeast-1.amazonaws.com/diab-readmit-pipeline:latest \
  /bin/bash
