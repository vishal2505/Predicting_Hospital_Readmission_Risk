#!/bin/bash
# Quick diagnostic script for Airflow connectivity issues

set -e

echo "🔍 Airflow Connection Diagnostics"
echo "=================================="
echo ""

# Step 1: Get instance details
echo "Step 1: Getting EC2 instance details..."
INSTANCE_DATA=$(aws ec2 describe-instances \
    --region ap-southeast-1 \
    --filters "Name=tag:Name,Values=diab-readmit-demo-airflow" \
    --query 'Reservations[0].Instances[0]' \
    --output json)

INSTANCE_ID=$(echo "$INSTANCE_DATA" | jq -r '.InstanceId')
PUBLIC_IP=$(echo "$INSTANCE_DATA" | jq -r '.PublicIpAddress')
STATE=$(echo "$INSTANCE_DATA" | jq -r '.State.Name')
LAUNCH_TIME=$(echo "$INSTANCE_DATA" | jq -r '.LaunchTime')

echo "   Instance ID: $INSTANCE_ID"
echo "   Public IP: $PUBLIC_IP"
echo "   State: $STATE"
echo "   Launch Time: $LAUNCH_TIME"
echo ""

if [ "$STATE" != "running" ]; then
    echo "❌ ERROR: Instance is not running!"
    exit 1
fi

# Step 2: Check Security Group
echo "Step 2: Checking Security Group rules..."
SG_ID=$(echo "$INSTANCE_DATA" | jq -r '.SecurityGroups[0].GroupId')
echo "   Security Group: $SG_ID"

PORT_8080_RULES=$(aws ec2 describe-security-groups \
    --region ap-southeast-1 \
    --group-ids "$SG_ID" \
    --query 'SecurityGroups[0].IpPermissions[?ToPort==`8080`]' \
    --output json)

if [ "$PORT_8080_RULES" == "[]" ]; then
    echo "❌ ERROR: Port 8080 is NOT open in Security Group!"
else
    echo "✅ Port 8080 is open"
    echo "$PORT_8080_RULES" | jq -r '.[0] | "   Port: \(.FromPort)-\(.ToPort), CIDR: \(.IpRanges[0].CidrIp)"'
fi
echo ""

# Step 3: Test network connectivity
echo "Step 3: Testing network connectivity..."
echo "   Testing connection to $PUBLIC_IP:8080..."

if timeout 5 bash -c "echo > /dev/tcp/$PUBLIC_IP/8080" 2>/dev/null; then
    echo "✅ Port 8080 is REACHABLE"
else
    echo "❌ Port 8080 is NOT REACHABLE (connection refused or timeout)"
    echo ""
    echo "This usually means:"
    echo "   1. Docker containers are not running on the instance"
    echo "   2. Airflow webserver failed to start"
    echo "   3. Webserver is not binding to 0.0.0.0:8080"
fi
echo ""

# Step 4: Check user_data execution
echo "Step 4: Checking user_data execution logs..."
echo "   Fetching cloud-init output log..."

COMMAND_ID=$(aws ssm send-command \
    --region ap-southeast-1 \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["tail -100 /var/log/cloud-init-output.log"]' \
    --output text \
    --query 'Command.CommandId')

echo "   Command ID: $COMMAND_ID (waiting 5 seconds...)"
sleep 5

aws ssm get-command-invocation \
    --region ap-southeast-1 \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' \
    --output text | tail -50

echo ""

# Step 5: Check Docker containers
echo "Step 5: Checking Docker containers..."

COMMAND_ID=$(aws ssm send-command \
    --region ap-southeast-1 \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["docker ps -a"]' \
    --output text \
    --query 'Command.CommandId')

echo "   Command ID: $COMMAND_ID (waiting 5 seconds...)"
sleep 5

DOCKER_OUTPUT=$(aws ssm get-command-invocation \
    --region ap-southeast-1 \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' \
    --output text)

echo "$DOCKER_OUTPUT"
echo ""

# Check if any airflow containers are running
if echo "$DOCKER_OUTPUT" | grep -q "airflow-webserver"; then
    echo "✅ Airflow webserver container exists"
    
    # Check container status
    if echo "$DOCKER_OUTPUT" | grep "airflow-webserver" | grep -q "Up"; then
        echo "✅ Airflow webserver is running"
    else
        echo "❌ Airflow webserver container exists but is not running!"
        echo ""
        echo "Fetching webserver logs..."
        COMMAND_ID=$(aws ssm send-command \
            --region ap-southeast-1 \
            --instance-ids "$INSTANCE_ID" \
            --document-name "AWS-RunShellScript" \
            --parameters 'commands=["docker logs airflow-webserver --tail 50"]' \
            --output text \
            --query 'Command.CommandId')
        sleep 5
        aws ssm get-command-invocation \
            --region ap-southeast-1 \
            --command-id "$COMMAND_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'StandardOutputContent' \
            --output text
    fi
else
    echo "❌ Airflow webserver container does NOT exist!"
    echo ""
    echo "This means the user_data script failed or didn't run docker compose."
fi

echo ""
echo "=================================="
echo "NEXT STEPS:"
echo "=================================="
echo ""
echo "If containers are not running, connect to the instance and debug:"
echo ""
echo "1. Connect via Session Manager:"
echo "   aws ssm start-session --region ap-southeast-1 --target $INSTANCE_ID"
echo ""
echo "2. Check if airflow.env file was created:"
echo "   cat /opt/airflow/airflow.env"
echo ""
echo "3. Check if repo was cloned:"
echo "   ls -la /opt/airflow/repo/"
echo ""
echo "4. Try running docker compose manually:"
echo "   cd /opt/airflow/repo"
echo "   sudo docker compose -f airflow-docker-compose.yaml up -d"
echo ""
echo "5. Check webserver logs:"
echo "   sudo docker logs airflow-webserver"
echo ""
echo "6. Check if webserver is listening:"
echo "   sudo docker exec airflow-webserver netstat -tlnp | grep 8080"
echo ""
