# last updated Mar 25 2025, 11:00am
FROM python:3.12-slim

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 21 headless), procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jdk-headless procps bash wget && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME directory and symlink the java binary there (only if missing)
    mkdir -p /usr/lib/jvm/java-21-openjdk-amd64/bin && \
    [ -f /usr/lib/jvm/java-21-openjdk-amd64/bin/java ] || ln -s "$(which java)" /usr/lib/jvm/java-21-openjdk-amd64/bin/java

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py ./
COPY model_train.py ./
COPY utils/ ./utils/
COPY data/ ./data/
COPY conf/ ./conf/

# --- ADDITIONS FOR S3 CONNECTIVITY ---
# Download the Hadoop-AWS and AWS SDK bundle JARs for S3a support.
RUN wget -O /tmp/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    wget -O /tmp/aws-java-sdk-bundle-1.12.639.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.639/aws-java-sdk-bundle-1.12.639.jar && \
    mkdir -p /opt/spark/jars-extra && \
    mv /tmp/hadoop-aws-3.3.4.jar /opt/spark/jars-extra/ && \
    mv /tmp/aws-java-sdk-bundle-1.12.639.jar /opt/spark/jars-extra/
# --- END ADDITIONS ---

# Default command - runs the main pipeline script
# Can be overridden in ECS task definition or docker run
CMD ["python", "main.py"]