"""
Model Governance Script - Production Version
Makes automated decisions about model retraining based on monitoring results

This script:
1. Reads monitoring results from model-registry bucket (JSON)
2. Extracts latest snapshot metrics (AUC, PSI)
3. Applies governance thresholds:
   - AUC < 0.70 OR PSI > 0.25 → Retrain (critical)
   - PSI > 0.10 → Schedule Retrain (warning)
   - Otherwise → No Action (healthy)
4. Saves governance decision as JSON to model-registry bucket
5. Optionally triggers retraining (if auto_retrain=true)

Environment Variables:
    AWS_REGION: AWS region (default: ap-southeast-1)
    DATAMART_BASE_URI: S3 base URI (e.g., s3a://bucket/prefix/)
    MODEL_ALGORITHM: Model algorithm name (optional, auto-selects best if not provided)
    AUTO_RETRAIN: Enable auto-retrain trigger (default: false)
    MODEL_CONFIG_S3_PATH: S3 path to model config (for auto-selection)

Governance Thresholds:
    AUC_THRESHOLD = 0.70 (below this triggers retrain)
    PSI_WARNING = 0.10 (moderate drift)
    PSI_CRITICAL = 0.25 (significant drift, triggers retrain)
"""

import os
import sys
import json
import boto3
from datetime import datetime

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
MODEL_CONFIG_S3_PATH = os.environ.get("MODEL_CONFIG_S3_PATH", "s3://diab-readmit-123456-datamart/config/model_config.json")
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "false").lower() == "true"

# Model algorithm - can be overridden, otherwise auto-selects best
MODEL_ALGORITHM_OVERRIDE = os.environ.get("MODEL_ALGORITHM", None)

# Ensure S3A protocol
if DATAMART_BASE_URI.startswith("s3://"):
    DATAMART_BASE_URI = DATAMART_BASE_URI.replace("s3://", "s3a://")

# Governance thresholds
AUC_THRESHOLD = 0.70
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25


def get_best_model_algorithm():
    """
    Get the best model algorithm from model comparison results
    Same logic as model monitoring
    
    Returns:
        str: Model algorithm name (e.g., 'xgboost')
    """
    print("=" * 80)
    print("Identifying Best Model")
    print("=" * 80)
    
    # If MODEL_ALGORITHM is explicitly provided, use it
    if MODEL_ALGORITHM_OVERRIDE:
        print(f"✓ Using explicitly specified model: {MODEL_ALGORITHM_OVERRIDE}")
        return MODEL_ALGORITHM_OVERRIDE
    
    # Otherwise, load from model comparison
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Try to get model registry info from config
    try:
        # Step 1: Parse config S3 path (config is in datamart bucket)
        config_s3_path = MODEL_CONFIG_S3_PATH.replace("s3://", "").replace("s3a://", "")
        config_bucket = config_s3_path.split("/")[0]
        config_key = "/".join(config_s3_path.split("/")[1:])
        
        print(f"Step 1: Loading config from s3://{config_bucket}/{config_key}")
        response = s3_client.get_object(Bucket=config_bucket, Key=config_key)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        # Step 2: Extract model registry bucket from config (different from config bucket!)
        model_registry_bucket = config["model_config"]["model_registry_bucket"]
        model_registry_prefix = config["model_config"]["model_registry_prefix"]
        
        print(f"Step 2: Model registry is at s3://{model_registry_bucket}/{model_registry_prefix}")
        
        # Step 3: Load comparison from model registry bucket
        comparison_key = f"{model_registry_prefix}latest_model_comparison.json"
        print(f"Step 3: Loading comparison from s3://{model_registry_bucket}/{comparison_key}")
        
        response = s3_client.get_object(Bucket=model_registry_bucket, Key=comparison_key)
        comparison = json.loads(response['Body'].read().decode('utf-8'))
        
        recommended_model = comparison.get('recommended_model')
        recommendation_reason = comparison.get('recommendation_reason')
        
        print(f"✓ Best Model: {recommended_model}")
        print(f"  Reason: {recommendation_reason}")
        print()
        
        return recommended_model
        
    except Exception as e:
        print(f"⚠ Could not load model comparison: {e}")
        print("  Falling back to default: xgboost")
        print()
        return "xgboost"


def make_governance_decision():
    """
    Main governance function
    
    Workflow:
    1. Identify best model algorithm (or use override)
    2. Load model registry config
    3. Read monitoring results from model-registry bucket (JSON)
    4. Get latest snapshot metrics
    5. Apply governance thresholds
    6. Make decision (Retrain / Schedule Retrain / No Action)
    7. Save governance decision as JSON to model-registry bucket
    8. Optionally trigger retraining
    """
    print("=" * 80)
    print("Model Governance")
    print("=" * 80)
    
    # Load model registry config
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    config_s3_path = MODEL_CONFIG_S3_PATH.replace("s3://", "").replace("s3a://", "")
    config_bucket = config_s3_path.split("/")[0]
    config_key = "/".join(config_s3_path.split("/")[1:])
    
    response = s3_client.get_object(Bucket=config_bucket, Key=config_key)
    config = json.loads(response['Body'].read().decode('utf-8'))
    model_registry_bucket = config["model_config"]["model_registry_bucket"]
    model_registry_prefix = config["model_config"]["model_registry_prefix"]
    
    # Get the model algorithm to govern
    MODEL_ALGORITHM = get_best_model_algorithm()
    
    print(f"Governing Model: {MODEL_ALGORITHM}")
    print(f"Auto-Retrain: {AUTO_RETRAIN}")
    print()
    
    # --- Read monitoring data from model registry ---
    print("Loading monitoring results from model registry...")
    monitoring_key = f"{model_registry_prefix}monitoring/{MODEL_ALGORITHM}_latest_monitoring.json"
    
    try:
        response = s3_client.get_object(Bucket=model_registry_bucket, Key=monitoring_key)
        monitoring_data = json.loads(response['Body'].read().decode('utf-8'))
        
        print(f"✓ Loaded monitoring from s3://{model_registry_bucket}/{monitoring_key}")
        print(f"  Last updated: {monitoring_data['last_updated']}")
        print(f"  Latest snapshot: {monitoring_data['latest_snapshot_month']}")
        print()
        
    except Exception as e:
        print(f"✗ Error loading monitoring results: {e}")
        print("  Make sure monitoring task has run successfully")
        sys.exit(1)
    
    # --- Extract latest metrics ---
    print("Analyzing latest metrics...")
    latest_metrics = monitoring_data['latest_metrics']
    snapshot_month = monitoring_data['latest_snapshot_month']
    
    auc = latest_metrics.get('auc')
    psi = latest_metrics.get('psi')
    gini = latest_metrics.get('gini')
    row_count = latest_metrics.get('row_count')
    
    print(f"✓ Latest snapshot: {snapshot_month}\n")
    
    print("=" * 80)
    print("Latest Metrics")
    print("=" * 80)
    print(f"📅 Snapshot Month: {snapshot_month}")
    print(f"📊 Records: {row_count:,}")
    if auc is not None:
        print(f"📈 AUC: {auc:.4f} (threshold: {AUC_THRESHOLD:.2f})")
        gini = 2 * auc - 1
        print(f"📉 GINI: {gini:.4f}")
    else:
        print(f"📈 AUC: N/A")
    
    if psi is not None:
        print(f"🔄 PSI: {psi:.4f} (warning: {PSI_WARNING:.2f}, critical: {PSI_CRITICAL:.2f})")
    else:
        print(f"🔄 PSI: N/A")
    
    print()
    
    # --- Apply governance logic ---
    print("=" * 80)
    print("Governance Decision")
    print("=" * 80)
    
    retrain_needed = False
    decision = "No Action"
    reason = []
    
    # Check AUC threshold
    if auc is not None and auc < AUC_THRESHOLD:
        decision = "Retrain"
        retrain_needed = True
        reason.append(f"AUC ({auc:.4f}) below threshold ({AUC_THRESHOLD:.2f})")
        print(f"🚨 Critical: AUC degradation detected")
    
    # Check PSI thresholds
    if psi is not None:
        if psi > PSI_CRITICAL:
            decision = "Retrain"
            retrain_needed = True
            reason.append(f"PSI ({psi:.4f}) exceeds critical threshold ({PSI_CRITICAL:.2f})")
            print(f"🚨 Critical: Significant model drift detected")
        elif psi > PSI_WARNING and decision != "Retrain":
            decision = "Schedule Retrain"
            reason.append(f"PSI ({psi:.4f}) exceeds warning threshold ({PSI_WARNING:.2f})")
            print(f"⚠️ Warning: Moderate model drift detected")
    
    if decision == "No Action":
        print(f"✅ Model is healthy - no action required")
    
    # Print decision summary
    print()
    print(f"Decision: {decision}")
    if reason:
        print(f"Reason: {'; '.join(reason)}")
    print()
    
    # --- Save governance decision ---
    print("=" * 80)
    print("Saving Governance Decision")
    print("=" * 80)
    
    try:
        # Create governance decision JSON
        governance_json = {
            "algorithm": MODEL_ALGORITHM,
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "latest_snapshot_month": snapshot_month,
            "metrics": {
                "auc": float(auc) if auc is not None else None,
                "gini": float(gini) if gini is not None else None,
                "psi": float(psi) if psi is not None else None,
                "row_count": int(row_count) if row_count is not None else None
            },
            "thresholds": {
                "auc_threshold": AUC_THRESHOLD,
                "psi_warning": PSI_WARNING,
                "psi_critical": PSI_CRITICAL
            },
            "decision": decision,
            "retrain_needed": retrain_needed,
            "reason": '; '.join(reason) if reason else "Model metrics within acceptable range",
            "auto_retrain_enabled": AUTO_RETRAIN
        }
        
        # Save to model registry bucket
        governance_key = f"{model_registry_prefix}governance/{MODEL_ALGORITHM}_latest_governance.json"
        s3_client.put_object(
            Bucket=model_registry_bucket,
            Key=governance_key,
            Body=json.dumps(governance_json, indent=2),
            ContentType='application/json'
        )
        
        governance_s3_path = f"s3://{model_registry_bucket}/{governance_key}"
        print(f"✓ Governance decision saved to:\n  {governance_s3_path}\n")
        
        # Print decision summary
        print("Governance Decision:")
        print(json.dumps(governance_json, indent=2))
        print()
        
    except Exception as e:
        print(f"✗ Error saving governance decision: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # --- Trigger retraining (optional) ---
    if retrain_needed and AUTO_RETRAIN:
        print("=" * 80)
        print("Auto-Retrain Trigger")
        print("=" * 80)
        print("🔄 Retraining required and AUTO_RETRAIN=true")
        print()
        print("Note: Automated retraining trigger would be implemented here.")
        print("Options:")
        print("  1. Trigger Airflow DAG via REST API")
        print("  2. Send SNS notification to start training pipeline")
        print("  3. Write trigger file to S3 that training DAG monitors")
        print()
        print("For production implementation, use Airflow TriggerDagRunOperator")
        print("or boto3 to invoke training DAG programmatically.")
        print()
        
        # Example: Write trigger file to S3 datamart
        trigger_key = f"triggers/retrain_trigger_{MODEL_ALGORITHM}.json"
        trigger_data = {
            "snapshot_month": snapshot_month,
            "decision": decision,
            "reason": '; '.join(reason),
            "triggered_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Use datamart bucket for trigger files (operational data)
        datamart_bucket = DATAMART_BASE_URI.replace("s3a://", "").replace("s3://", "").split("/")[0]
        s3_client.put_object(
            Bucket=datamart_bucket,
            Key=trigger_key,
            Body=json.dumps(trigger_data, indent=2),
            ContentType='application/json'
        )
        print(f"✓ Trigger file written to: s3://{datamart_bucket}/{trigger_key}")
        print()
    
    elif retrain_needed and not AUTO_RETRAIN:
        print("=" * 80)
        print("Manual Intervention Required")
        print("=" * 80)
        print("🔄 Retraining required but AUTO_RETRAIN=false")
        print()
        print("Action needed:")
        print("  1. Review governance decision in S3")
        print("  2. Manually trigger training DAG: diab_model_training")
        print("  3. Or set AUTO_RETRAIN=true to enable automatic retraining")
        print()
    
    print("=" * 80)
    print("Model Governance Complete")
    print("=" * 80)
    
    # Return decision for programmatic use
    return {
        "decision": decision,
        "retrain_needed": retrain_needed,
        "auc": auc,
        "psi": psi,
        "snapshot_month": snapshot_month
    }


if __name__ == "__main__":
    try:
        result = make_governance_decision()
        
        # Exit with code based on decision
        # 0 = No Action, 1 = Schedule Retrain, 2 = Retrain
        if result["decision"] == "Retrain":
            sys.exit(2)
        elif result["decision"] == "Schedule Retrain":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)
