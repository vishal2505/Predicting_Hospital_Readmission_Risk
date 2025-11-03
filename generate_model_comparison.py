"""
Generate Model Comparison Summary
Reads all trained models from S3 and creates comparison table
Run after all parallel training tasks complete

This script contains all model comparison logic, separate from training.
"""

import os
import json
import boto3
from datetime import datetime


def load_config(config_path="conf/model_config.json"):
    """Load model configuration from S3 or local file"""
    s3_uri = os.environ.get("MODEL_CONFIG_S3_URI")
    if s3_uri:
        if s3_uri.startswith("s3://"):
            try:
                _, _, rest = s3_uri.partition("s3://")
                bucket, _, key = rest.partition("/")
                local_tmp = "/tmp/model_config.json"
                s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
                print(f"Downloading model config from s3://{bucket}/{key}")
                s3.download_file(bucket, key, local_tmp)
                with open(local_tmp, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"✗ Error downloading config from {s3_uri}: {e}")
                raise

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    raise FileNotFoundError(f"Model config not found at {config_path} and no valid S3 config provided.")



def load_model_performance_from_s3(config):
    """
    Load performance metrics for all trained models from S3
    
    Returns:
        dict: {model_name: metadata_dict}
    """
    print("\n" + "=" * 80)
    print("Loading Model Performance Data from S3")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    bucket = config["model_config"]["model_registry_bucket"]
    base_prefix = config["model_config"]["model_registry_prefix"]
    
    # Get enabled algorithms
    enabled = config.get("training_config", {}).get("enabled_algorithms", {})
    algorithms = [alg for alg, is_enabled in enabled.items() if is_enabled]
    
    if not algorithms:
        # Fallback to old-style config
        algorithms = config.get("training_config", {}).get("algorithms", 
                                                           ["logistic_regression", "random_forest", "xgboost"])
    
    print(f"Looking for models: {algorithms}")
    
    all_results = {}
    
    for algorithm in algorithms:
        try:
            # Load latest performance.json
            performance_key = f"{base_prefix}{algorithm}/latest/performance.json"
            print(f"\n  Loading {algorithm}...")
            print(f"    Path: s3://{bucket}/{performance_key}")
            
            response = s3_client.get_object(Bucket=bucket, Key=performance_key)
            performance_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Load metadata for additional info (optional, for extra context)
            try:
                metadata_key = f"{base_prefix}{algorithm}/latest/metadata.json"
                response = s3_client.get_object(Bucket=bucket, Key=metadata_key)
                metadata = json.loads(response['Body'].read().decode('utf-8'))
            except:
                metadata = {}
            
            # Combine data in the format expected by generate_model_comparison()
            all_results[algorithm] = {
                'performance': performance_data.get('metrics', {}),
                'training_samples': performance_data.get('data_info', {}).get('training_samples', 0),
                'test_samples': performance_data.get('data_info', {}).get('test_samples', 0),
                'oot_samples': performance_data.get('data_info', {}).get('oot_samples', 0),
                'feature_importance': performance_data.get('feature_importance', {}),
                'training_date': performance_data.get('training_date', ''),
                'temporal_splits': performance_data.get('temporal_splits', {})
            }
            
            print(f"    ✓ Loaded successfully")
            print(f"      OOT AUC: {performance_data.get('metrics', {}).get('oot', {}).get('auc_roc', 0.0):.4f}")
            print(f"      OOT GINI: {performance_data.get('metrics', {}).get('oot', {}).get('gini', 0.0):.4f}")
        
        except s3_client.exceptions.NoSuchKey:
            print(f"    ⚠ Model not found (may not have been trained yet)")
        except Exception as e:
            print(f"    ✗ Error loading {algorithm}: {e}")
    
    print(f"\n✓ Loaded {len(all_results)} models")
    return all_results


def generate_model_comparison(all_results, config):
    """
    Generate comprehensive model comparison summary
    
    Args:
        all_results: Dictionary with structure {model_name: {metadata, performance}}
        config: Configuration dictionary
    
    Returns:
        dict: Comparison data with summary table and best model recommendation
    """
    print("\n" + "=" * 80)
    print("Generating Model Comparison Summary")
    print("=" * 80)
    
    comparison_rows = []
    
    for model_name, data in all_results.items():
        performance = data['performance']
        
        row = {
            'model': model_name,
            'train_samples': data.get('training_samples', 0),
            'test_samples': data.get('test_samples', 0),
            'oot_samples': data.get('oot_samples', 0),
            # Test metrics
            'test_auc': performance.get('test', {}).get('auc_roc', 0.0),
            'test_gini': performance.get('test', {}).get('gini', 0.0),
            'test_pr_auc': performance.get('test', {}).get('pr_auc', 0.0),
            'test_accuracy': performance.get('test', {}).get('accuracy', 0.0),
            'test_precision': performance.get('test', {}).get('precision', 0.0),
            'test_recall': performance.get('test', {}).get('recall', 0.0),
            'test_f1': performance.get('test', {}).get('f1', 0.0),
            # OOT metrics
            'oot_auc': performance.get('oot', {}).get('auc_roc', 0.0),
            'oot_gini': performance.get('oot', {}).get('gini', 0.0),
            'oot_pr_auc': performance.get('oot', {}).get('pr_auc', 0.0),
            'oot_accuracy': performance.get('oot', {}).get('accuracy', 0.0),
            'oot_precision': performance.get('oot', {}).get('precision', 0.0),
            'oot_recall': performance.get('oot', {}).get('recall', 0.0),
            'oot_f1': performance.get('oot', {}).get('f1', 0.0),
            # Feature importance summary
            'top_feature': data.get('feature_importance', {}).get('top_features', [{}])[0].get('feature', 'N/A') if data.get('feature_importance', {}).get('top_features') else 'N/A'
        }
        comparison_rows.append(row)
    
    # Sort by OOT GINI (descending)
    comparison_rows.sort(key=lambda x: x['oot_gini'], reverse=True)
    
    # Identify best models by different metrics
    best_models = {
        'oot_gini': max(comparison_rows, key=lambda x: x['oot_gini'])['model'],
        'oot_auc': max(comparison_rows, key=lambda x: x['oot_auc'])['model'],
        'oot_pr_auc': max(comparison_rows, key=lambda x: x['oot_pr_auc'])['model'],
        'oot_f1': max(comparison_rows, key=lambda x: x['oot_f1'])['model'],
        'test_gini': max(comparison_rows, key=lambda x: x['test_gini'])['model']
    }
    
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'training_date': list(all_results.values())[0].get('training_date', '') if all_results else '',
        'models_compared': len(comparison_rows),
        'comparison_table': comparison_rows,
        'best_models': best_models,
        'recommended_model': comparison_rows[0]['model'] if comparison_rows else 'N/A',
        'recommendation_reason': f"Best OOT GINI: {comparison_rows[0]['oot_gini']:.4f}" if comparison_rows else 'N/A',
        'temporal_splits': list(all_results.values())[0].get('temporal_splits', {}) if all_results else {}
    }
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Test AUC':<10} {'Test GINI':<11} {'OOT AUC':<10} {'OOT GINI':<11} {'OOT PR-AUC':<11}")
    print("-" * 80)
    
    for row in comparison_rows:
        print(f"{row['model']:<25} {row['test_auc']:<10.4f} {row['test_gini']:<11.4f} "
              f"{row['oot_auc']:<10.4f} {row['oot_gini']:<11.4f} {row['oot_pr_auc']:<11.4f}")
    
    print("\n" + "-" * 80)
    print(f"🏆 RECOMMENDED MODEL: {comparison_data['recommended_model']}")
    print(f"   Reason: {comparison_data['recommendation_reason']}")
    print(f"\n📊 Best Models by Metric:")
    print(f"   OOT GINI:   {best_models['oot_gini']}")
    print(f"   OOT AUC:    {best_models['oot_auc']}")
    print(f"   OOT PR-AUC: {best_models['oot_pr_auc']}")
    print(f"   OOT F1:     {best_models['oot_f1']}")
    print("=" * 80)
    
    return comparison_data


def save_model_comparison_to_s3(comparison_data, config):
    """
    Save model comparison summary to S3 in multiple formats
    Saves as JSON and CSV for easy access
    """
    print("\n" + "=" * 80)
    print("Saving Model Comparison to S3")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    bucket = config["model_config"]["model_registry_bucket"]
    base_prefix = config["model_config"]["model_registry_prefix"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON version
    json_key = f"{base_prefix}model_comparison_{timestamp}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=json_key,
        Body=json.dumps(comparison_data, indent=2).encode('utf-8')
    )
    print(f"✓ Saved JSON: s3://{bucket}/{json_key}")
    
    # Save CSV version for easy viewing in Excel/spreadsheets
    csv_content = "Model,Train_Samples,Test_Samples,OOT_Samples,"
    csv_content += "Test_AUC,Test_GINI,Test_PR_AUC,Test_Accuracy,Test_Precision,Test_Recall,Test_F1,"
    csv_content += "OOT_AUC,OOT_GINI,OOT_PR_AUC,OOT_Accuracy,OOT_Precision,OOT_Recall,OOT_F1,Top_Feature\n"
    
    for row in comparison_data['comparison_table']:
        csv_content += f"{row['model']},{row['train_samples']},{row['test_samples']},{row['oot_samples']},"
        csv_content += f"{row['test_auc']:.4f},{row['test_gini']:.4f},{row['test_pr_auc']:.4f},"
        csv_content += f"{row['test_accuracy']:.4f},{row['test_precision']:.4f},{row['test_recall']:.4f},{row['test_f1']:.4f},"
        csv_content += f"{row['oot_auc']:.4f},{row['oot_gini']:.4f},{row['oot_pr_auc']:.4f},"
        csv_content += f"{row['oot_accuracy']:.4f},{row['oot_precision']:.4f},{row['oot_recall']:.4f},{row['oot_f1']:.4f},"
        csv_content += f"{row['top_feature']}\n"
    
    csv_key = f"{base_prefix}model_comparison_{timestamp}.csv"
    s3_client.put_object(
        Bucket=bucket,
        Key=csv_key,
        Body=csv_content.encode('utf-8')
    )
    print(f"✓ Saved CSV: s3://{bucket}/{csv_key}")
    
    # Update latest comparison pointers
    latest_json_key = f"{base_prefix}latest_model_comparison.json"
    latest_csv_key = f"{base_prefix}latest_model_comparison.csv"
    
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': json_key},
        Key=latest_json_key
    )
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': csv_key},
        Key=latest_csv_key
    )
    
    print(f"✓ Updated latest JSON: s3://{bucket}/{latest_json_key}")
    print(f"✓ Updated latest CSV: s3://{bucket}/{latest_csv_key}")
    
    print(f"\n" + "=" * 80)
    print("✓ Model Comparison Saved Successfully!")
    print("=" * 80)
    print(f"\n📥 Download Comparison:")
    print(f"   JSON: aws s3 cp s3://{bucket}/{latest_json_key} -")
    print(f"   CSV:  aws s3 cp s3://{bucket}/{latest_csv_key} comparison.csv")
    
    return {
        'json_path': f"s3://{bucket}/{json_key}",
        'csv_path': f"s3://{bucket}/{csv_key}",
        'latest_json': f"s3://{bucket}/{latest_json_key}",
        'latest_csv': f"s3://{bucket}/{latest_csv_key}"
    }


def main():
    """
    Main function to generate model comparison
    Run after all parallel training tasks complete in DAG
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON GENERATION")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Load all model performance data from S3
    all_results = load_model_performance_from_s3(config)
    
    if len(all_results) < 2:
        print(f"\n⚠ Warning: Only {len(all_results)} model(s) found. Need at least 2 for comparison.")
        print("Exiting without generating comparison.")
        return
    
    # Generate comparison
    comparison_data = generate_model_comparison(all_results, config)
    
    # Save to S3
    comparison_paths = save_model_comparison_to_s3(comparison_data, config)
    
    print(f"\n✓ Comparison Generation Complete!")
    print(f"\n📊 Access your comparison:")
    print(f"   Latest JSON: {comparison_paths['latest_json']}")
    print(f"   Latest CSV:  {comparison_paths['latest_csv']}")


if __name__ == "__main__":
    main()
