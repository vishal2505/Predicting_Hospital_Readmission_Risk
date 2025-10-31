output "airflow_public_ip" {
  description = "Public IP of the Airflow EC2 instance"
  value       = aws_instance.airflow.public_ip
}

output "airflow_url" {
  description = "Airflow Web UI URL"
  value       = "http://${aws_instance.airflow.public_ip}:8080"
}

output "ecr_repository_url" {
  description = "ECR repository URL for the pipeline image"
  value       = aws_ecr_repository.repo.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.pipeline.name
}

output "ecs_task_definition" {
  description = "ECS Task Definition family:revision for data processing"
  value       = "${aws_ecs_task_definition.pipeline.family}:${aws_ecs_task_definition.pipeline.revision}"
}

output "ecs_model_training_task_definition" {
  description = "ECS Task Definition family:revision for model training (higher resources)"
  value       = "${aws_ecs_task_definition.model_training.family}:${aws_ecs_task_definition.model_training.revision}"
}

output "ecs_subnets" {
  description = "Subnets for ECS tasks"
  value       = join(",", data.aws_subnets.default.ids)
}

output "ecs_security_groups" {
  description = "Security groups for ECS tasks"
  value       = aws_security_group.ecs_tasks_sg.id
}

output "ecs_task_role_arn" {
  description = "ECS task role ARN"
  value       = aws_iam_role.ecs_task_role.arn
}

output "datamart_bucket" {
  description = "S3 datamart bucket name"
  value       = "diab-readmit-123456-datamart"
}

output "model_registry_bucket" {
  description = "S3 model registry bucket name"
  value       = "diab-readmit-123456-model-registry"
}
