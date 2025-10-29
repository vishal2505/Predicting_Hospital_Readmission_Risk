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
  description = "ECS Task Definition family:revision"
  value       = "${aws_ecs_task_definition.pipeline.family}:${aws_ecs_task_definition.pipeline.revision}"
}
