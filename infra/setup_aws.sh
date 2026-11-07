#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STARTER_DIR="${PROJECT_DIR}/starter"

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="finbert-api"
ECS_CLUSTER="finbert-cluster"
ECS_SERVICE="finbert-api-service"
TASK_FAMILY="finbert-api"
CONTAINER_NAME="finbert-api"
SECURITY_GROUP_NAME="finbert-sg"
LOG_GROUP="/ecs/finbert-api"
EXECUTION_ROLE_NAME="ecsTaskExecutionRole"
MODEL_ID="baptle/FinBERT_market_based"

export AWS_PAGER=""

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

allow_http_ingress() {
  local output

  if ! output="$(aws ec2 authorize-security-group-ingress \
    --group-id "$1" \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region "${AWS_REGION}" 2>&1)"; then
    if [[ "${output}" != *"InvalidPermission.Duplicate"* ]]; then
      echo "${output}" >&2
      exit 1
    fi
  fi
}

for command_name in aws docker jq; do
  require_command "${command_name}"
done

echo "Checking AWS identity..."
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:bootstrap"

echo "Ensuring ECR repository exists..."
if ! aws ecr describe-repositories \
  --repository-names "${ECR_REPOSITORY}" \
  --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws ecr create-repository \
    --repository-name "${ECR_REPOSITORY}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE \
    --region "${AWS_REGION}" >/dev/null
fi

echo "Building and pushing bootstrap image..."
aws ecr get-login-password --region "${AWS_REGION}" |
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"
docker build \
  --tag "${IMAGE_URI}" \
  --tag "${ECR_REGISTRY}/${ECR_REPOSITORY}:latest" \
  "${STARTER_DIR}"
docker push "${IMAGE_URI}"
docker push "${ECR_REGISTRY}/${ECR_REPOSITORY}:latest"

echo "Ensuring ECS task execution role exists..."
if ! EXECUTION_ROLE_ARN="$(aws iam get-role \
  --role-name "${EXECUTION_ROLE_NAME}" \
  --query Role.Arn \
  --output text 2>/dev/null)"; then
  EXECUTION_ROLE_ARN="$(aws iam create-role \
    --role-name "${EXECUTION_ROLE_NAME}" \
    --assume-role-policy-document "file://${SCRIPT_DIR}/ecs-task-execution-trust-policy.json" \
    --query Role.Arn \
    --output text)"
fi
aws iam attach-role-policy \
  --role-name "${EXECUTION_ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

echo "Ensuring network resources exist..."
VPC_ID="$(aws ec2 describe-vpcs \
  --filters Name=is-default,Values=true \
  --region "${AWS_REGION}" \
  --query 'Vpcs[0].VpcId' \
  --output text)"

SG_ID="$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SECURITY_GROUP_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --region "${AWS_REGION}" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)"

if [[ "${SG_ID}" == "None" ]]; then
  SG_ID="$(aws ec2 create-security-group \
    --group-name "${SECURITY_GROUP_NAME}" \
    --description "Public HTTP access to the FinBERT API" \
    --vpc-id "${VPC_ID}" \
    --region "${AWS_REGION}" \
    --query GroupId \
    --output text)"
fi
allow_http_ingress "${SG_ID}"

SUBNETS="$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=${VPC_ID}" Name=default-for-az,Values=true \
  --region "${AWS_REGION}" \
  --query 'Subnets[].SubnetId' \
  --output text | tr '\t' ',')"

echo "Ensuring ECS cluster and log group exist..."
CLUSTER_STATUS="$(aws ecs describe-clusters \
  --clusters "${ECS_CLUSTER}" \
  --region "${AWS_REGION}" \
  --query 'clusters[0].status' \
  --output text)"
if [[ "${CLUSTER_STATUS}" != "ACTIVE" ]]; then
  aws ecs create-cluster \
    --cluster-name "${ECS_CLUSTER}" \
    --region "${AWS_REGION}" >/dev/null
fi

if ! aws logs describe-log-groups \
  --log-group-name-prefix "${LOG_GROUP}" \
  --region "${AWS_REGION}" \
  --query "logGroups[?logGroupName=='${LOG_GROUP}'].logGroupName | [0]" \
  --output text | grep -qx "${LOG_GROUP}"; then
  aws logs create-log-group \
    --log-group-name "${LOG_GROUP}" \
    --region "${AWS_REGION}"
fi

echo "Registering ECS task definition..."
TASK_DEFINITION_FILE="$(mktemp /tmp/finbert-task-definition.XXXXXX.json)"
trap 'rm -f "${TASK_DEFINITION_FILE}"' EXIT

jq -n \
  --arg family "${TASK_FAMILY}" \
  --arg role "${EXECUTION_ROLE_ARN}" \
  --arg image "${IMAGE_URI}" \
  --arg container "${CONTAINER_NAME}" \
  --arg region "${AWS_REGION}" \
  --arg log_group "${LOG_GROUP}" \
  --arg model_id "${MODEL_ID}" \
  '{
    family: $family,
    executionRoleArn: $role,
    networkMode: "awsvpc",
    requiresCompatibilities: ["FARGATE"],
    cpu: "2048",
    memory: "4096",
    runtimePlatform: {
      cpuArchitecture: "X86_64",
      operatingSystemFamily: "LINUX"
    },
    containerDefinitions: [
      {
        name: $container,
        image: $image,
        essential: true,
        command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"],
        portMappings: [
          {
            containerPort: 80,
            hostPort: 80,
            protocol: "tcp"
          }
        ],
        environment: [
          {name: "MODEL_SOURCE", value: "huggingface"},
          {name: "HF_MODEL_ID", value: $model_id}
        ],
        logConfiguration: {
          logDriver: "awslogs",
          options: {
            "awslogs-group": $log_group,
            "awslogs-region": $region,
            "awslogs-stream-prefix": "ecs"
          }
        }
      }
    ]
  }' >"${TASK_DEFINITION_FILE}"

TASK_DEFINITION_ARN="$(aws ecs register-task-definition \
  --cli-input-json "file://${TASK_DEFINITION_FILE}" \
  --region "${AWS_REGION}" \
  --query taskDefinition.taskDefinitionArn \
  --output text)"

echo "Ensuring ECS service exists..."
SERVICE_STATUS="$(aws ecs describe-services \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --region "${AWS_REGION}" \
  --query 'services[0].status' \
  --output text)"

if [[ "${SERVICE_STATUS}" == "ACTIVE" ]]; then
  aws ecs update-service \
    --cluster "${ECS_CLUSTER}" \
    --service "${ECS_SERVICE}" \
    --task-definition "${TASK_DEFINITION_ARN}" \
    --desired-count 1 \
    --region "${AWS_REGION}" >/dev/null
else
  aws ecs create-service \
    --cluster "${ECS_CLUSTER}" \
    --service-name "${ECS_SERVICE}" \
    --task-definition "${TASK_DEFINITION_ARN}" \
    --desired-count 1 \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
    --region "${AWS_REGION}" >/dev/null
fi

echo "Waiting for ECS service stability..."
aws ecs wait services-stable \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --region "${AWS_REGION}"

echo
echo "AWS infrastructure is ready."
echo "ECR registry : ${ECR_REGISTRY}"
echo "ECS cluster  : ${ECS_CLUSTER}"
echo "ECS service  : ${ECS_SERVICE}"
echo "Security group: ${SG_ID}"
