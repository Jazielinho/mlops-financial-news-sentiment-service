#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/starter/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

for command_name in aws gh git; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Required command not found: ${command_name}" >&2
    exit 1
  }
done

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${AWS_ACCESS_KEY_ID:?Missing AWS_ACCESS_KEY_ID in starter/.env}"
: "${AWS_SECRET_ACCESS_KEY:?Missing AWS_SECRET_ACCESS_KEY in starter/.env}"
: "${AWS_SESSION_TOKEN:?Missing AWS_SESSION_TOKEN in starter/.env}"
AWS_REGION="${AWS_REGION:-us-east-1}"

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ORIGIN_URL="$(git -C "${PROJECT_DIR}" remote get-url origin)"
REPOSITORY="$(gh repo view "${ORIGIN_URL}" --json nameWithOwner --jq .nameWithOwner)"

set_secret() {
  local name="$1"
  local value="$2"

  printf '%s' "${value}" | gh secret set "${name}" --repo "${REPOSITORY}"
}

set_secret AWS_ACCESS_KEY_ID "${AWS_ACCESS_KEY_ID}"
set_secret AWS_SECRET_ACCESS_KEY "${AWS_SECRET_ACCESS_KEY}"
set_secret AWS_SESSION_TOKEN "${AWS_SESSION_TOKEN}"
set_secret AWS_REGION "${AWS_REGION}"
set_secret ECR_REGISTRY "${ECR_REGISTRY}"

echo "GitHub Actions secrets configured for ${REPOSITORY}."
