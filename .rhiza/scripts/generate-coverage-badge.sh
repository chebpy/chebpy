#!/bin/sh
# Generate a coverage badge endpoint JSON for shields.io
# This script reads _tests/coverage.json and creates a shields.io endpoint JSON file

set -e

BLUE="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

COVERAGE_JSON="_tests/coverage.json"
OUTPUT_DIR="_book/tests"
BADGE_JSON="${OUTPUT_DIR}/coverage-badge.json"

if [ ! -f "${COVERAGE_JSON}" ]; then
  printf "%b[WARN] Coverage JSON file not found at ${COVERAGE_JSON}, skipping badge generation%b\n" "$YELLOW" "$RESET"
  exit 0
fi

printf "%b[INFO] Generating coverage badge from ${COVERAGE_JSON}...%b\n" "$BLUE" "$RESET"

# Extract coverage percentage and round it
if ! COVERAGE=$(COVERAGE_JSON="${COVERAGE_JSON}" python3 << 'PYTHON_SCRIPT'
import json
import sys
import os

coverage_json = os.environ['COVERAGE_JSON']

try:
    with open(coverage_json, 'r') as f:
        data = json.load(f)
    percent = data['totals']['percent_covered']
    print(f'{percent:.0f}')
except Exception as e:
    print(f'Error extracting coverage: {e}', file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT
); then
  printf "%b[ERROR] Failed to extract coverage percentage%b\n" "$YELLOW" "$RESET"
  exit 1
fi

if [ -z "${COVERAGE}" ]; then
  printf "%b[ERROR] Coverage percentage is empty%b\n" "$YELLOW" "$RESET"
  exit 1
fi

printf "%b[INFO] Coverage: ${COVERAGE}%%%b\n" "$BLUE" "$RESET"

# Determine badge color based on coverage percentage
if [ "${COVERAGE}" -ge 90 ]; then
  COLOR="brightgreen"
elif [ "${COVERAGE}" -ge 80 ]; then
  COLOR="green"
elif [ "${COVERAGE}" -ge 70 ]; then
  COLOR="yellowgreen"
elif [ "${COVERAGE}" -ge 60 ]; then
  COLOR="yellow"
elif [ "${COVERAGE}" -ge 50 ]; then
  COLOR="orange"
else
  COLOR="red"
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Generate shields.io endpoint JSON
cat > "${BADGE_JSON}" << EOF
{
  "schemaVersion": 1,
  "label": "coverage",
  "message": "${COVERAGE}%",
  "color": "${COLOR}"
}
EOF

printf "%b[INFO] Coverage badge JSON generated at ${BADGE_JSON}%b\n" "$BLUE" "$RESET"
