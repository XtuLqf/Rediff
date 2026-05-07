#!/usr/bin/env bash
set -euo pipefail

DATAROOT="./Dataset"
CLASS_EMBEDDING="att"
CHECK_SPLIT="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/check_dataset_mats.sh [--dataroot PATH] [--class-embedding att] [--check-split]

Options:
  --dataroot PATH          Dataset root directory (default: ./Dataset)
  --class-embedding TYPE   att only (default: att)
  --check-split            Also require split_10percent.mat and split_30percent.mat
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataroot)
      DATAROOT="$2"
      shift 2
      ;;
    --class-embedding)
      CLASS_EMBEDDING="$2"
      shift 2
      ;;
    --check-split)
      CHECK_SPLIT="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$CLASS_EMBEDDING" != "att" ]]; then
  echo "Error: this project is fixed to att, use --class-embedding att." >&2
  exit 2
fi

datasets=("AWA2" "CUB" "SUN")
required=("res101.mat" "ce_ce.mat" "con_paco.mat" "${CLASS_EMBEDDING}_splits.mat")
if [[ "$CHECK_SPLIT" == "true" ]]; then
  required+=("split_10percent.mat" "split_30percent.mat")
fi

missing_any=0

echo "Checking dataroot: $DATAROOT"
echo "class_embedding: $CLASS_EMBEDDING"
echo "check split mats: $CHECK_SPLIT"
echo

for ds in "${datasets[@]}"; do
  ds_dir="$DATAROOT/$ds"
  echo "[$ds]"

  if [[ ! -d "$ds_dir" ]]; then
    echo "  Missing directory: $ds_dir"
    missing_any=1
    echo
    continue
  fi

  ds_missing=0
  for f in "${required[@]}"; do
    if [[ -f "$ds_dir/$f" ]]; then
      echo "  OK   $f"
    else
      echo "  MISS $f"
      ds_missing=1
      missing_any=1
    fi
  done

  if [[ "$ds_missing" -eq 0 ]]; then
    echo "  Status: COMPLETE"
  else
    echo "  Status: INCOMPLETE"
  fi
  echo
done

if [[ "$missing_any" -eq 0 ]]; then
  echo "All required .mat files are present."
  exit 0
else
  echo "Some required .mat files are missing."
  exit 1
fi
