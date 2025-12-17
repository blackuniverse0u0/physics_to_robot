#!/bin/bash

# ------------------------------------------------------------------
# [설정] Python 스크립트 파일명
PYTHON_SCRIPT="ppo_mujoco.py"

# 학습할 환경 리스트 (v5 버전)
ENVS=(
    "Ant-v5"
    "HalfCheetah-v5"
    "Hopper-v5"
    "Humanoid-v5"
    "InvertedDoublePendulum-v5"
    "InvertedPendulum-v5"
    "Reacher-v5"
    "Walker2d-v5"
)

# 로그를 저장할 폴더 생성
mkdir -p logs

echo "Starting Sequential Training..."

for env in "${ENVS[@]}"
do
    echo "--------------------------------------------------"
    echo "Current Environment: $env"
    echo "Status: Training started..."
    
    # Python 스크립트 실행
    # stdout과 stderr를 모두 logs 폴더 내의 파일로 리다이렉션합니다.
    # --num-envs 4: 병렬 환경 개수는 CPU 코어 수에 맞춰 조절하세요.
    python $PYTHON_SCRIPT \
        --env "$env" \
        --mode train \
        --num-envs 32 \
        > "logs/${env}_training.log" 2>&1
        
    echo "Status: Training finished for $env"
done

echo "=================================================="
echo "All environments have been trained successfully."
echo "Check the 'logs' folder for detailed output."