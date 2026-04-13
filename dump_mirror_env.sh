#!/bin/bash
set -x

OUTPUT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/mirror_env_dump

mkdir -p $OUTPUT

echo '=== conda env list ==='
conda env list

echo '=== 当前激活环境 pip freeze ==='
pip freeze > $OUTPUT/pip_freeze.txt
echo "pip freeze 已保存到 $OUTPUT/pip_freeze.txt"

echo '=== conda list ==='
conda list > $OUTPUT/conda_list.txt
echo "conda list 已保存到 $OUTPUT/conda_list.txt"

echo '=== python version ==='
python --version
which python

echo '=== 导出 conda 环境 yaml ==='
conda env export > $OUTPUT/env_export.yaml
echo "env yaml 已保存到 $OUTPUT/env_export.yaml"

echo '=== 打包整个 conda 环境到 /mnt (用于复用) ==='
# conda pack -o $OUTPUT/mirror_env.tar.gz  # 取消注释可打包整个环境（较慢）

echo '=== 完成 ==='
