# Ray OOM 问题修复说明

## 问题描述

训练过程中出现以下错误：

```
(raylet) Workers killed due to memory usage (10.164.166.24) over last time period.
ray.exceptions.ActorUnavailableError: The actor is temporarily unavailable
```

## 问题原因

1. **Ray 内存监控过于严格**：Ray 默认会监控内存使用，当内存超过阈值时会杀死 worker
2. **没有限制 Ray 资源**：没有设置 `num_cpus` 和 `object_store_memory`，导致 Ray 占用过多资源
3. **Docker 内存限制**：容器的内存限制可能不够

## 解决方案

### 1. 禁用 Ray 内存监控

```bash
export RAY_memory_monitor_refresh_ms=0  # 禁用内存监控
export RAY_memory_usage_threshold=1.0  # 设置阈值为 100%
export RAY_DISABLE_MEMORY_MONITOR=1  # 完全禁用内存监控
export RAY_DISABLE_DASHBOARD=1  # 禁用 Ray Dashboard（避免 OpenTelemetry 兼容性问题）
```

### 1.5. 增加系统限制和限制线程数

```bash
ulimit -n 65535          # 增加文件描述符限制
ulimit -u unlimited       # 增加进程数限制
ulimit -s unlimited       # 增加栈大小限制
export OPENBLAS_NUM_THREADS=1  # 限制 OpenBLAS 线程数
export OMP_NUM_THREADS=1        # 限制 OpenMP 线程数
```

### 2. 限制 Ray 资源

在训练命令中添加：

```bash
ray_init.num_cpus=32
```

**注意**：`ray_init` 配置只支持 `num_cpus` 参数，不支持 `num_gpus` 或 `object_store_memory` 参数。添加这些参数会导致配置错误。

### 3. 增加 Docker 内存

在 Hope 配置中增加内存分配：

```ini
worker.memory = 400000  # 400GB
```

## 修改的文件

1. **run_alfworld_hope.sh**
   - 添加了 Ray 环境变量
   - 添加了 `ray_init` 配置

2. **submit_alfworld_conda.sh**
   - 添加了 Ray 环境变量
   - 添加了 `ray_init` 配置

3. **submit_alfworld_mirror.sh**
   - 添加了 Ray 环境变量

## 使用方法

```bash
# 使用 Hotel 镜像（推荐）
bash submit_alfworld_mirror.sh 2 1

# 使用本地 conda 环境
bash submit_alfworld_conda.sh 2 1
```

## 其他注意事项

1. **pynvml 警告**：`The pynvml package is deprecated`
   - 这是警告，不影响运行
   - 可以通过安装 `nvidia-ml-py` 来解决

2. **内存监控**：如果仍然出现 OOM，可以：
   - 减少 batch size
   - 减少 worker 数量
   - 增加 Docker 内存分配

3. **pkg_resources 缺失**：
   - 错误：`ModuleNotFoundError: No module named 'pkg_resources'`
   - 原因：setuptools 70+ 版本中 `pkg_resources` 已被弃用
   - 解决：复制原始环境中的 `pkg_resources` 目录到用户环境

4. **依赖包缺失**：
   - 错误：`ModuleNotFoundError: No module named 'xxx'`
   - 原因：用户 conda 环境缺少 verl-agent 依赖包
   - 解决：安装 requirements.txt 中的所有包或从原始环境同步 site-packages
   - 主要依赖：torch, vllm, ray, transformers, accelerate, numpy, pandas, alfworld 等

5. **OpenTelemetry 兼容性问题**：
   - 错误：`TypeError: Meter.create_histogram() got an unexpected keyword argument 'explicit_bucket_boundaries_advisory'`
   - 原因：Ray 2.41.0 与 OpenTelemetry 1.26.0 不兼容
   - 解决：修改 Ray 代码 `open_telemetry_metric_recorder.py`，删除 `explicit_bucket_boundaries_advisory` 参数

6. **线程创建失败**：
   - 错误：`RuntimeError: can't start new thread` 或 `OpenBLAS blas_thread_init: pthread_create failed`
   - 原因：系统线程限制（ulimit）太低，OpenBLAS 尝试创建过多线程
   - 解决：增加 ulimit 限制并限制 OpenBLAS 线程数
   - 已在脚本中添加：`ulimit -u unlimited`, `export OPENBLAS_NUM_THREADS=1`
