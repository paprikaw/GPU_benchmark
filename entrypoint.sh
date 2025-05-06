#!/bin/sh
set -e

# 脚本入口读取 CLIENT_ID 环境变量
echo "=== Testing MPS connectivity: Client ${CLIENT_ID} ==="

# 编译 timer 程序
nvcc -g -G "/workspace/GPU_timer.cu" -o "/workspace/timer-${CLIENT_ID}"

# 查看 /dev/shm
ls -la /dev/shm

# 运行测试并打印耗时
"/workspace/timer-${CLIENT_ID}"

echo "=== Testing MPS connectivity completed: Client ${CLIENT_ID} ==="