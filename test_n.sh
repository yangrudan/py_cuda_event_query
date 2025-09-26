#!/bin/bash

# 循环执行100次torchrun命令
for ((i=1; i<=10; i++)); do
    echo "开始执行第 $i 次..."
    torchrun --nproc_per_node=8 --master_port=1234 test_in_pressure.py
    # 检查命令执行结果
    if [ $? -ne 0 ]; then
        echo "第 $i 次执行失败！"
        # 可以选择在这里退出循环，或继续执行
        # exit 1
    fi
    echo "第 $i 次执行完成"
    echo "------------------------"
done

echo "所有执行已完成"

