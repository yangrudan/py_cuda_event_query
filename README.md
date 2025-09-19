> **方式**: 手动注入event或者利用torch已有event，新建线程 使用 cudaEventQuery  查询end_event的状态，当返回值是 cudaSuccess 的时候表示event结束(达到cuda.synchronize效果)，再计算持续时间
>

# 安装使用
```bash
python setup.py install
torchrun --nproc_per_node=4 --master_port=1234 test.py --sync_mode
```

## 运行结果

### 异步模式
<img width="3727" height="1605" alt="截图 2025-09-19 14-01-53" src="https://github.com/user-attachments/assets/5f1d5b88-fb61-4fd3-8970-f269aefbee28" />

### 同步模式
<img width="3727" height="1605" alt="截图 2025-09-19 13-58-02" src="https://github.com/user-attachments/assets/a6bb80ee-402b-4fd3-bda0-aed78288a1a4" />

