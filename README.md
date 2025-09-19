> **方式**: 手动注入event或者利用torch已有event，新建线程 使用 cudaEventQuery  查询end_event的状态，当返回值是 cudaSuccess 的时候表示event结束(达到cuda.synchronize效果)，再计算持续时间
>

# 安装使用
```bash
python setup.py install
torchrun --nproc_per_node=4 --master_port=1234 test.py --sync_mode
```
