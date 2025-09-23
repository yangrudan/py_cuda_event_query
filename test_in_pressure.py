import torch
import torch.distributed as dist
import time
import argparse
from async_timer import AsyncTimer  # 导入C++计时器类

def main(index):

    # 2. 第一次正常通信测试（验证分布式环境）
    gradient = torch.tensor([rank, rank, rank], dtype=torch.float32, device=device)
    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        #print(f"Rank {rank} [同步模式] all_reduce结果: {gradient.cpu().numpy()}")
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        #print(f"Rank {rank} [异步模式] all_reduce结果: {gradient.cpu().numpy()}")


    #dist.barrier(device_ids=[device.index])  # 指定设备，消除警告
    # 获取当前CUDA流（NCCL通信默认使用当前流）
    current_stream = torch.cuda.current_stream(torch.cuda.current_device())
    # 将流指针转换为整数传递给C++
    stream_ptr = current_stream.cuda_stream

    timer = AsyncTimer()  # 实例化计时器
    gradient = torch.tensor([rank, rank, rank], dtype=torch.float32, device=device)

    if rank == 1:
        #print(f"\nRank {rank} 开始休眠3秒（模拟节点延迟）...")
        time.sleep(3)  # CPU休眠，不阻塞GPU计时器
   # timer.start(stream_ptr)  # 1. 记录开始事件（GPU时间戳）

    # 3. 启动模拟异常通信（同步/异步，不等待完成）
    if args.sync_mode:
        # 同步模式：all_reduce会阻塞直到完成，启动后立即记录结束事件
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        timer.end(stream_ptr)  # 3. 记录结束事件（GPU时间戳）
    else:
        # 异步模式：启动all_reduce后立即返回work对象，不阻塞
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
        # 异步等待通信完成（仅为了确保梯度计算完成，不影响计时）
        timer.start(stream_ptr)  # 1. 记录开始事件（GPU时间戳）
        work.wait()

        timer.end(stream_ptr)  # 3. 记录结束事件（GPU时间戳）

    # 4. 等待计时器完成（C++线程异步查询CUDA事件）
    #print(f"Rank {rank} 等待计时器结果...")
    while not timer.is_completed():
        # 等待期间可执行其他CPU/GPU任务（体现异步计时优势）
        time.sleep(0.1)  # 降低CPU占用

    # 5. 获取并打印耗时
    elapsed = timer.get_elapsed()
    if rank == 1:
        if elapsed > 100:
            print(f" \n!!!RANK {rank} in index {index} - 总耗时: {elapsed:.4f}ms")
            return
    else:
        if abs(elapsed - 3000) > 200:
            print(f" \n!!!RANK {rank} in index {index} - 总耗时: {elapsed:.4f}ms")
    if rank == 0:
        print(f"=======current index is {index}=======")
    # print(f" \n!!!RANK {rank} - 总耗时: {elapsed:.4f}ms")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="使用同步all_reduce")
    args = parser.parse_args()

    # 1. 初始化分布式环境（指定GPU设备，消除警告）
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    torch.cuda.set_device(device)  # 显式设置当前进程的GPU设备，消除警告
    for i in range(5000):
         main(i)

    # 6. 清理资源
    del timer  # 销毁计时器（调用C++析构函数）
    dist.destroy_process_group()
