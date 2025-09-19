import torch
import torch.distributed as dist
import time
import argparse
from async_timer import AsyncTimer  # 直接导入C++定义的类

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="使用同步all_reduce")
    args = parser.parse_args()

    # 1. 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    # 2. 第一次正常通信（Python层调用all_reduce）
    gradient = torch.tensor([rank, rank, rank], dtype=torch.float32, device=device)
    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        print(f"Rank {rank} [同步0000] all_reduce后梯度: {gradient.cpu().numpy()}\n")
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        print(f"Rank {rank} [异步0000] all_reduce后梯度: {gradient.cpu().numpy()}\n")

    # 3. 异常场景测试（使用C++ AsyncTimer类）
    gradient = torch.tensor([rank, rank, rank], dtype=torch.float32, device=device)
    timer = AsyncTimer()  # 实例化C++计时器（无需手动管理指针）

    # 全局同步：确保所有进程同时开始计时
    dist.barrier()
    timer.start()  # 开始计时（非阻塞）

    # 制造异常：Rank 0休眠15秒（CPU操作，不阻塞GPU）
    if rank == 0:
        time.sleep(15)

    # 执行分布式all_reduce（Python层）
    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        timer.end()  # 结束计时（非阻塞）
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
        # 异步等待通信完成（期间可插入其他GPU计算）
        while not work.is_completed():
            pass
        timer.end()  # 结束计时（非阻塞）

    # 等待计时完成并获取结果
    while not timer.is_completed():
        pass  # 等待期间可执行其他任务
    elapsed = timer.get_elapsed()
    print(f"异常情况下 RANK{rank} elapsed time: {float(elapsed):.4f}ms\n")

    # 4. 清理资源（Python垃圾回收会自动调用C++析构函数）
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
