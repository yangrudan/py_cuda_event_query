import argparse
import torch
import torch.distributed as dist
import os
import time
from functools import wraps
from torch.distributed import distributed_c10d as c10d
from torch.distributed.distributed_c10d import ProcessGroupNCCL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="启用同步通信模式")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Mock gradient
    gradient = torch.tensor([rank] * 3, dtype=torch.float32, device=device)
    print(f"Rank {rank} 初始梯度: {gradient.cpu().numpy()}")

    # 先进行一次正常的通信
    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)

    if args.sync_mode:
        print(f"Rank {rank} [同步0000]all_reduce后梯度: {gradient.cpu().numpy()}\n")
    else:
        work.wait()
        print(f"Rank {rank} [异步0000]all_reduce后梯度: {gradient.cpu().numpy()}\n")
    
    # 进行异常捕获
    # 创建事件
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()   
    
    # 制造异常
    if rank == 0:
        time.sleep(15)
    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)

    if args.sync_mode:
        print(f"Rank {rank} [同步]")
    else:
        work.wait()
        print(f"Rank {rank} [异步]")

    end.record()
    torch.cuda.synchronize()	
    elapsed = start.elapsed_time(end) 
    print(f"异常情况下 RANK{rank} elapsed time :{elapsed}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    # cmd：torchrun --nproc_per_node=4 this_script.py
    main()

