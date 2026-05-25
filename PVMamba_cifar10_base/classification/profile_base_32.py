import argparse
import time
import torch

from config import get_config
from models import build_model


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def try_count_flops(model, x):
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x).total() / 1e9
        return flops
    except Exception as e:
        print("[Warning] FLOPs统计失败：")
        print(e)
        return None


def measure_memory_and_time(model, x, repeat=20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(repeat):
            _ = model(x)
        torch.cuda.synchronize()
        end = time.time()

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    avg_time_ms = (end - start) / repeat * 1000
    return peak_mem_gb, avg_time_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--data-path", default="/home/jlx_75/datasets/cifar10", type=str)
    parser.add_argument("--output", default="./output_profile_base", type=str)
    parser.add_argument("--opts", nargs="*", default=None)
    args = parser.parse_args()

    config = get_config(args)

    model = build_model(config)
    model.cuda()
    model.eval()

    img_size = 32
    x = torch.randn(args.batch_size, 3, img_size, img_size).cuda()

    params_m = count_params(model)
    flops_g = try_count_flops(model, x)
    mem_gb, avg_time_ms = measure_memory_and_time(model, x)

    print("=" * 60)
    print("Profile Result: PVMamba-Tiny Base")
    print(f"Config: {args.cfg}")
    print("Input size: 32 x 32")
    print(f"Batch size: {args.batch_size}")
    print(f"Params(M): {params_m:.4f}")
    if flops_g is not None:
        print(f"FLOPs(G): {flops_g:.4f}")
    else:
        print("FLOPs(G): failed")
    print(f"Peak inference memory(GB): {mem_gb:.4f}")
    print(f"Average inference time(ms/batch): {avg_time_ms:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
