#!/usr/bin/env python3
"""Hold MIG0 near a target memory fraction until this process is stopped."""

import argparse
import signal
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-fraction", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if not 0.5 <= args.target_fraction < 0.99:
        parser.error("target-fraction must be between 0.5 and 0.99")
    if args.chunk_mib < 16:
        parser.error("chunk-mib must be at least 16")

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; set CUDA_VISIBLE_DEVICES to the MIG0 UUID")
    torch.cuda.set_device(device)

    held = []
    chunk_bytes = args.chunk_mib * 1024 * 1024
    print(f"Holding {torch.cuda.get_device_name(device)} toward {args.target_fraction:.0%}; Ctrl-C or kill this PID to release.", flush=True)
    try:
        while not stopping:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            target_used = int(total_bytes * args.target_fraction)
            current_used = total_bytes - free_bytes
            need = target_used - current_used
            if need > 0:
                # Leave a small allocator margin and handle fragmentation/OOM by
                # reducing the final allocation until the request fits.
                amount = min(chunk_bytes, need)
                while amount >= 1024 * 1024:
                    try:
                        held.append(torch.empty(amount // 4, dtype=torch.float32, device=device))
                        break
                    except torch.cuda.OutOfMemoryError:
                        amount //= 2
                        torch.cuda.empty_cache()
                if amount < 1024 * 1024:
                    print("Could not allocate another safe chunk; monitoring current usage.", flush=True)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            used = total_bytes - free_bytes
            print(f"MIG memory {used / 2**20:.0f}/{total_bytes / 2**20:.0f} MiB ({used / total_bytes:.1%}); holding {len(held)} chunks", flush=True)
            time.sleep(args.poll_seconds)
    finally:
        held.clear()
        torch.cuda.empty_cache()
        print("Released held MIG memory.", flush=True)


if __name__ == "__main__":
    main()
