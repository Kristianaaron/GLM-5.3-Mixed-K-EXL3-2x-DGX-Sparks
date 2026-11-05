#!/usr/bin/env python3
"""Free page cache on a GB10 (DGX Spark) without root, so CUDA allocations succeed.

GB10's GPU allocates from the same DRAM as the page cache, and the CUDA driver does not reliably wait for the kernel to
reclaim clean cache: with ~1-5 GB MemFree and ~100 GB "available" as cache, cudaMalloc fails with out of memory (NVIDIA
troubleshooting guides). Two user-level steps:

  1. posix_fadvise(DONTNEED) on every file under the given paths (drops their clean pages; files must be readable);
  2. if MemFree is still below --target-free-gib, allocate and touch anonymous memory in 1 GiB steps so the kernel
     reclaims clean cache, then release
     it. MemAvailable is re-read before every step and never taken below --keep-avail-gib; it stops at --max-gib.

  python3 reclaim.py --target-free-gib 100 [--fadvise DIR_OR_FILE ...]
Exit 0 when MemFree >= target afterwards, 1 otherwise (callers fail closed). Run it only while no GPU job is starting.
"""
from __future__ import annotations

import argparse
import mmap
import os
import sys
from pathlib import Path

def meminfo() -> dict[str, float]:
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) / 2**20  # GiB
    return out


def fadvise_tree(paths: list[str]) -> int:
    n = 0
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else (x for x in p.rglob("*") if x.is_file() and not x.is_symlink())
        for f in files:
            try:
                fd = os.open(str(f), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fdatasync(fd)  # DONTNEED skips dirty pages
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


def pressure(target_free: float, keep_avail: float, max_gib: int, step_gib: int = 1) -> int:
    held = []
    try:
        while len(held) * step_gib < max_gib:
            m = meminfo()
            if m["MemFree"] >= target_free or m["MemAvailable"] - step_gib < keep_avail:
                break
            b = mmap.mmap(-1, step_gib << 30, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            for off in range(0, step_gib << 30, 4096):  # touch every page so it is really backed
                b[off] = 1
            held.append(b)
        return len(held) * step_gib
    finally:
        for b in held:
            b.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-free-gib", type=float, required=True)
    ap.add_argument("--keep-avail-gib", type=float, default=8.0)
    ap.add_argument("--max-gib", type=int, default=110)
    ap.add_argument("--fadvise", nargs="*", default=[])
    a = ap.parse_args(argv)
    m0 = meminfo()
    n = fadvise_tree(list(a.fadvise) + [])
    m1 = meminfo()
    used = pressure(a.target_free_gib, a.keep_avail_gib, a.max_gib) if m1["MemFree"] < a.target_free_gib else 0
    m2 = meminfo()
    f = lambda m: f"MemFree {m['MemFree']:.1f} Buffers {m['Buffers']:.1f} Cached {m['Cached']:.1f} Avail {m['MemAvailable']:.1f}"  # noqa: E731
    print(f"reclaim: before {f(m0)}; fadvise {n} files -> {f(m1)}; pressure {used} GiB -> {f(m2)} GiB", flush=True)
    return 0 if m2["MemFree"] >= a.target_free_gib else 1


if __name__ == "__main__":
    sys.exit(main())
