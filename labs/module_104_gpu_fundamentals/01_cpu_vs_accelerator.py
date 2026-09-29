import statistics
import time

import torch


SIZES = [256, 1024, 2048]
WARMUP_RUNS = 3
MEASURED_RUNS = 10


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:

    # Warm up the accelerator / runtime.
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            _ = a @ b

    synchronize(device)

    timings = []

    with torch.inference_mode():

        for _ in range(MEASURED_RUNS):

            synchronize(device)

            start = time.perf_counter()

            _ = a @ b

            synchronize(device)

            elapsed = time.perf_counter() - start

            timings.append(elapsed)

    median_time = statistics.median(timings)

    n = a.shape[0]

    # Approximate FLOPs for NxN matrix multiplication:
    # 2 * N^3
    flops = 2 * (n ** 3)

    gflops_per_second = (
        flops / median_time
    ) / 1_000_000_000

    return median_time, gflops_per_second


if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available on this Mac."
    )


cpu = torch.device("cpu")
mps = torch.device("mps")


print("CPU device:", cpu)
print("Accelerator device:", mps)


for size in SIZES:

    print("\n" + "=" * 60)
    print(f"MATRIX SIZE: {size} x {size}")
    print("=" * 60)

    # --------------------------------------------------
    # Create tensors on CPU
    # --------------------------------------------------

    a_cpu = torch.randn(
        size,
        size,
        dtype=torch.float32,
    )

    b_cpu = torch.randn(
        size,
        size,
        dtype=torch.float32,
    )


    # --------------------------------------------------
    # CPU benchmark
    # --------------------------------------------------

    cpu_time, cpu_gflops = benchmark_matmul(
        a_cpu,
        b_cpu,
        cpu,
    )


    # --------------------------------------------------
    # Move tensors to accelerator BEFORE timing
    # --------------------------------------------------

    a_mps = a_cpu.to(mps)
    b_mps = b_cpu.to(mps)

    synchronize(mps)


    # --------------------------------------------------
    # MPS benchmark
    # --------------------------------------------------

    mps_time, mps_gflops = benchmark_matmul(
        a_mps,
        b_mps,
        mps,
    )


    print("\nCPU")
    print("-" * 30)
    print(f"Median time: {cpu_time * 1000:.3f} ms")
    print(f"Throughput: {cpu_gflops:.2f} GFLOP/s")


    print("\nMPS")
    print("-" * 30)
    print(f"Median time: {mps_time * 1000:.3f} ms")
    print(f"Throughput: {mps_gflops:.2f} GFLOP/s")


    print("\nSpeed ratio")
    print("-" * 30)

    speedup = cpu_time / mps_time

    print(
        f"MPS / CPU speed ratio: {speedup:.2f}x"
    )