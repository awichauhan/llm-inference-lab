import statistics
import time

import torch


WARMUP_RUNS = 3
MEASURED_RUNS = 10


def synchronize() -> None:
    torch.mps.synchronize()


def benchmark(operation) -> float:
    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):
            _ = operation()

    synchronize()

    timings = []

    with torch.inference_mode():

        for _ in range(MEASURED_RUNS):

            synchronize()

            start = time.perf_counter()

            _ = operation()

            synchronize()

            elapsed = time.perf_counter() - start

            timings.append(elapsed)

    return statistics.median(timings)


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available.")


device = torch.device("mps")

print("Device:", device)


# ==================================================
# 1. MEMORY-HEAVY WORKLOAD
# Element-wise addition
# ==================================================

ELEMENTS = 16_000_000

a = torch.randn(
    ELEMENTS,
    dtype=torch.float32,
    device=device,
)

b = torch.randn(
    ELEMENTS,
    dtype=torch.float32,
    device=device,
)


add_time = benchmark(
    lambda: a + b
)


# float32 = 4 bytes
#
# Read A = 4 bytes
# Read B = 4 bytes
# Write C = 4 bytes
#
# Approximate logical traffic:
# 12 bytes / element

add_bytes = ELEMENTS * 12

add_bandwidth = (
    add_bytes / add_time
) / 1_000_000_000


add_flops = ELEMENTS

add_gflops = (
    add_flops / add_time
) / 1_000_000_000


print("\nELEMENT-WISE ADD")
print("-" * 50)

print(
    f"Median time: {add_time * 1000:.3f} ms"
)

print(
    f"Approx logical bandwidth: "
    f"{add_bandwidth:.2f} GB/s"
)

print(
    f"Compute rate: {add_gflops:.2f} GFLOP/s"
)

print(
    "Arithmetic intensity:",
    f"{add_flops / add_bytes:.4f} FLOP/byte"
)


# ==================================================
# 2. COMPUTE-HEAVY WORKLOAD
# Matrix multiplication
# ==================================================

N = 2048

x = torch.randn(
    N,
    N,
    dtype=torch.float32,
    device=device,
)

w = torch.randn(
    N,
    N,
    dtype=torch.float32,
    device=device,
)


matmul_time = benchmark(
    lambda: x @ w
)


# Approximate matrix multiplication work:
#
# 2 * N^3 FLOPs

matmul_flops = 2 * (N ** 3)

matmul_gflops = (
    matmul_flops / matmul_time
) / 1_000_000_000


# Idealized minimum logical bytes:
#
# read X
# read W
# write result
#
# each matrix = N^2 float32 values

matmul_bytes = (
    3
    * (N ** 2)
    * 4
)

ideal_intensity = (
    matmul_flops
    / matmul_bytes
)


print("\nMATRIX MULTIPLICATION")
print("-" * 50)

print(
    f"Median time: {matmul_time * 1000:.3f} ms"
)

print(
    f"Compute rate: {matmul_gflops:.2f} GFLOP/s"
)

print(
    "Idealized arithmetic intensity:",
    f"{ideal_intensity:.2f} FLOP/byte"
)