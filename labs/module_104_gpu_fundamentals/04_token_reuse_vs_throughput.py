import statistics
import time

import torch


HIDDEN_SIZE = 2048
TOKEN_COUNTS = [1, 8, 32, 128, 512]

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
print("Hidden size:", HIDDEN_SIZE)


# --------------------------------------------------
# Same weight matrix for every experiment
# --------------------------------------------------

weight = torch.randn(
    HIDDEN_SIZE,
    HIDDEN_SIZE,
    dtype=torch.float32,
    device=device,
)


for tokens in TOKEN_COUNTS:

    # Shape:
    #
    # [tokens, hidden]
    #
    # Similar to having multiple token representations
    # going through the same linear layer.

    x = torch.randn(
        tokens,
        HIDDEN_SIZE,
        dtype=torch.float32,
        device=device,
    )


    elapsed = benchmark(
        lambda: x @ weight
    )


    # --------------------------------------------------
    # FLOPs
    #
    # Matrix multiplication:
    #
    # [M, K] @ [K, N]
    #
    # FLOPs ≈ 2 * M * K * N
    #
    # Here:
    # M = tokens
    # K = hidden_size
    # N = hidden_size
    # --------------------------------------------------

    flops = (
        2
        * tokens
        * HIDDEN_SIZE
        * HIDDEN_SIZE
    )


    gflops = (
        flops / elapsed
    ) / 1_000_000_000


    # --------------------------------------------------
    # Idealized minimum memory traffic
    #
    # Read X
    # Read weight
    # Write output
    #
    # float32 = 4 bytes
    # --------------------------------------------------

    input_bytes = (
        tokens
        * HIDDEN_SIZE
        * 4
    )

    weight_bytes = (
        HIDDEN_SIZE
        * HIDDEN_SIZE
        * 4
    )

    output_bytes = (
        tokens
        * HIDDEN_SIZE
        * 4
    )

    total_bytes = (
        input_bytes
        + weight_bytes
        + output_bytes
    )


    arithmetic_intensity = (
        flops / total_bytes
    )


    print("\n" + "=" * 60)

    print(
        f"TOKENS: {tokens}"
    )

    print("-" * 60)

    print(
        "Input shape:",
        tuple(x.shape)
    )

    print(
        f"Median time: "
        f"{elapsed * 1000:.3f} ms"
    )

    print(
        f"Compute throughput: "
        f"{gflops:.2f} GFLOP/s"
    )

    print(
        f"Idealized arithmetic intensity: "
        f"{arithmetic_intensity:.2f} FLOP/byte"
    )