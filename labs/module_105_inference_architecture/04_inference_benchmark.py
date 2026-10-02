import argparse
import gc
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


# ============================================================
# DEVICE
# ============================================================

def get_device(requested_device: str) -> torch.device:

    if requested_device != "auto":
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device: torch.device) -> None:

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    tokenizer,
    prompt_tokens: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:

    text = (
        "Artificial intelligence systems use transformer "
        "architectures to process and generate language. "
    ) * 200

    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    if input_ids.shape[1] < prompt_tokens:
        raise ValueError("Base text did not produce enough tokens.")

    input_ids = input_ids[:, :prompt_tokens]

    # [1, T] → [B, T]
    input_ids = input_ids.repeat(
        batch_size,
        1,
    )

    return input_ids.to(device)


# ============================================================
# MEMORY
# ============================================================

def model_parameter_bytes(model) -> int:

    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )


def current_torch_memory(device: torch.device):

    if device.type == "cuda":
        return torch.cuda.memory_allocated()

    if device.type == "mps":
        return torch.mps.current_allocated_memory()

    return None


def current_driver_memory(device: torch.device):

    if device.type == "cuda":
        return torch.cuda.memory_reserved()

    if device.type == "mps":
        return torch.mps.driver_allocated_memory()

    return None


# ============================================================
# TIMED GENERATION
# ============================================================

def benchmark_once(
    model,
    input_ids,
    output_tokens,
    device,
):

    synchronize(device)

    request_start = time.perf_counter()

    with torch.inference_mode():

        # ----------------------------------------------------
        # PREFILL
        # ----------------------------------------------------

        prefill_start = time.perf_counter()

        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )

        synchronize(device)

        prefill_end = time.perf_counter()

        past_key_values = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        synchronize(device)

        first_token_time = time.perf_counter()


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        decode_times = []

        for _ in range(output_tokens - 1):

            decode_start = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            past_key_values = outputs.past_key_values

            synchronize(device)

            decode_end = time.perf_counter()

            decode_times.append(
                decode_end - decode_start
            )

    request_end = time.perf_counter()


    prefill = (
        prefill_end - prefill_start
    )

    ttft = (
        first_token_time - request_start
    )

    decode_total = sum(decode_times)

    e2e = (
        request_end - request_start
    )

    mean_itl = (
        statistics.mean(decode_times)
        if decode_times
        else 0.0
    )

    median_itl = (
        statistics.median(decode_times)
        if decode_times
        else 0.0
    )

    return {
        "prefill_s": prefill,
        "ttft_s": ttft,
        "decode_total_s": decode_total,
        "mean_itl_s": mean_itl,
        "median_itl_s": median_itl,
        "e2e_s": e2e,
    }


# ============================================================
# MEMORY PROBE
#
# Separate from timing so memory inspection does not distort
# benchmark latency measurements.
# ============================================================

def measure_memory(
    model,
    input_ids,
    output_tokens,
    device,
):

    observed_torch = []

    baseline_torch = current_torch_memory(device)
    baseline_driver = current_driver_memory(device)

    if baseline_torch is not None:
        observed_torch.append(baseline_torch)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )

        synchronize(device)

        memory = current_torch_memory(device)

        if memory is not None:
            observed_torch.append(memory)

        past_key_values = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        for _ in range(output_tokens - 1):

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            past_key_values = outputs.past_key_values

            synchronize(device)

            memory = current_torch_memory(device)

            if memory is not None:
                observed_torch.append(memory)


    if device.type == "cuda":

        peak_torch = (
            torch.cuda.max_memory_allocated()
        )

        memory_method = (
            "CUDA allocator peak memory"
        )

    elif device.type == "mps":

        peak_torch = (
            max(observed_torch)
            if observed_torch
            else None
        )

        memory_method = (
            "maximum sampled PyTorch MPS allocation"
        )

    else:

        peak_torch = None

        memory_method = (
            "memory measurement unavailable"
        )


    return {
        "baseline_torch_bytes":
            baseline_torch,

        "baseline_driver_bytes":
            baseline_driver,

        "peak_torch_bytes":
            peak_torch,

        "final_driver_bytes":
            current_driver_memory(device),

        "method":
            memory_method,
    }


# ============================================================
# HELPERS
# ============================================================

def median_metric(results, key):

    return statistics.median(
        run[key]
        for run in results
    )


def bytes_to_mib(value):

    if value is None:
        return None

    return value / (1024 ** 2)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="LLM inference benchmark"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
    )

    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--json-out",
        default="reports/latest_benchmark.json",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    if args.prompt_tokens <= 0:
        raise ValueError(
            "prompt-tokens must be > 0"
        )

    if args.output_tokens <= 0:
        raise ValueError(
            "output-tokens must be > 0"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be > 0"
        )

    device = get_device(args.device)


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model
    )

    model.eval()

    model = model.to(device)


    input_ids = build_prompt(
        tokenizer,
        args.prompt_tokens,
        args.batch_size,
        device,
    )


    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------

    with torch.inference_mode():

        warmup = model(
            input_ids=input_ids,
            use_cache=True,
        )

    synchronize(device)

    del warmup

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps":
        torch.mps.empty_cache()


    # --------------------------------------------------------
    # TIMING RUNS
    # --------------------------------------------------------

    runs = []

    for _ in range(args.runs):

        result = benchmark_once(
            model,
            input_ids,
            args.output_tokens,
            device,
        )

        runs.append(result)


    # --------------------------------------------------------
    # MEDIAN METRICS
    # --------------------------------------------------------

    prefill = median_metric(
        runs,
        "prefill_s",
    )

    ttft = median_metric(
        runs,
        "ttft_s",
    )

    decode_total = median_metric(
        runs,
        "decode_total_s",
    )

    mean_itl = median_metric(
        runs,
        "mean_itl_s",
    )

    median_itl = median_metric(
        runs,
        "median_itl_s",
    )

    e2e = median_metric(
        runs,
        "e2e_s",
    )


    # --------------------------------------------------------
    # THROUGHPUT
    # --------------------------------------------------------

    batch_output_tokens = (
        args.batch_size
        * args.output_tokens
    )

    aggregate_output_tokens_per_second = (
        batch_output_tokens / e2e
    )


    decode_tokens = (
        args.batch_size
        * max(args.output_tokens - 1, 0)
    )

    decode_tokens_per_second = (
        decode_tokens / decode_total
        if decode_total > 0
        else 0.0
    )


    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory = measure_memory(
        model,
        input_ids,
        args.output_tokens,
        device,
    )

    parameter_bytes = model_parameter_bytes(
        model
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "timestamp":
            datetime.now().isoformat(),

        "configuration": {

            "model":
                args.model,

            "device":
                str(device),

            "dtype":
                str(
                    next(
                        model.parameters()
                    ).dtype
                ),

            "prompt_tokens":
                args.prompt_tokens,

            "output_tokens_per_request":
                args.output_tokens,

            "batch_size":
                args.batch_size,

            "measured_runs":
                args.runs,
        },

        "latency_ms": {

            "prefill":
                prefill * 1000,

            "model_local_ttft":
                ttft * 1000,

            "mean_itl_tpot":
                mean_itl * 1000,

            "median_itl":
                median_itl * 1000,

            "decode_total":
                decode_total * 1000,

            "end_to_end":
                e2e * 1000,
        },

        "throughput": {

            "decode_tokens_per_second":
                decode_tokens_per_second,

            "aggregate_output_tokens_per_second":
                aggregate_output_tokens_per_second,
        },

        "memory_mib": {

            "model_parameters":
                bytes_to_mib(
                    parameter_bytes
                ),

            "baseline_torch":
                bytes_to_mib(
                    memory[
                        "baseline_torch_bytes"
                    ]
                ),

            "observed_peak_torch":
                bytes_to_mib(
                    memory[
                        "peak_torch_bytes"
                    ]
                ),

            "baseline_driver":
                bytes_to_mib(
                    memory[
                        "baseline_driver_bytes"
                    ]
                ),

            "final_driver":
                bytes_to_mib(
                    memory[
                        "final_driver_bytes"
                    ]
                ),
        },

        "memory_measurement_method":
            memory["method"],
    }


    # --------------------------------------------------------
    # HUMAN-READABLE OUTPUT
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("LLM INFERENCE BENCHMARK")
    print("=" * 65)

    print(
        f"Model:              {args.model}"
    )

    print(
        f"Device:             {device}"
    )

    print(
        f"Prompt tokens:      {args.prompt_tokens}"
    )

    print(
        f"Output tokens:      {args.output_tokens}"
    )

    print(
        f"Batch size:         {args.batch_size}"
    )

    print("-" * 65)

    print(
        f"Prefill:            "
        f"{prefill * 1000:.3f} ms"
    )

    print(
        f"Model-local TTFT:   "
        f"{ttft * 1000:.3f} ms"
    )

    print(
        f"Mean ITL / TPOT:    "
        f"{mean_itl * 1000:.3f} ms"
    )

    print(
        f"Median ITL:         "
        f"{median_itl * 1000:.3f} ms"
    )

    print(
        f"End-to-end:         "
        f"{e2e * 1000:.3f} ms"
    )

    print("-" * 65)

    print(
        f"Decode throughput:  "
        f"{decode_tokens_per_second:.2f} tok/s"
    )

    print(
        f"Aggregate output:   "
        f"{aggregate_output_tokens_per_second:.2f} tok/s"
    )

    print("-" * 65)

    print(
        f"Model parameters:   "
        f"{bytes_to_mib(parameter_bytes):.2f} MiB"
    )


    observed_peak = bytes_to_mib(
        memory["peak_torch_bytes"]
    )

    if observed_peak is not None:

        print(
            f"Observed peak mem:  "
            f"{observed_peak:.2f} MiB"
        )

    print(
        f"Memory method:       "
        f"{memory['method']}"
    )


    # --------------------------------------------------------
    # JSON OUTPUT
    # --------------------------------------------------------

    output_path = Path(
        args.json_out
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )


    print("-" * 65)

    print(
        f"JSON report:        {output_path}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()