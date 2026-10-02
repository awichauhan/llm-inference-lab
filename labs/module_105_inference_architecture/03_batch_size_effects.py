import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

BATCH_SIZES = [1, 2, 4, 8]

PROMPT_TOKENS = 64
OUTPUT_TOKENS = 32

MEASURED_RUNS = 3


def get_device() -> torch.device:
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


def build_prompt(
    tokenizer,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:

    text = (
        "Artificial intelligence systems use transformer "
        "architectures to process and generate language. "
    ) * 100

    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    return input_ids[:, :token_count].to(device)


def run_batch(
    model,
    base_input: torch.Tensor,
    batch_size: int,
    output_tokens: int,
    device: torch.device,
) -> dict:

    # [1, T] → [B, T]
    input_ids = base_input.repeat(
        batch_size,
        1,
    )

    synchronize(device)

    request_start = time.perf_counter()

    with torch.inference_mode():

        # ----------------------------------------------
        # PREFILL
        # ----------------------------------------------

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


        # ----------------------------------------------
        # DECODE
        # ----------------------------------------------

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


    prefill_time = (
        prefill_end - prefill_start
    )

    ttft = (
        first_token_time - request_start
    )

    decode_total = sum(decode_times)

    mean_tpot = statistics.mean(
        decode_times
    )

    e2e = (
        request_end - request_start
    )


    # Across the entire batch.
    total_output_tokens = (
        batch_size * output_tokens
    )

    aggregate_throughput = (
        total_output_tokens / e2e
    )


    decode_tokens = (
        batch_size
        * (output_tokens - 1)
    )

    decode_throughput = (
        decode_tokens / decode_total
    )


    return {
        "prefill": prefill_time,
        "ttft": ttft,
        "tpot": mean_tpot,
        "decode_total": decode_total,
        "e2e": e2e,
        "aggregate_throughput": aggregate_throughput,
        "decode_throughput": decode_throughput,
    }


def median_results(
    results: list[dict],
) -> dict:

    return {
        key: statistics.median(
            result[key]
            for result in results
        )
        for key in results[0]
    }


device = get_device()

print("Device:", device)
print("Prompt tokens:", PROMPT_TOKENS)
print("Output tokens/request:", OUTPUT_TOKENS)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.eval()
model = model.to(device)


base_input = build_prompt(
    tokenizer,
    PROMPT_TOKENS,
    device,
)


# ==================================================
# BATCH SWEEP
# ==================================================

for batch_size in BATCH_SIZES:

    # Warm-up at this batch size.
    with torch.inference_mode():

        warmup_ids = base_input.repeat(
            batch_size,
            1,
        )

        model(
            input_ids=warmup_ids,
            use_cache=True,
        )

    synchronize(device)


    runs = []

    for _ in range(MEASURED_RUNS):

        result = run_batch(
            model,
            base_input,
            batch_size,
            OUTPUT_TOKENS,
            device,
        )

        runs.append(result)


    result = median_results(runs)


    print("\n" + "=" * 65)

    print(
        f"BATCH SIZE: {batch_size}"
    )

    print("-" * 65)

    print(
        f"Prefill: "
        f"{result['prefill'] * 1000:.3f} ms"
    )

    print(
        f"Model-local TTFT: "
        f"{result['ttft'] * 1000:.3f} ms"
    )

    print(
        f"Mean TPOT / ITL: "
        f"{result['tpot'] * 1000:.3f} ms"
    )

    print(
        f"E2E batch latency: "
        f"{result['e2e'] * 1000:.3f} ms"
    )

    print(
        f"Decode throughput: "
        f"{result['decode_throughput']:.2f} tokens/s"
    )

    print(
        f"Aggregate output throughput: "
        f"{result['aggregate_throughput']:.2f} tokens/s"
    )
