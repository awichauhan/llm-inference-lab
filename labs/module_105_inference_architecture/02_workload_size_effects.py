import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

PROMPT_LENGTHS = [8, 32, 128, 256]
OUTPUT_LENGTHS = [4, 16, 32, 64]

FIXED_OUTPUT_TOKENS = 16
FIXED_PROMPT_TOKENS = 64

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


def build_exact_prompt(
    tokenizer,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:

    text = (
        "Artificial intelligence systems process information "
        "using neural networks and transformer architectures. "
    ) * 100

    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    return tokens[:, :token_count].to(device)


def run_generation(
    model,
    input_ids: torch.Tensor,
    output_tokens: int,
    device: torch.device,
) -> dict:

    synchronize(device)

    request_start = time.perf_counter()

    # --------------------------------------------------
    # PREFILL
    # --------------------------------------------------

    with torch.inference_mode():

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

        # --------------------------------------------------
        # DECODE
        # --------------------------------------------------

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

    local_ttft = (
        first_token_time - request_start
    )

    decode_total = sum(decode_times)

    mean_itl = (
        statistics.mean(decode_times)
        if decode_times
        else 0.0
    )

    e2e = request_end - request_start

    return {
        "prefill": prefill_time,
        "ttft": local_ttft,
        "decode_total": decode_total,
        "mean_itl": mean_itl,
        "e2e": e2e,
    }


def median_result(results: list[dict]) -> dict:
    return {
        key: statistics.median(
            result[key]
            for result in results
        )
        for key in results[0]
    }


device = get_device()

print("Device:", device)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.eval()
model = model.to(device)


# --------------------------------------------------
# Warm-up
# --------------------------------------------------

warmup_input = build_exact_prompt(
    tokenizer,
    32,
    device,
)

with torch.inference_mode():
    model(
        input_ids=warmup_input,
        use_cache=True,
    )

synchronize(device)


# ==================================================
# EXPERIMENT A
#
# Prompt length → prefill cost
# ==================================================

print("\n")
print("=" * 70)
print("EXPERIMENT A — PROMPT LENGTH VS PREFILL")
print("=" * 70)

print(
    f"Output length fixed at "
    f"{FIXED_OUTPUT_TOKENS} tokens"
)


for prompt_tokens in PROMPT_LENGTHS:

    input_ids = build_exact_prompt(
        tokenizer,
        prompt_tokens,
        device,
    )

    runs = []

    for _ in range(MEASURED_RUNS):

        result = run_generation(
            model,
            input_ids,
            FIXED_OUTPUT_TOKENS,
            device,
        )

        runs.append(result)


    result = median_result(runs)


    print("\n" + "-" * 70)

    print(
        f"Prompt tokens: {prompt_tokens}"
    )

    print(
        f"Prefill: "
        f"{result['prefill'] * 1000:.3f} ms"
    )

    print(
        f"Model-local TTFT: "
        f"{result['ttft'] * 1000:.3f} ms"
    )

    print(
        f"Mean ITL: "
        f"{result['mean_itl'] * 1000:.3f} ms"
    )

    print(
        f"E2E: "
        f"{result['e2e'] * 1000:.3f} ms"
    )


# ==================================================
# EXPERIMENT B
#
# Output length → decode cost
# ==================================================

print("\n")
print("=" * 70)
print("EXPERIMENT B — OUTPUT LENGTH VS DECODE")
print("=" * 70)

print(
    f"Prompt length fixed at "
    f"{FIXED_PROMPT_TOKENS} tokens"
)


fixed_input = build_exact_prompt(
    tokenizer,
    FIXED_PROMPT_TOKENS,
    device,
)


for output_tokens in OUTPUT_LENGTHS:

    runs = []

    for _ in range(MEASURED_RUNS):

        result = run_generation(
            model,
            fixed_input,
            output_tokens,
            device,
        )

        runs.append(result)


    result = median_result(runs)


    decode_tokens = max(
        output_tokens - 1,
        1,
    )

    decode_tokens_per_second = (
        decode_tokens
        / result["decode_total"]
        if result["decode_total"] > 0
        else 0.0
    )


    print("\n" + "-" * 70)

    print(
        f"Output tokens: {output_tokens}"
    )

    print(
        f"Prefill: "
        f"{result['prefill'] * 1000:.3f} ms"
    )

    print(
        f"Decode total: "
        f"{result['decode_total'] * 1000:.3f} ms"
    )

    print(
        f"Mean ITL / TPOT: "
        f"{result['mean_itl'] * 1000:.3f} ms"
    )

    print(
        f"Decode tokens/sec: "
        f"{decode_tokens_per_second:.2f}"
    )

    print(
        f"E2E: "
        f"{result['e2e'] * 1000:.3f} ms"
    )