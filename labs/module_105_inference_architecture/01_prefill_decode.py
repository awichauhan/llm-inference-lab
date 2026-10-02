import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

PROMPT = (
    "Artificial intelligence systems can improve "
    "software engineering by"
)

MAX_NEW_TOKENS = 20


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


device = get_device()

print("Device:", device)


# --------------------------------------------------
# Load tokenizer + model
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.eval()
model = model.to(device)


# --------------------------------------------------
# Tokenize prompt
# --------------------------------------------------

inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
)

input_ids = inputs["input_ids"].to(device)


print("\nPROMPT")
print("-" * 60)
print(PROMPT)

print("\nPrompt tokens:", input_ids.shape[1])
print("Input shape:", tuple(input_ids.shape))


# ==================================================
# PREFILL
# ==================================================
#
# Entire prompt enters the model at once.
#
# Shape:
#
# [batch, prompt_tokens]
#
# The model processes every prompt position
# and builds the initial KV cache.
# ==================================================

synchronize(device)

request_start = time.perf_counter()


with torch.inference_mode():

    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        use_cache=True,
    )

    synchronize(device)

    prefill_end = time.perf_counter()


    # Last prompt position predicts first output token.

    next_token_logits = outputs.logits[:, -1, :]

    next_token_id = torch.argmax(
        next_token_logits,
        dim=-1,
        keepdim=True,
    )


    synchronize(device)

    first_token_time = time.perf_counter()


    # KV state created during prefill.
    past_key_values = outputs.past_key_values


    generated_tokens = [
        next_token_id.item()
    ]


    first_token_text = tokenizer.decode(
        next_token_id[0]
    )


    print("\nPREFILL")
    print("-" * 60)

    print(
        f"Prefill time: "
        f"{(prefill_end - prefill_start) * 1000:.3f} ms"
    )

    print(
        "First generated token:",
        repr(first_token_text)
    )


    # ==================================================
    # DECODE LOOP
    # ==================================================
    #
    # Only ONE new token enters on each step.
    #
    # The previous K/V states are reused.
    # ==================================================

    decode_intervals = []

    current_token = next_token_id


    for step in range(1, MAX_NEW_TOKENS):

        decode_start = time.perf_counter()


        outputs = model(
            input_ids=current_token,
            past_key_values=past_key_values,
            use_cache=True,
        )


        synchronize(device)


        next_token_logits = outputs.logits[:, -1, :]

        current_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )


        synchronize(device)

        decode_end = time.perf_counter()


        step_time = (
            decode_end - decode_start
        )

        decode_intervals.append(step_time)


        generated_tokens.append(
            current_token.item()
        )


        past_key_values = outputs.past_key_values


        token_text = tokenizer.decode(
            current_token[0]
        )


        print(
            f"Decode step {step:2d}: "
            f"{step_time * 1000:8.3f} ms"
            f"  token={repr(token_text)}"
        )


request_end = time.perf_counter()


# ==================================================
# METRICS
# ==================================================

local_ttft = (
    first_token_time - request_start
)

e2e_latency = (
    request_end - request_start
)

output_tokens = len(generated_tokens)


if decode_intervals:

    mean_itl = statistics.mean(
        decode_intervals
    )

    median_itl = statistics.median(
        decode_intervals
    )

else:
    mean_itl = 0.0
    median_itl = 0.0


generation_rate = (
    output_tokens / e2e_latency
)


generated_text = tokenizer.decode(
    generated_tokens
)


print("\n" + "=" * 60)
print("INFERENCE REPORT")
print("=" * 60)

print("Model:", MODEL_NAME)
print("Device:", device)

print(
    "Prompt tokens:",
    input_ids.shape[1]
)

print(
    "Output tokens:",
    output_tokens
)

print(
    f"Model-local TTFT: "
    f"{local_ttft * 1000:.3f} ms"
)

print(
    f"Mean ITL / TPOT: "
    f"{mean_itl * 1000:.3f} ms"
)

print(
    f"Median ITL: "
    f"{median_itl * 1000:.3f} ms"
)

print(
    f"End-to-end latency: "
    f"{e2e_latency * 1000:.3f} ms"
)

print(
    f"Output tokens/sec: "
    f"{generation_rate:.2f}"
)

print("\nGenerated continuation:")
print(repr(generated_text))