import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

PROMPT = (
    "Artificial intelligence systems can improve "
    "software engineering by"
)

MAX_NEW_TOKENS = 10


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
# Load model
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

input_ids = tokenizer(
    PROMPT,
    return_tensors="pt",
)["input_ids"].to(device)


print("\nPrompt:")
print(PROMPT)

print("\nInitial input shape:")
print(tuple(input_ids.shape))


# --------------------------------------------------
# NAIVE AUTOREGRESSIVE GENERATION
# --------------------------------------------------

generated_ids = input_ids

total_positions_processed = 0

synchronize(device)
start_time = time.perf_counter()


with torch.inference_mode():

    for step in range(MAX_NEW_TOKENS):

        sequence_length = generated_ids.shape[1]

        total_positions_processed += sequence_length

        print(
            f"\nStep {step + 1:2d} | "
            f"input shape = {tuple(generated_ids.shape)} | "
            f"positions processed = {sequence_length}"
        )

        outputs = model(
            input_ids=generated_ids,
            use_cache=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=1,
        )

        print(
            "Generated token:",
            repr(tokenizer.decode(next_token_id[0]))
        )


synchronize(device)
end_time = time.perf_counter()


# --------------------------------------------------
# REPORT
# --------------------------------------------------

elapsed = end_time - start_time

print("\n" + "=" * 60)
print("NAIVE GENERATION REPORT")
print("=" * 60)

print(
    "Prompt tokens:",
    input_ids.shape[1]
)

print(
    "Generated tokens:",
    MAX_NEW_TOKENS
)

print(
    "Total token positions processed:",
    total_positions_processed
)

print(
    f"Generation time: {elapsed * 1000:.3f} ms"
)

print("\nGenerated text:")

print(
    tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
)