import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


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


def run_normal(model, inputs, device):
    synchronize(device)

    start = time.perf_counter()

    outputs = model(**inputs)

    synchronize(device)

    elapsed = time.perf_counter() - start

    return outputs.logits, elapsed


def run_no_grad(model, inputs, device):
    synchronize(device)

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    synchronize(device)

    elapsed = time.perf_counter() - start

    return outputs.logits, elapsed


def run_inference_mode(model, inputs, device):
    synchronize(device)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

    synchronize(device)

    elapsed = time.perf_counter() - start

    return outputs.logits, elapsed


device = get_device()

print("Device:", device)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.eval()
model = model.to(device)


inputs = tokenizer(
    "Artificial intelligence is",
    return_tensors="pt"
)

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}


# Warm-up
with torch.inference_mode():
    for _ in range(3):
        model(**inputs)

synchronize(device)


# --------------------------------------------------
# Normal execution
# --------------------------------------------------

normal_logits, normal_time = run_normal(
    model,
    inputs,
    device
)

print("\nNORMAL EXECUTION")
print("-" * 50)
print("time:", normal_time)
print("requires_grad:", normal_logits.requires_grad)
print("grad_fn:", normal_logits.grad_fn)


# --------------------------------------------------
# no_grad
# --------------------------------------------------

no_grad_logits, no_grad_time = run_no_grad(
    model,
    inputs,
    device
)

print("\nNO_GRAD")
print("-" * 50)
print("time:", no_grad_time)
print("requires_grad:", no_grad_logits.requires_grad)
print("grad_fn:", no_grad_logits.grad_fn)


# --------------------------------------------------
# inference_mode
# --------------------------------------------------

inference_logits, inference_time = run_inference_mode(
    model,
    inputs,
    device
)

print("\nINFERENCE MODE")
print("-" * 50)
print("time:", inference_time)
print("requires_grad:", inference_logits.requires_grad)
print("grad_fn:", inference_logits.grad_fn)