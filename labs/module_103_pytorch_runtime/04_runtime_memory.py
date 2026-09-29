import gc

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


def bytes_to_mb(value: int) -> float:
    return value / (1024 ** 2)


def model_size_bytes(model) -> int:
    total = 0

    for parameter in model.parameters():
        total += parameter.numel() * parameter.element_size()

    return total


def print_mps_memory(label: str) -> None:
    print(f"\n{label}")

    allocated = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()

    print(
        "PyTorch allocated:",
        f"{bytes_to_mb(allocated):.2f} MB"
    )

    print(
        "Driver allocated:",
        f"{bytes_to_mb(driver):.2f} MB"
    )


def cleanup(device: torch.device) -> None:
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()

    synchronize(device)


def run_normal(model, inputs, device):
    cleanup(device)

    before = torch.mps.current_allocated_memory()

    outputs = model(**inputs)

    synchronize(device)

    after = torch.mps.current_allocated_memory()

    logits = outputs.logits

    print("\nNORMAL EXECUTION")
    print("-" * 50)
    print("requires_grad:", logits.requires_grad)
    print("grad_fn:", logits.grad_fn)

    print(
        "memory increase:",
        f"{bytes_to_mb(after - before):.2f} MB"
    )

    print_mps_memory(
        "Memory while normal output is alive"
    )

    return outputs


def run_no_grad(model, inputs, device):
    cleanup(device)

    before = torch.mps.current_allocated_memory()

    with torch.no_grad():
        outputs = model(**inputs)

    synchronize(device)

    after = torch.mps.current_allocated_memory()

    logits = outputs.logits

    print("\nNO_GRAD")
    print("-" * 50)
    print("requires_grad:", logits.requires_grad)
    print("grad_fn:", logits.grad_fn)

    print(
        "memory increase:",
        f"{bytes_to_mb(after - before):.2f} MB"
    )

    print_mps_memory(
        "Memory while no_grad output is alive"
    )

    return outputs


def run_inference_mode(model, inputs, device):
    cleanup(device)

    before = torch.mps.current_allocated_memory()

    with torch.inference_mode():
        outputs = model(**inputs)

    synchronize(device)

    after = torch.mps.current_allocated_memory()

    logits = outputs.logits

    print("\nINFERENCE MODE")
    print("-" * 50)
    print("requires_grad:", logits.requires_grad)
    print("grad_fn:", logits.grad_fn)

    print(
        "memory increase:",
        f"{bytes_to_mb(after - before):.2f} MB"
    )

    print_mps_memory(
        "Memory while inference output is alive"
    )

    return outputs


device = get_device()

print("Device:", device)

if device.type != "mps":
    raise RuntimeError(
        "This version of the lab currently expects Apple MPS."
    )


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.eval()
model = model.to(device)


inputs = tokenizer(
    "Artificial intelligence is changing the way software engineers build systems.",
    return_tensors="pt"
)

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}


print(
    "\nRaw model parameter size:",
    f"{bytes_to_mb(model_size_bytes(model)):.2f} MB"
)

print_mps_memory(
    "Memory after model + inputs are loaded"
)


# Warm-up

with torch.inference_mode():
    model(**inputs)

synchronize(device)

cleanup(device)


# --------------------------------------------------
# Normal
# --------------------------------------------------

normal_outputs = run_normal(
    model,
    inputs,
    device
)

del normal_outputs

cleanup(device)

print_mps_memory(
    "After deleting normal output"
)


# --------------------------------------------------
# no_grad
# --------------------------------------------------

no_grad_outputs = run_no_grad(
    model,
    inputs,
    device
)

del no_grad_outputs

cleanup(device)

print_mps_memory(
    "After deleting no_grad output"
)


# --------------------------------------------------
# inference_mode
# --------------------------------------------------

inference_outputs = run_inference_mode(
    model,
    inputs,
    device
)

del inference_outputs

cleanup(device)

print_mps_memory(
    "After deleting inference output"
)