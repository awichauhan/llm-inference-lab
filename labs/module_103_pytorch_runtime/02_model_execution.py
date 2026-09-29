import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def inspect_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}")
    print("-" * 50)

    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)
    print("device:", tensor.device)
    print("requires_grad:", tensor.requires_grad)


# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

device = get_device()

print("Selected device:", device)


# --------------------------------------------------
# 2. Load tokenizer and model
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.eval()


# --------------------------------------------------
# 3. Inspect model parameter BEFORE device transfer
# --------------------------------------------------

first_parameter = next(model.parameters())

inspect_tensor(
    "First model parameter BEFORE .to(device)",
    first_parameter
)


# --------------------------------------------------
# 4. Tokenize input
# --------------------------------------------------

prompt = "Artificial intelligence is"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inspect_tensor(
    "input_ids BEFORE .to(device)",
    inputs["input_ids"]
)

print("\nToken IDs:")
print(inputs["input_ids"])


# --------------------------------------------------
# 5. Move model to accelerator
# --------------------------------------------------

model = model.to(device)

first_parameter = next(model.parameters())

inspect_tensor(
    "First model parameter AFTER .to(device)",
    first_parameter
)


# --------------------------------------------------
# 6. Move input tensors to same device
# --------------------------------------------------

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

inspect_tensor(
    "input_ids AFTER .to(device)",
    inputs["input_ids"]
)


# --------------------------------------------------
# 7. Forward pass
# --------------------------------------------------

outputs = model(**inputs)

logits = outputs.logits

inspect_tensor(
    "Output logits",
    logits
)


# --------------------------------------------------
# 8. Predict next token
# --------------------------------------------------

next_token_logits = logits[:, -1, :]

next_token_id = torch.argmax(
    next_token_logits,
    dim=-1
)

predicted_id = next_token_id.item()

print("\nNext token ID:", predicted_id)
print(
    "Next token:",
    repr(tokenizer.decode([predicted_id]))
)


# --------------------------------------------------
# 9. Inspect autograd
# --------------------------------------------------

print("\nAutograd")
print("-" * 50)

print(
    "logits.requires_grad:",
    logits.requires_grad
)

print(
    "logits.grad_fn:",
    logits.grad_fn
)