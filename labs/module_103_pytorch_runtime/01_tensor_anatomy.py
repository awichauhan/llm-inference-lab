import torch

def inspect_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}")
    print("-" * 50)

    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)
    print("device:", tensor.device)

    print("num elements:", tensor.numel())
    print("bytes per element:", tensor.element_size())

    logical_bytes = tensor.numel() * tensor.element_size()

    print("logical tensor bytes:", logical_bytes)
    print("storage bytes:", tensor.untyped_storage().nbytes())

    print("stride:", tensor.stride())
    print("storage offset:", tensor.storage_offset())
    print("contiguous:", tensor.is_contiguous())

    print("data pointer:", tensor.data_ptr())


# --------------------------------------------------
# Experiment 1: FP32 tensor
# --------------------------------------------------

x = torch.arange(
    24,
    dtype=torch.float32
).reshape(2, 3, 4)

inspect_tensor("Original FP32 tensor", x)


# --------------------------------------------------
# Experiment 2: same values, different dtype
# --------------------------------------------------

x_fp16 = x.to(torch.float16)

inspect_tensor("FP16 tensor", x_fp16)


# --------------------------------------------------
# Experiment 3: transpose
# --------------------------------------------------

x_transposed = x.transpose(1, 2)

inspect_tensor("Transposed tensor", x_transposed)


# --------------------------------------------------
# Experiment 4: force contiguous storage
# --------------------------------------------------

x_contiguous = x_transposed.contiguous()

inspect_tensor("Contiguous copy", x_contiguous)