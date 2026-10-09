import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedCausalAttentionHead(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        head_size: int,
    ):
        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            head_size,
            bias=False,
        )

        self.key = nn.Linear(
            embedding_size,
            head_size,
            bias=False,
        )

        self.value = nn.Linear(
            embedding_size,
            head_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ):
        # x:
        # [B, T_new, embedding_size]

        queries = self.query(x)

        # IMPORTANT:
        # K and V are calculated only for x.
        keys_new = self.key(x)
        values_new = self.value(x)

        new_sequence_length = x.shape[1]

        # --------------------------------------------------
        # No previous cache = PREFILL
        # --------------------------------------------------

        if past_k is None:

            past_length = 0

            keys = keys_new
            values = values_new

        # --------------------------------------------------
        # Previous cache exists = DECODE
        # --------------------------------------------------

        else:

            past_length = past_k.shape[1]

            keys = torch.cat(
                [past_k, keys_new],
                dim=1,
            )

            values = torch.cat(
                [past_v, values_new],
                dim=1,
            )

        total_length = keys.shape[1]

        # --------------------------------------------------
        # Causal mask using absolute token positions
        # --------------------------------------------------

        query_positions = torch.arange(
            past_length,
            past_length + new_sequence_length,
            device=x.device,
        )

        key_positions = torch.arange(
            total_length,
            device=x.device,
        )

        mask = (
            key_positions.unsqueeze(0)
            <= query_positions.unsqueeze(1)
        )

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        attention_scores = (
            queries @ keys.transpose(-2, -1)
        )

        attention_scores = (
            attention_scores
            * (keys.shape[-1] ** -0.5)
        )

        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        output = (
            attention_weights @ values
        )

        # This becomes the cache for the next call.
        present_k = keys
        present_v = values

        print(
            f"Input T={new_sequence_length} | "
            f"past T={past_length} | "
            f"new K={tuple(keys_new.shape)} | "
            f"cache T={total_length}"
        )

        return output, present_k, present_v


# ==================================================
# EXPERIMENT
# ==================================================

torch.manual_seed(42)

BATCH_SIZE = 1
EMBEDDING_SIZE = 8
HEAD_SIZE = 4


attention = CachedCausalAttentionHead(
    embedding_size=EMBEDDING_SIZE,
    head_size=HEAD_SIZE,
)


# --------------------------------------------------
# Simulate embeddings for a 3-token prompt
# --------------------------------------------------

prompt_embeddings = torch.randn(
    BATCH_SIZE,
    3,
    EMBEDDING_SIZE,
)


print("\n" + "=" * 60)
print("PREFILL")
print("=" * 60)


prefill_output, k_cache, v_cache = attention(
    prompt_embeddings
)


print("\nK cache shape after prefill:")
print(tuple(k_cache.shape))

print("V cache shape after prefill:")
print(tuple(v_cache.shape))


# --------------------------------------------------
# Simulate first generated token
# --------------------------------------------------

token_4_embedding = torch.randn(
    BATCH_SIZE,
    1,
    EMBEDDING_SIZE,
)


print("\n" + "=" * 60)
print("DECODE STEP 1")
print("=" * 60)


decode_output, k_cache, v_cache = attention(
    token_4_embedding,
    past_k=k_cache,
    past_v=v_cache,
)


print("\nK cache shape after decode:")
print(tuple(k_cache.shape))

print("V cache shape after decode:")
print(tuple(v_cache.shape))


# --------------------------------------------------
# Another generated token
# --------------------------------------------------

token_5_embedding = torch.randn(
    BATCH_SIZE,
    1,
    EMBEDDING_SIZE,
)


print("\n" + "=" * 60)
print("DECODE STEP 2")
print("=" * 60)


decode_output, k_cache, v_cache = attention(
    token_5_embedding,
    past_k=k_cache,
    past_v=v_cache,
)


print("\nK cache shape after second decode:")
print(tuple(k_cache.shape))

print("V cache shape after second decode:")
print(tuple(v_cache.shape))