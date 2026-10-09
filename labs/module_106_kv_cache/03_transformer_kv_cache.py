import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# ONE CACHED ATTENTION HEAD
# ==================================================

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
        x,
        past_k=None,
        past_v=None,
    ):
        queries = self.query(x)

        keys_new = self.key(x)
        values_new = self.value(x)

        new_length = x.shape[1]

        if past_k is None:
            past_length = 0

            keys = keys_new
            values = values_new

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

        # Absolute positions for correct causal masking.
        query_positions = torch.arange(
            past_length,
            past_length + new_length,
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

        scores = (
            queries @ keys.transpose(-2, -1)
        )

        scores = scores * (
            keys.shape[-1] ** -0.5
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        weights = F.softmax(
            scores,
            dim=-1,
        )

        output = weights @ values

        return output, keys, values


# ==================================================
# MULTI-HEAD ATTENTION
# ==================================================

class CachedMultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        number_of_heads: int,
    ):
        super().__init__()

        if embedding_size % number_of_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by number_of_heads"
            )

        head_size = (
            embedding_size // number_of_heads
        )

        self.heads = nn.ModuleList([
            CachedCausalAttentionHead(
                embedding_size=embedding_size,
                head_size=head_size,
            )
            for _ in range(number_of_heads)
        ])

        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size,
        )

    def forward(
        self,
        x,
        past_cache=None,
    ):
        head_outputs = []
        present_cache = []

        for head_index, head in enumerate(self.heads):

            # No cache during prefill.
            if past_cache is None:
                past_k = None
                past_v = None

            else:
                past_k, past_v = (
                    past_cache[head_index]
                )

            output, present_k, present_v = head(
                x,
                past_k=past_k,
                past_v=past_v,
            )

            head_outputs.append(output)

            present_cache.append(
                (present_k, present_v)
            )

        concatenated = torch.cat(
            head_outputs,
            dim=-1,
        )

        output = self.output_projection(
            concatenated
        )

        return output, present_cache


# ==================================================
# TRANSFORMER BLOCK
# ==================================================

class CachedTransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        number_of_heads: int,
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = CachedMultiHeadAttention(
            embedding_size=embedding_size,
            number_of_heads=number_of_heads,
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_size,
                4 * embedding_size,
            ),
            nn.ReLU(),
            nn.Linear(
                4 * embedding_size,
                embedding_size,
            ),
        )

    def forward(
        self,
        x,
        past_cache=None,
    ):
        attention_output, present_cache = (
            self.attention(
                self.layer_norm_1(x),
                past_cache=past_cache,
            )
        )

        x = x + attention_output

        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x, present_cache


# ==================================================
# TRANSFORMER STACK
# ==================================================

class CachedTransformerStack(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        number_of_heads: int,
        number_of_layers: int,
    ):
        super().__init__()

        # ModuleList instead of Sequential:
        # every block now has extra cache state.
        self.blocks = nn.ModuleList([
            CachedTransformerBlock(
                embedding_size=embedding_size,
                number_of_heads=number_of_heads,
            )
            for _ in range(number_of_layers)
        ])

    def forward(
        self,
        x,
        past_key_values=None,
    ):
        present_key_values = []

        for layer_index, block in enumerate(
            self.blocks
        ):

            if past_key_values is None:
                layer_cache = None

            else:
                layer_cache = (
                    past_key_values[layer_index]
                )

            x, present_cache = block(
                x,
                past_cache=layer_cache,
            )

            present_key_values.append(
                present_cache
            )

        return x, present_key_values


# ==================================================
# CACHE INSPECTION
# ==================================================

def print_cache(
    cache,
    title,
):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for layer_index, layer_cache in enumerate(cache):

        for head_index, (k, v) in enumerate(
            layer_cache
        ):
            print(
                f"Layer {layer_index} | "
                f"Head {head_index} | "
                f"K={tuple(k.shape)} | "
                f"V={tuple(v.shape)}"
            )


# ==================================================
# EXPERIMENT
# ==================================================

torch.manual_seed(42)

BATCH_SIZE = 1
EMBEDDING_SIZE = 8
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 2


model = CachedTransformerStack(
    embedding_size=EMBEDDING_SIZE,
    number_of_heads=NUMBER_OF_HEADS,
    number_of_layers=NUMBER_OF_LAYERS,
)


# --------------------------------------------------
# PREFILL: 3-token prompt
# --------------------------------------------------

prompt_embeddings = torch.randn(
    BATCH_SIZE,
    3,
    EMBEDDING_SIZE,
)

prefill_output, cache = model(
    prompt_embeddings
)

print_cache(
    cache,
    "CACHE AFTER PREFILL",
)


# --------------------------------------------------
# DECODE 1: one new token
# --------------------------------------------------

token_4 = torch.randn(
    BATCH_SIZE,
    1,
    EMBEDDING_SIZE,
)

decode_output, cache = model(
    token_4,
    past_key_values=cache,
)

print_cache(
    cache,
    "CACHE AFTER DECODE STEP 1",
)


# --------------------------------------------------
# DECODE 2: another new token
# --------------------------------------------------

token_5 = torch.randn(
    BATCH_SIZE,
    1,
    EMBEDDING_SIZE,
)

decode_output, cache = model(
    token_5,
    past_key_values=cache,
)

print_cache(
    cache,
    "CACHE AFTER DECODE STEP 2",
)