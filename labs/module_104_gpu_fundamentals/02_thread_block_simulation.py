import math


NUM_ELEMENTS = 100
THREADS_PER_BLOCK = 32


num_blocks = math.ceil(
    NUM_ELEMENTS / THREADS_PER_BLOCK
)


print("Elements:", NUM_ELEMENTS)
print("Threads per block:", THREADS_PER_BLOCK)
print("Blocks required:", num_blocks)


for block_idx in range(num_blocks):

    print(f"\nBLOCK {block_idx}")
    print("-" * 40)

    for thread_idx in range(THREADS_PER_BLOCK):

        global_idx = (
            block_idx * THREADS_PER_BLOCK
            + thread_idx
        )

        if global_idx < NUM_ELEMENTS:

            print(
                f"block={block_idx:2d} "
                f"thread={thread_idx:2d} "
                f"global={global_idx:3d}"
            )