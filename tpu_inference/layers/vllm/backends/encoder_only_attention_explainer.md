# Architectural Explainer: EncoderOnlyAttention (ModernBERT) Support on TPUs

This document explains the architectural design and execution pipelines implemented to support `EncoderOnlyAttention` (used by models like ModernBERT) on TPUs inside the `tpu-inference` repository.

---

## Background: The Challenge of Encoder Attention on TPU

In vLLM, all sequences in a batch are packed into a flat 1D token dimension `[num_tokens, num_heads, head_dim]` to avoid padding overhead. 
*   For autoregressive **decoders**, `tpu-inference` uses the **Ragged Paged Attention (RPA)** kernel, which reads/writes Key and Value states from a **Paged KV Cache** in HBM using paged `block_tables`.
*   For **encoders** (like ModernBERT), there is no autoregressive phase. They process the entire sequence bidirectionally in a single pass. Consequently, **no KV Cache is allocated**, and no `block_tables` are generated.

Because the TPU's paged RPA kernel is strictly tied to a KV Cache, pure encoder blocks were previously unsupported on TPUs.

To solve this, we implemented **two distinct JAX-based execution pathways** inside `PallasAttentionBackendImpl.forward` that handle cache-free bidirectional attention.

---

## Pathway 1: High-Performance Flat-Padded FlashAttention (Default)

This is the recommended, high-performance pathway. It completely avoids paged memory overhead and runs with maximum locality in TPU SRAM.

### The `batch_size = 1` Flat-Padding Trick
The contiguous TPU `flash_attention` Pallas kernel expects batched 4D inputs `[batch_size, num_heads, seq_len, head_dim]`. Slicing the 1D packed tensor into a 4D batched tensor using `jax.vmap` would force us to pad *every* sequence to a constant `max_seq_len`, introducing massive memory and latency overhead.

Instead, we **treat the entire packed batch of sequences as a single giant sequence with `batch_size = 1`**:
1.  **Reshape & Transpose**: The 1D packed Q/K/V tensors are reshaped and transposed to head-first: `[num_heads, num_tokens, head_dim]`.
2.  **128-Token Alignment**: The Pallas kernel requires sequence lengths to be multiples of 128. We pad the *total* token length `num_tokens` to the next multiple of 128.
    *   *This introduces a maximum of **127 tokens of padding in total** for the entire batch, near-zero overhead.*
3.  **Batch Expansion**: We add a dummy batch dimension to get `[1, num_heads, padded_num_tokens, head_dim]`.
4.  **Flat `SegmentIds`**: We construct a flat `SegmentIds` array of shape `[1, padded_num_tokens]` by repeating the sequence indices `[0, 1, 2, ...]` by their respective `seq_lens`.
5.  **Attention Execution**: We run `sharded_flash_attention` with `batch_size=1` and `causal=False`. The kernel uses the `SegmentIds` to automatically mask out attention between different sequences, preventing any cross-sequence attention leakage.
6.  **Unpad**: We squeeze the batch dimension, slice off the small padding, and transpose/reshape back to vLLM's 1D packed shape.

---

## Pathway 2: Experimental Paged RPA (Benchmarking Option)

For completeness and to allow researchers to profile the performance gap between contiguous attention and paged attention, we also implemented a mathematically correct **RPA-based pipeline**.

### Real Batching and Dummy Cache Allocation
Since RPA requires a causal mask to apply `mm_prefix_range` (the vision-token mask override), we cannot use the `batch_size=1` flat-padding trick. Instead, we must use **real batching**:
1.  **Dummy TPU Cache Allocation**: Inside JAX, we dynamically allocate a dummy paged KV Cache in TPU HBM of static size `batch_size * max_pages_per_seq` (where `max_pages_per_seq = max_model_len // 16`).
2.  **Contiguous Page Mapping**: We generate a static `page_indices` array mapping each active sequence in the batch contiguously to its dummy HBM pages.
3.  **Paged Execution**: We instantiate the sharded RPA kernel via `sharded_ragged_paged_attention` and run it bidirectionally (`use_causal_mask=False`) with `update_kv_cache=True` (which performs the redundant K/V writes into the dummy cache to satisfy the `head_dim=64` kernel limit).
4.  **Safety**: Because RPA runs individually per sequence up to its `kv_lens[i]`, there is no cross-sequence attention leakage.

---

## How to Run and Benchmark

Both pathways are fully integrated into a single unified backend `PallasAttentionBackend` without any registry conflicts.

### Running Unit Tests
Verify that both pathways execute cleanly by running the pytest suite inside your TPU VM environment:
```bash
pytest tests/layers/vllm/backends/test_flash_attn.py
```

### Benchmarking the Pipelines
You can easily toggle between the two pipelines using the `TPU_INFERENCE_USE_RPA_FOR_ENCODER` environment variable to measure the exact TPU latency and bandwidth differences during ModernBERT inference:

*   **To benchmark contiguous FlashAttention (Default - Highly Performant)**:
    ```bash
    python3 benchmark_serving.py ...
    ```
*   **To benchmark paged RPA (Experimental - High HBM overhead)**:
    ```bash
    TPU_INFERENCE_USE_RPA_FOR_ENCODER=1 python3 benchmark_serving.py ...
    ```
