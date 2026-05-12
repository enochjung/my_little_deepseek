# storage

## Purpose

Provides physical memory abstraction for data allocation and management.

Instead of using Rust `Vec` for model inference (except during tokenization), all data is allocated through `storage` to enable future GPU support and efficient memory management.

## Memory Types

- `storage::Host` - CPU memory for model data, weights, and intermediate computations
- `storage::Device` - GPU memory (infrastructure for future CUDA implementation)

## Responsibilities

- Allocate and deallocate memory for tensors and buffers
- Provide raw pointers to underlying memory for `kernel` operations
- Manage memory layout and alignment for efficient computation
- Track memory lifecycle and prevent use-after-free

## Dependencies

**Outgoing:**
- None - completely independent module

**Incoming:**
- Called by `tensor` to allocate/deallocate memory for tensor data
- Called by `config` for storing loaded files (weights, vocab, merges, etc.)

## Example Usage

```rust
// pseudo code
buffer = host_allocate(size_in_bytes)
ptr = storage_pointer(buffer)

kernel_call(ptr, ...)

release_storage(buffer)
```

## Architecture Notes

- Memory management is abstracted from high-level modules
- All file data from `config` is stored in `storage::Host`
- All tensor data (weights, activations, embeddings) is stored in `storage`
- Raw pointers are only exposed to `kernel` for computation
- Storage provides the foundation for multi-device support (CPU/GPU)
