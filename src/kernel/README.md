# kernel

## Purpose

Low-level CPU computation kernels and architecture-specific math helpers.

All actual numerical computations happen here. This module has no dependencies on other modules and handles raw pointer operations for maximum efficiency.

## Responsibilities

- Matrix operations (multiplication, transpose, etc.)
- Attention mechanism computations
- Activation functions (GELU, ReLU, SiLU, etc.)
- Normalization operations (RMS norm, layer norm, etc.)
- Tensor element-wise operations
- Architecture-specific optimizations (SIMD, cache-aware algorithms)

## Dependencies

**Outgoing:**
- None - completely independent module

**Incoming:**
- Called by `tensor` to perform actual data updates on raw memory pointers

## Example Usage

```rust
// pseudo code
weight_ptr = storage_pointer(weight_buffer)
input_ptr = storage_pointer(input_buffer)
output_ptr = storage_pointer(output_buffer)

kernel_matmul(weight_ptr, input_ptr, output_ptr, rows, cols, batch_size)
kernel_rms_norm(output_ptr, weight_ptr, output_size, epsilon)
```

## Architecture Notes

- No allocations or high-level abstractions
- Works exclusively with primitive C-style pointers (obtained from `storage`)
- All functions are unsafe - caller is responsible for pointer validity and memory layout
- Designed for maximum performance on CPU baseline, extensible to GPU/SIMD implementations
