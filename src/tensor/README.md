# tensor

## Purpose

Provides tensor abstractions and calculation helpers for managing model data and computations.

Tensors are the primary data structure for all model weights, activations, and embeddings. This module bridges high-level computation (called by `model` and `session`) with low-level kernel operations.

## Responsibilities

- Define tensor traits and types for numerical data
- Allocate tensors via `storage` backend
- Provide calculation functions that:
  - Request raw memory pointers from `storage`
  - Call `kernel` for actual mathematical operations
  - Update tensor data in-place
- Support reshaping, slicing, and transposition operations
- Track tensor metadata (shape, dtype, etc.)

## Dependencies

**Outgoing:**
- Calls `storage` to allocate/deallocate tensor data
- Calls `kernel` to perform actual computations on raw pointers

**Incoming:**
- Used by `model` to store weights and compute intermediate activations
- Used by `session` to store embedding vectors

## Example Usage

```rust
// pseudo code
weights = create_tensor(shape)
embedding = create_tensor(shape)

updated_embedding = tensor_apply_rms_norm(embedding, scale, epsilon)
output_tensor = tensor_matmul(input_tensor, weight_tensor)
```

## Architecture Notes

- Tensor is the primary interface for `model` and `session` - never access `storage` directly
- All tensor modifications happen through calculation functions (never expose raw pointers to callers)
- Tensor holds exclusive ownership of its storage memory
- Calculations are lazy or immediate depending on implementation (sync or async)
- Only `model` calls tensor calculation functions; `session` only reads results
