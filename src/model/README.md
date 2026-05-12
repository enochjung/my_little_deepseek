# model

## Purpose

The actual inference worker that performs transformer computations to generate model outputs.

When given a configuration and a session with embedding vectors, `model` orchestrates the inference pipeline to compute the next token probability distribution.

## Responsibilities

- Load and initialize model weights from `config`
- Orchestrate forward passes through transformer layers
- Manage computation flow: input → embedding → transformer layers → LM head → output logits
- Implement tokenizer operations for token-to-embedding mapping
- Support multiple concurrent `session` instances with shared model weights

## Dependencies

**Outgoing:**
- Uses `config` to access parsed model files and hyperparameters
- Uses `tensor` objects for storing and computing on weights and intermediate activations
- Directly calls `tensor` calculation functions (never accesses `storage` directly)

**Incoming:**
- Receives `tensor` objects from `session` containing embedding vectors
- Processes tensors and returns computed outputs to `session`

## Example Usage

```rust
// pseudo code
config = build_config(...)
model = build_model_from_config(config)
session_embedding = session_embedding_tensor()

next_token = model_forward(model, session_embedding)
```

## Architecture Notes

- Each `model` instance can serve multiple `session` instances
- Model weights are stored as `tensor` objects, not accessed directly from `storage`
- All computation is driven through `tensor` calculation methods, which internally use `kernel` for actual math
- A single `model` with the same `config` will produce identical outputs across different `session` instances
