# session

## Purpose

Manages chat session state and orchestrates inference for ongoing conversations.

Each `session` maintains its own embedding vector representing the current conversation state, independent of other sessions.

## Responsibilities

- Store current session embedding vector as a `tensor`
- Maintain chat history and special tokens (like padding, bos, eos)
- Invoke `model` inference to compute next tokens
- Handle token-to-embedding mapping for user input
- Update session state with model outputs

## Dependencies

**Outgoing:**
- Holds embedding vectors as `tensor` objects
- Calls `model` to compute next token predictions
  - Multiple sessions can share the same `model` instance

**Incoming:**
- Created and managed by application code

## Example Usage

```rust
// pseudo code
session = create_session(shared_model)

input_tokens = tokenize_user_input(text)
session_embedding = build_embedding_tensor(input_tokens)

next_token = model_infer(shared_model, session_embedding)
session_update(session, next_token)
```

## Architecture Notes

- Session is independent of other sessions - different sessions with the same `model` produce different outputs based on their conversation history
- All embeddings are stored as `tensor` objects to leverage tensor computation capabilities
- Session does not directly perform calculations - it delegates to `model` for inference
- Special tokens (padding, begin-of-sequence, end-of-sequence) are managed here for protocol-specific requirements
