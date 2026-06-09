# Source Tree

This directory contains the core implementation of the inference engine. The project follows a strict architectural pattern: **Model** (Immutable, Stateless) vs. **Session** (Mutable, Stateful). 

## Module Architecture

- **`config`**
  Defines the model architecture and configuration. It parses metadata from model artifacts to provide a blueprint for initialization. It does not load full weights, but acts as the registry for mapping files to the `device` layer.

- **`device`**
  A hardware abstraction layer for memory management. It provides the physical storage for both static model weights and dynamic hidden states (like KV Caches). While currently supporting CPU-based `mmap`, it is designed to facilitate future integration with GPU libraries.
  - *Relationship*: `config` induces file mapping here; `session` allocates KV Caches here; `tensor` relies on this for data backing.

* **`kernel`**
  **Strictly for CPU computation.** This module contains low-level, SIMD-optimized mathematical operations specifically designed for the CPU. It should conceptually be part of `device/cpu/`, but is separated for implementation convenience.
  * *Relationship*: Called exclusively by the CPU-specific implementation within the `device` module.

- **`tensor`**
  Provides a 2D view over raw device memory. It manages ownership and ensures memory safety/immutability for specific 2D memory regions. It offers high-level interfaces for matrix operations.
  - *Relationship*: Uses `device` to perform computations on its backing memory.

- **`model`**
  The fixed, immutable neural network definition. Once initialized from a `config`, it remains constant. It handles the allocation of all temporary buffers required for inference and supplies them to sub-modules as needed (sub-modules never allocate their own buffers).
  - *Sub-modules*:
    - `attention`: Multi-head/Grouped-query attention mechanisms.
    - `feed_forward`: FFN layers with non-linear activation.
    - `rms_norm`: Layer normalization.
    - `rope`: Rotary Positional Embeddings.
    - `sampling`: Logit processing and token selection.
    - `token_embedding`: Input ID to vector conversion.
    - `tokenizer`: Byte-level BPE text encoding/decoding.

- **`session`**
  Manages the mutable state of an inference process. While the `model` is shared and immutable, multiple sessions can exist simultaneously, each maintaining its own unique KV Cache and token history.
  - *Relationship*: References the `model` for computation and uses `device`/`tensor` to store evolving hidden states as text generation progresses.

- **`error`**
  Provides consistent, domain-specific error handling for the entire engine.

---

## Data & Execution Flow

1. **Initialization (`main` -> `model`)**: The `main` process provides the `config`. `model` initializes the architecture, creates the tokenizer, and maps weights into `device` memory.
2. **Session Creation**: Each `session` is created from the `model`, allocating its own scratchpad and KV Cache buffers.
3. **Inference**: When processing input, the `session` coordinates with the `model` to perform the forward pass.
4. **Buffering**: `model` allocates necessary temporary `tensor` buffers and passes them down to layers for computation, ensuring efficient memory reuse.