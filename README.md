# 🐋 my_little_deepseek

A toy implementation of **DeepSeek-R1-Distill-Qwen-1.5B** inference in pure Rust.

## 🎯 Project Vision & Core Constraints

This project is uniquely focused on a single fixed inference target. To maintain simplicity and low-level control, it strictly adheres to the following core constraints:
* **Environment:** Pure Rust with `std` and `libc` only.
* **Scope:** Inference only, no training capabilities.
* **Target:** Single model target with fixed architecture and config expectations.
* **Compute Baseline:** CPU only, single core, single thread for the current baseline.

**Origin model:** [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

## 🚀 Getting Started

Because the model weights are too large, `model/model.safetensors` is not included in this repository. You must download it manually from the official DeepSeek HuggingFace repository and place it in the `model/` directory.

To run the engine, use release mode for optimal performance:

```bash
cargo run --release
```

## ✨ Current Status & Constraints

* **Functional Inference:** The forward pass and token generation loop are complete.
* **Environment:** Pure Rust (only `std` and `libc`).
* **Compute:** Currently supports F32 and CPU-only single-thread execution.
* **Unsupported:** Korean text normalization is missing and there are no plans to implement it.

## 💻 Demonstration

Here is an example of the actual inference output from the engine:

```text
[] Initializing... done!
---------------------------------
[User]: hello!

[Assistant]: 
<think>
Alright, the user said "hello!" and then "hello!" again. I should respond in a friendly and welcoming manner. Maybe say hello back and offer help. Keepit simple and open-ended so they feel comfortable to ask anything.
</think>

Hello! How can I assist you today?<|end_of_sentence|>
[User]: What's your name?

[Assistant]: 
<think>
Okay, the user just asked, "What's your name?" I need to figure out how to respond appropriately. Since I'm an AI, I don't have a physical name, but I can provide information about me. I should keep it friendly and open-ended to encourage the user to share more. Maybe I can say something like, "I'm DeepSeek-R1, an AI assistant created exclusively by the Chinese Company DeepSeek. I'm here to help you with any questions or tasks you have in mind!" That should cover it and invite them to continue the conversation.
</think>

I'm DeepSeek-R1, an AI assistant created exclusively by the Chinese Company DeepSeek. I'm here to help you with any questions or tasks you have in mind!<|end_of_sentence|>
[User]: /exit
---------------------------------
Goodbye!

```

## 🏗️ Module Architecture

This project strictly separates the **Immutable Model** from the **Mutable Session**.

* **`config`**: Defines the model architecture and configuration metadata. Acts as a registry mapping files to the device layer.
* **`device`**: A hardware abstraction layer for memory management (currently CPU `mmap`). Provides storage for static weights and dynamic hidden states.
* **`kernel`**: CPU-specific, low-level SIMD-optimized math operations.
* **`tensor`**: Provides a safe 2D view over raw device memory. Offers high-level interfaces for matrix operations.
* **`model`**: The fixed, immutable neural network definition. It allocates necessary temporary buffers and passes them to sub-modules (`attention`, `feed_forward`, `tokenizer`, etc.).
* **`session`**: Manages the mutable state of an inference process, including unique KV Caches and token history.
* **`error`**: Consistent, domain-specific error handling across the entire engine.

## 🗺️ Roadmap

* **Latency Check**: Implement execution time measurement functionality to track inference steps (essential for future optimization).
* **CPU Multi-threading**: Add multi-thread support for CPU matrix computations.
* **NVIDIA GPU Support**: Port major calculations to CUDA for hardware acceleration.