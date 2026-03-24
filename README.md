# 🐋 my_little_deepseek

Toy implementation of **DeepSeek-R1-Distill-Qwen-1.5B** inference in pure Rust.

- Origin model: 🔗 https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## 🧠 What This Is

This project is a Rust implementation focused on a single fixed inference target: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

Core constraints:
- Pure Rust with `std` and `libc` only.
- Inference only (no training).
- Single model target: only `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` with its fixed architecture/config expectations.
- CPU only, single core, single thread (current baseline).
- Work with AI agents 🤖.

## ✅ Current Status

- Model data loading (Unicode, exclusions, merges, vocab)
- Prompt token assembly with model-specific special tokens
- Tokenizer pipeline (normalizer, pretokenizer, model/BPE encoding)

## 🚧 Roadmap

- Implement embedding lookup and decoder forward pass.
- Implement LM head and token selection loop.
- Add runtime/latency measurement for inference steps.
- Optimize for speed on CPU baseline.
- Port major matrix operations to CUDA.

## 🗂️ Project Structure

```text
src/                             # Rust source root
├── main.rs
└── inference/                   # Top-level inference module
    ├── error.rs
    ├── mod.rs
    ├── data/                    # Model data loaders/parsers
    │   ├── exclusion.rs
    │   ├── merge.rs
    │   ├── mod.rs
    │   ├── unicode.rs
    │   └── vocab.rs
    ├── engine/                  # Inference orchestration layer
    │   ├── mod.rs
    │   ├── special_token.rs
    │   └── tokenizer/           # Tokenizer pipeline
    │       ├── mod.rs
    │       ├── model/           # Vocab + merge encoding
    │       │   ├── merge.rs
    │       │   ├── mod.rs
    │       │   └── vocab.rs
    │       ├── normalizer/      # Text normalization stage
    │       │   └── mod.rs
    │       └── pretokenizer/    # Split and byte-level stage
    │           ├── byte_level.rs
    │           ├── mod.rs
    │           └── split.rs
    └── utils/                   # Shared low-level helpers
        ├── mmap.rs
        └── mod.rs
```