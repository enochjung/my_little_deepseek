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
- Work with AI agents. 🤖

## ✅ Current Status

- Model data loading (Unicode, exclusions, merges, vocab, model.safetensors)
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
├── inference/                   # Top-level inference module
│   ├── data/                    # Model data loaders/parsers
│   │   ├── exclusion.rs
│   │   ├── merge.rs
│   │   ├── mod.rs
│   │   ├── unicode.rs
│   │   ├── vocab.rs
│   │   └── weight.rs
│   ├── engine/                  # Inference orchestration layer
│   │   ├── decoder/
│   │   ├── embedding/
│   │   │   └── mod.rs
│   │   ├── lm_head/
│   │   ├── tokenizer/           # Tokenizer pipeline
│   │   │   ├── model/           # Vocab + merge encoding
│   │   │   │   ├── merge.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── vocab.rs
│   │   │   ├── normalizer/      # Text normalization stage
│   │   │   │   └── mod.rs
│   │   │   ├── pretokenizer/    # Split and byte-level stage
│   │   │   │   ├── byte_level.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── split.rs
│   │   │   └── mod.rs
│   │   ├── mod.rs
│   │   └── special_token.rs
│   ├── tensor/
│   │   └── mod.rs
│   ├── utils/                   # Shared low-level helpers
│   │   ├── mmap.rs
│   │   └── mod.rs
│   ├── error.rs
│   └── mod.rs
└── main.rs
```