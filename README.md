# 🐋 my_little_deepseek

Toy implementation of **DeepSeek-R1-Distill-Qwen-1.5B** inference in pure Rust.

Origin model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## Overview

This project is a Rust implementation focused on a single fixed inference target: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B.

Core constraints:
- Pure Rust with std and libc only.
- Inference only, no training.
- Single model target with fixed architecture and config expectations.
- CPU only, single core, single thread for the current baseline.

For the module architecture and dependency flow, see [src/README.md](src/README.md).

## Current Status

- Model data loading for Unicode, exclusions, merges, vocab, and model.safetensors.
- Prompt token assembly with model specific special tokens.
- Tokenizer pipeline with normalizer, pretokenizer, and model/BPE encoding.

## Roadmap

- Implement embedding lookup and decoder forward pass.
- Implement LM head and token selection loop.
- Add runtime and latency measurement for inference steps.
- Optimize for speed on the CPU baseline.
- Port major matrix operations to CUDA.