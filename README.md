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


```
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

```
[] Initializing... done!
---------------------------------
[User]: What is the capital of South Korea? Be concise.

[Assistant]: 
<think>
Okay, so I need to figure out the capital of South Korea. I remember that South Korea is a country in South Asia, and it's known for its diverse culture and history. I think the capital is a major city, but I'm not exactly sure which one. Let me try to recall or think through this.

First, I know that the capital of South Korea is often referred to as "Seoul." I've heard that term before, especially in the context of South Korea's political system. Seoul is the capital, and it's also the administrative center. I think it's the most populous city, so that makes sense as the capital.

I should also consider the other major cities in South Korea. There's also Pusan, which is a city in the north, and Gwangju, which is in the south. But I'm pretty sure Seoul is the main one. I remember hearing that Seoul is the economic hub of South Korea, so that's another reason why it's the capital.

I wonder if there are any other names associated with Seoul. Maybe it's called "Korea" or "South Korea," but those are the country names, not the capital. So, the capital is definitely Seoul.

I should also think about the geography. Seoul is located in the Yangon Bay, which is a bay in the north of South Korea. It's a large city with a lot of history and culture, which supports why it's the capital. It's also known for its modern architecture and vibrant nightlife.

I don't think there are any other capitals in South Korea besides Seoul. The other cities are administrative centers, but Seoul is the administrative capital and the economic and political center. So, I'm pretty confident that Seoul is the capital.

To sum up, the capital of South Korea is Seoul, which is the administrative center and the economic hub. It's also the most populous city, making it the capital. I don't think there are any other capitals in South Korea besides Seoul.
</think>

The capital of South Korea is Seoul.<|end_of_sentence|>
[User]:  /exit
---------------------------------
Goodbye!
```