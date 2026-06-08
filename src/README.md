# Source Tree

This directory contains the implementation split by responsibility.

## Core Modules

- `config` - knows how the model should work and parses the fixed configuration files before model initialization.
- `kernel` - low-level math only; it has no dependency on other modules and is called by `tensor`.
- `model` - builds the actual inference model from `config` and drives calculation through `tensor`.
- `session` - keeps chat state and the session embedding tensor, then asks `model` for the next-token result.
- `storage` - owns the physical memory used by file data and tensor data.
- `tensor` - manages tensor data, stores memory through `storage`, and updates values by calling `kernel`.

## Dependency Flow

- `config` uses `storage::Host` to keep parsed file data in memory.
- `model` uses `config` for initialization and uses `tensor` for weights and intermediate values.
- `session` keeps its embedding vector as a `tensor` and passes that tensor to `model`.
- `tensor` calls `storage` for allocation and raw pointers, then calls `kernel` for actual computation.
- `kernel` performs primitive-pointer computation and does not call other project modules.

## Quick Flow

`config` -> `model` -> `session` -> `tensor` -> `storage` / `kernel`


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