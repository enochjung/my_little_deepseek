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

## Module Notes

See the README in each subdirectory for the local details of that module.
