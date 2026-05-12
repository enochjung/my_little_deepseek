# config

## Purpose

Stores model configuration and provides parsing helpers for model asset files.

When `config` is set with the necessary files and parameters, it defines a fixed model architecture that can generate consistent inference outputs.

## Configuration Contents

The `config` module handles:

**Files (5 fixed-format files):**
- `unicode` - Unicode normalization data (from UnicodeData.txt)
- `composition_exclusions` - Exceptions to Unicode composition (from CompositionExclusions.txt)
- `vocab` - Token vocabulary mapping (from vocab.json)
- `merges` - Byte-pair encoding merge rules (from merges.json)
- `weights` - Model weights and architecture (from model.safetensors)

**Fields (2 configuration parameters):**
- `num_hidden_layers` - Number of transformer layers in the model
- `rms_norm_epsilon` - Epsilon value for RMS normalization

## Dependencies

**Outgoing:**
- Calls `storage::Host` to store parsed file data in host memory

**Incoming:**
- Called by `model` to access parsed configuration and provide initialization data

## Example Usage

```rust
// pseudo code
config = build_config(
    unicode_file,
    composition_exclusion_file,
    vocab_file,
    merge_file,
    weight_file,
    num_hidden_layers,
    rms_norm_epsilon,
)

model = build_model_from_config(config)
```

## Architecture Notes

- All 5 files must be present and in fixed formats (no format variation support planned)
- Configuration is immutable once the model is initialized
- File data is stored in `storage::Host` memory for efficient access by the model
