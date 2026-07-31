# models.dream.__init__ — intentionally empty.

# The dream submodules are imported explicitly by callers:
#   - `byte_tokenizer` (pure Python, no torch) is imported directly
#   - `modeling_dream`, `configuration_dream`, `tokenization_dream`
#     (torch/transformers) are imported by `tools/finetune_*.py` and
#     `tools/inference.py`.
# Keeping this file empty preserves the dependency-free byte-native core.
