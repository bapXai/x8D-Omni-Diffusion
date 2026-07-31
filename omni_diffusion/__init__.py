# The top-level package must stay dependency-free: importing `omni_diffusion`
# should never pull in torch/datasets/etc. Submodules (`.models.dream`,
# `.data`, `.x8d_export`, ...) are imported explicitly by callers so that the
# byte-native core (`byte_tokenizer`, `x8d_export`) remains pure Python.
