# models.__init__ — intentionally empty.

# The submodules under `omni_diffusion.models` must not be eagerly imported
# here: `dream` pulls in torch. Callers import `byte_tokenizer` or `x8d_export`
# (pure Python) or `dream.modeling_dream` (torch) explicitly so the byte-native
# core stays dependency-free.
