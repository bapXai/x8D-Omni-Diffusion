# coding=utf-8
"""Explicit transformers auto-registration for the Dream model classes.

Kept out of ``__init__.py`` so that importing the byte-native core
(``byte_tokenizer``, ``x8d_export``) never requires torch/transformers.
Torch-dependent callers (``tools/*.py``) call ``register_dream_classes()``
explicitly.
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_dream import DreamConfig
from .modeling_dream import DreamModel


def register_dream_classes() -> None:
    """Register DreamModel/DreamConfig with the transformers auto APIs."""
    AutoConfig.register("Dream", DreamConfig)
    AutoModelForCausalLM.register(DreamConfig, DreamModel)
    DreamConfig.register_for_auto_class()
    DreamModel.register_for_auto_class("AutoModelForCausalLM")
