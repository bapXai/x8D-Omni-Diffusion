# Config Mapping: DiffusionGemma 26B-A4B → x8D-Omni-Diffusion (byte-native)

**Date:** 2026-07-31
**Source:** `google/diffusiongemma-26B-A4B-it` `config.json`, `tokenizer_config.json`,
`preprocessor_config.json`, vLLM blog.

## 1. Pipeline-level mapping

| DiffusionGemma | x8D-Omni-Diffusion (ours) |
|---|---|
| `DiffusionGemmaPipeline` (diffusers) | `tools/finetune_dream_v4_51_3.py` + `omni_diffusion/models/dream/generation_utils.py` |
| `DiffusionGemmaForBlockDiffusion` (transformers) | `DreamModel` (`modeling_dream.py`) |
| `Gemma4Processor` (vision patch soft tokens) | `omni_diffusion/processor/...` (image/video/patch/audio) |
| `BlockRefinementScheduler` (diffusers) | `generation_utils.py` `_sample()` `alg in {entropy, maskgit, origin}` |
| `tokenizer_config.json` GemmaTokenizer | `tokenization_dream.py` → **replaced by byte-native** |

Key difference: DiffusionGemma's *text* still goes through a **262144-token BPE tokenizer**
(Gemma tokenizer with specials injected). Ours must go through **raw bytes only**.

## 2. Special-token ID mapping

### 2.1 DiffusionGemma (from tokenizer_config.json, actual string specials)

| role | token string | note |
|---|---|---|
| bos | `<bos>` | id 2 |
| eos | `<eos>` | ids [1, 106] (multi-stop) |
| pad | `<pad>` | id 0 |
| unk | `<unk>` | — |
| mask | `<mask>` | diffusion MASK state |
| boi | `<\|image>` | id 255999 begin-of-image |
| eoi | `<image\|>` | id 258882 end-of-image |
| image | `<\|image\|>` | id 258880 image-patch placeholder |
| boa | `<\|audio>` | begin-of-audio (tokenizer only; no encoder) |
| eoa | `<audio\|>` | end-of-audio |
| think | `<\|think\|>` | thinking mode |
| soc/eoc | `<\|channel>` / `<channel\|>` | reasoning channel delimiters |
| sot/eot | `<\|turn>` / `<turn\|>` | turn delimiters |
| stc/etc | `<\|tool_call>` / `<tool_call\|>` | tool call |
| std/etd | `<\|tool>` / `<tool\|>` | tool start/end |
| str/etr | `<\|tool_response>` / `<tool_response\|>` | tool response |
| escape | `<\|"\|>` | escape |
| video | `<\|video\|>` | extra special |

### 2.2 x8D byte-native (the ONLY legal set, vocab = 264)

| id | state | maps from |
|---|---|---|
| 0–255 | raw bytes | any text/image/audio/code/binary — `list(data_bytes)` |
| 256 | MASK | DiffusionGemma `<mask>` / DREAM mask_token_id |
| 257 | PAD | DiffusionGemma `<pad>` / DREAM pad_token_id 151643 |
| 258 | BOS | DiffusionGemma `<bos>` |
| 259 | EOS | DiffusionGemma `<eos>` (single id; multi-stop unsupported — byte-stream has one terminator) |
| 260 | IMG_START | DiffusionGemma boi 255999 |
| 261 | IMG_END | DiffusionGemma eoi 258882 |
| 262 | AUD_START | DiffusionGemma boa (we actually implement audio!) |
| 263 | AUD_END | DiffusionGemma eoa |

**Deliberately dropped:** unk (bytes are complete, nothing is unknown), think/turn/
tool/channel/escape tokens (byte-native text carries `<|think|>` etc. as raw UTF-8 bytes
0–255 — no vocab pollution; specials reserved strictly for modality boundaries).

## 3. Config-field mapping (configuration_dream.py)

### 3.1 Keep as-is (already correct for diffusion)

| DiffusionGemma | DreamConfig (current) |
|---|---|
| `use_bidirectional_attention: "vision"` | n/a — DREAM is bidirectional over canvas by design |
| `tie_word_embeddings: true` | `false` → **flip to true** (byte vocab tied embed/lm_head, standard for byte LMs) |
| `final_logit_softcapping: 30` | **add** `final_logit_softcap=30.0` |
| `hidden_activation: gelu_pytorch_tanh` | `silu` → keep `silu` (DREAM uses SwiGLU; not adopting Gemma gelu) |

### 3.2 Adopt (new fields for byte-native + DiffusionGemma insights)

| new field | value | reason |
|---|---|---|
| `vocab_size` | **264** (was 176264) | byte law, issue #2 |
| `mask_token_id` | **256** (was 151666) | byte law |
| `pad_token_id` | **257** (was 151643) | byte law |
| `bos_token_id` | 258 | byte law |
| `eos_token_id` | 259 | byte law |
| `img_start_token_id` | 260 | byte law |
| `img_end_token_id` | 261 | byte law |
| `aud_start_token_id` | 262 | byte law |
| `aud_end_token_id` | 263 | byte law |
| `canvas_length` | 256 | DiffusionGemma §3.3 |
| `max_denoising_steps` | 48 | DiffusionGemma canvas budget |
| `diffusion_sampler` | `"entropy_bound"` | DiffusionGemma §3.5 |
| `diffusion_entropy_bound` | 0.1 | vLLM override |
| `self_conditioning` | true | DiffusionGemma §3.2 |
| `self_conditioning_ffn` | small MLP | softmax×embed → FFNN → add |
| `num_global_key_value_heads` | 2 (was: k/v heads 4) | DiffusionGemma global-head trick → KDA issue #7 |
| `sliding_window` | 1024 (was 4096/None) | DiffusionGemma sliding layers |
| `layer_types` | `sliding/full` pattern 6:1 → **KDA 3:1** for 28 layers: 14 sliding + 9 full + 5 KDA-fused | issue #7 |

### 3.3 Target config_dream_resume.json (byte-native v0)

```json
{
  "_name_or_path": "bapx/x8D-Omni-Diffusion",
  "architectures": ["DreamModel"],
  "auto_map": {
    "AutoConfig": "configuration_dream.DreamConfig",
    "AutoModel": "modeling_dream.DreamModel"
  },
  "bos_token_id": 258,
  "eos_token_id": 259,
  "mask_token_id": 256,
  "pad_token_id": 257,
  "img_start_token_id": 260,
  "img_end_token_id": 261,
  "aud_start_token_id": 262,
  "aud_end_token_id": 263,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "num_global_key_value_heads": 2,
  "max_position_embeddings": 131072,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "final_logit_softcap": 30.0,
  "sliding_window": 1024,
  "canvas_length": 256,
  "max_denoising_steps": 48,
  "diffusion_sampler": "entropy_bound",
  "diffusion_entropy_bound": 0.1,
  "self_conditioning": true,
  "tie_word_embeddings": true,
  "use_cache": true,
  "torch_dtype": "bfloat16",
  "vocab_size": 264
}
```

## 4. What this replaces in the codebase

| file | change |
|---|---|
| `configuration_dream.py` | new fields above; `vocab_size=264`, ids 256–263; rope validation for sliding/full |
| `tokenization_dream.py` | **delete BPE path** (`vocab.json`/`merges.txt`) → `byte_tokenizer.py` |
| `omni_diffusion/tokenizer.py` `update_tokenizer` | no longer adds 16 string tokens; specials are fixed ids 260–263 |
| `constants.py` | keep string constants for *interface* templates only; IDs come from `byte_tokenizer` |
| `dataset_qwen2.py` | `special_token_id=151643` → `257`; byte-encode text |
| `modeling_dream.py` | embed + `lm_head` → 264; add self-conditioning hook; add `final_logit_softcap`; optional KDA layers |
| `generation_utils.py` | `alg="entropy_bound"`, adaptive stopping, byte re-noise |

## 5. How to test our framework (no GPU needed for shape tests)

Local env today: **no torch/transformers installed**, no HF token, no `tests/` dir.

1. **Install minimal stack** (pure CPU, small):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install transformers tokenizers
   ```
2. **Unit tests to add** in `tests/`:
   - `test_byte_tokenizer.py`:
     - `tokenize(b"hello")` == `[104,101,108,108,111]` (no vocab lookup)
     - round-trip `decode(tokenize(x)) == x` for all 256 bytes
     - vocab_size == 264; special ids 256–263
     - no `vocab.json`/`merges.txt` referenced anywhere
   - `test_config.py`:
     - `DreamConfig(vocab_size=264)`; mask=256, pad=257, bos=258, eos=259
     - tie_word_embeddings=True
     - `len(layer_types)==28`, exactly 5 KDA-fused layers, 14 sliding, 9 full
   - `test_model_shapes.py`:
     - tiny DreamModel (2 layers, hidden 64) fwd on `[1, 256]` canvas → `[1, 256, 264]`
     - embed weight shape `[264, 64]`, lm_head tied
     - self-conditioning adds FFN path; logits finite after 2 denoise steps
   - `test_generation_utils.py`:
     - `_sample(alg="entropy_bound")` returns ids in 0–255 for non-special slots
     - re-noise budget: accepted positions have entropy sum ≤ 0.1
     - adaptive stop fires on stable top-1
   - `test_data.py`: `dataset_qwen2` forward_process produces mask ids 256, not 151643
3. **CI smoke** (fast): `python -m pytest tests/ -v` — run locally after venv install.
4. **Full train/generate** needs GPU + real weights — later milestone.

## 6. Open questions before coding

1. Keep `hidden_size 3584/28 layers` (DREAM 7B shape) or shrink to DiffusionGemma
   `2816/30 layers`? Suggest keep DREAM 7B shape; only add byte vocab + new sampling.
2. `tie_word_embeddings` flip requires re-init from scratch (no pretrained byte embed
   exists) — confirm OK to train byte-native from scratch on x8Dsub-byte 0.001 scale.
3. KDA (issue #7) vs simple sliding/full pattern — implement plain sliding/full first,
   KDA as phase 2.
