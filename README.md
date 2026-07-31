# x8D-Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion

<font size=3><div align='center' > [[🌐 Homepage](https://bapxai.github.io/x8D-Omni-Diffusion)] [[🤗 Model Weight](https://huggingface.co/bapx/x8D-Omni-Diffusion)]</div></font>

This repository hosts code of **x8D-Omni-Diffusion**, the first any-to-any multimodal language model build on a mask-based discrete diffusion model. By modeling a joint distribution over discrete tokens of text, images, and speech, x8D-Omni-Diffusion exhibiting strong capability in multimodal comprehension and generation.
<p align="center">
    <img src="asset/teaser.png" width="95%" height="60%">
</p>


## Experimental Results

- **Evaluation on visual tasks.**
<p align="center">
    <img src="asset/visual_task.png" width="90%" height="60%">
</p>

- **Evaluation on speech tasks.**
<p align="center">
    <img src="asset/speech_task.png" width="90%" height="60%">
</p>

- **Qualitative Results.**
<p align="center">
    <img src="asset/qualitative_results.png" width="90%" height="60%">
</p>


## Requirements and Installation

### Prepare Environment
```
docker pull shenyunhang/pytorch:24.11-py3_2024-1224
git clone https://github.com/bapXai/x8D-Omni-Diffusion.git
cd x8D-Omni-Diffusion
git submodule update --init --recursive
pip install -r requirements_ds_gpu.txt
pip install -e .
```

### Prepare Pre-trained Weight

#### x8D-Omni-Diffusion

- Download the x8D-Omni-Diffusion from https://huggingface.co/bapx/x8D-Omni-Diffusion.
- Put it into '../models/x8D-Omni-Diffusion'

#### Audio Encoder and Audio Decoder

- Download the Audio Encoder from https://huggingface.co/THUDM/glm-4-voice-tokenizer.
- Put it into '../models/THUDM/glm-4-voice-tokenizer'

- Download the Audio Decoder from https://huggingface.co/THUDM/glm-4-voice-decoder.
- Put it into '../models/THUDM/glm-4-voice-decoder'

#### Images are raw bytes (no tokenizer)

The byte-native framework needs NO image tokenizer: images are raw 8-bit byte
streams (ids 0-255) placed between IMG_START/IMG_END on the diffusion canvas.
MagViT-v2 was removed — see `omni_diffusion/data/processor/image_processor.py`.

## Dataset import (byte-native)

Any Hugging Face dataset can be pulled straight into a byte-native x8D
container with no tokenizer and no `datasets`/torch dependency — it is fetched
via the datasets-server HTTP API and packed into an 8x8 DSpark-compressed
`.x8dds.gguf` stream. Text, image, and audio fields all arrive as raw 8-bit
bytes (ids 0-255); text is UTF-8, media is raw bytes, numerics are little-endian.

```
python3 tools/import_hf_dataset.py --dataset sarvamai/indic-diarbench --config Assamese --split train --length 50 --out ./datasets/
```


## SFT
#### 1. Data Format
Please convert SFT data into following data format.

**ASR Data Format**
```jsonc
{
  "messages": [
    {
      "content": "Convert the speech to text.\n<|audio|>",
      "role": "user"
    },
    {
      "content": "misery and horror were within that shadow and beyond it nothing that my spirit could look up to i stood for some moments as one stunned and then my manhood trained to some purpose by the usage of the sea",
      "role": "assistant"
    }
  ],
  "audios": [
    "datasets/fixie-ai/librispeech_asr/train.100.clean/4853-27670-0013.wav"
  ]
}
```

**TTS Data Format**
```jsonc
{
  "messages": [
    {
      "content": "Convert the text to speech.\nThe King of Tunis tore out the eyes of his father, Muley Assem, and his ambassadors have not been the less favourably received by the emperor.",
      "role": "user"
    },
    {
      "content": "<|audio|>",
      "role": "assistant"
    }
  ],
  "audios": [
    "datasets/mythicinfinity/libritts/train-clean-100/7178/34645/7178_34645_000012_000013.wav"
  ]
}
```

**T2I Data Format**
```jsonc
{
  "messages": [
    {
      "content": "Generate an image based on the provided text description.\nA group of 1920s girls at college immersed in their studies at a dark academia university.",
      "role": "user"
    },
    {
      "content": "<|image|>",
      "role": "assistant"
    }
  ],
  "images": [
    "datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB/00000001.jpg"
  ]
}
```

**VQA & Caption Data Format**
```jsonc
{
  "messages": [
    {
      "content": "How many triangles are there?\nChoices:\nA. 3\nB. 2\nC. 4\nD. 1\nE. 5\nAnswer with the option's letter from the given choices directly.\n<|image|>",
      "role": "user"
    },
    {
      "content": "A",
      "role": "assistant"
    }
  ],
  "images": [
    "datasets/lmms-lab/LLaVA-OneVision-Data/images/tqa(cauldron,llava_format)/16b35f931bd4fad5826f9e254521e7cb.png"
  ]
}
```

**Speech to Image Data Format**
```jsonc
{
  "messages": [
    {
      "content": "Please generate an image based on the input audio.",
      "role": "system"
    }
    {
      "content": "<|audio|>",
      "role": "user"
    },
    {
      "content": "<|image|>",
      "role": "assistant"
    }
  ],
  "audios": [
    "BLIP3o-Pretrain-JourneyDB/00000001.wav"
  ],
  "images": [
    "BLIP3o-Pretrain-JourneyDB/00000001.jpg"
  ]
}
```

**Spoken VQA Data Format**
```jsonc
{
  "messages": [
    {
      "content": "Please response the input audio based on the given image.",
      "role": "system"
    }
    {
      "content": "<|audio|>\n<|image|>",
      "role": "user"
    },
    {
      "content": "He is angling himself to better hit the ball with the racket.\n<|audio|>",
      "role": "assistant"
    }
  ],
  "audios": [
    "LLaVA-OneVision-Data-TTS/visual7w(cauldron,llava_format)/8_q.wav",
    "LLaVA-OneVision-Data-TTS/visual7w(cauldron,llava_format)/8_a.wav",
  ],
  "images": [
    "datasets/lmms-lab/LLaVA-OneVision-Data/images/visual7w(cauldron,llava_format)/c6c616d095b776d4fdfa68e7b900bff5.png"
  ]
}
```

#### 2. SFT
```
bash scripts/deepspeed/diffusion_dream/finetune.sh 3072 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `DATA_PATH` to your data config. 
- Set `MODEL_NAME_OR_PATH`, `AUDIO_TOKENIZER_PATH`, `AUDIO_MODEL_NAME_OR_PATH`, `IMAGE_TOKENIZER_PATH` to the path of pretrained models.
- Modify other variables as needed for your environment.


## Inference

Here we implement a simple script for inference.
It includes examples of speech-to-image, text-to-image, spoken visual question answering, visual question answering, TTS, and ASR tasks. 

- Set `model_name_or_path` to model weights.
- Set `output_dir` to output path.
- Set `image_tokenizer_path` to the path of the image tokenizer.
- Set `audio_tokenizer_path` to the path of the audio encoder.
- Set `flow_path` to the path of the audio decoder.

```
PYTHONPATH=$PYTHONPATH:third_party/GLM-4-Voice/ python tools/inference.py --model_name_or_path model_name_or_path --output_dir output --image_tokenizer_path image_tokenizer_path --audio_tokenizer_path audio_tokenizer_path --flow_path flow_path
```

## Evaluation

#### 1. Data Preparation

- Download test split of Librispeech from https://huggingface.co/datasets/openslr/librispeech_asr.
- Transform the data into jsonl file following [this format](#1-data-format).

- Download test split of Libritts from https://huggingface.co/datasets/mythicinfinity/libritts.
- Transform the data into jsonl file following [this format](#1-data-format).

- Download MME data and evaluation script from https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation.

#### 2. Evaluation

In the following command:
- Set `model_name_or_path` to model weights.
- Set `image_tokenizer_path` to the path of the image tokenizer.
- Set `audio_tokenizer_path` to the path of the audio encoder.
- Set `flow_path` to the path of the audio decoder.
- Set `JSON_PATH` to the path of transformed benchmark data.

Evaluate Librispeech
```
./scripts/deepspeed/evaluate_librispeech.sh model_name_or_path audio_tokenizer_path flow_path 
```

Evaluate LibriTTS
```
./scripts/deepspeed/evaluate_libritts.sh model_name_or_path audio_tokenizer_path flow_path
```

Evaluate MME 
- Set `mme_dir` to the path of downloaded MME data and evaluation script.
```
./scripts/deepspeed/evaluate_imageqa_mme.sh model_name_or_path image_tokenizer_path mme_dir
```

## Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@misc{x8domni2026,
  title={x8D-Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion},
  author={getwinharris},
  howpublished={https://bapx.in},
  year={2026}
}
```
