# 🇳🇵 Chatterbox-TTS (Nepali Edition)

Fine-tuned text-to-speech for the Nepali language, based on the **Chatterbox-Multilingual-500M** architecture. This repository contains the custom training logic, bug fixes for the Devanagari script, and tools for high-fidelity Nepali voice cloning.

![Nepali TTS Preview](./Chatterbox-Multilingual.png)

## 🚀 Key Features
* **Full Devanagari Support**: Patched tokenizer and alignment analyzer to handle complex Nepali character clusters without cutting off.
* **Seamless Voice Cloning**: High-quality zero-shot cloning of Nepali voices using a 5-10 second reference clip.
* **Optimized for Mac**: Pre-configured for **Apple Silicon (MPS)** acceleration and memory-efficient training on M2/M3 chips.
* **Clean Inference**: Dedicated Gradio UI and test scripts for rapid experimentation.

## 📦 Model Files
Due to size restrictions (2.1 GB), the fine-tuned model weights are hosted on Hugging Face:
- **Final Model (`.safetensors`)**: [your-huggingface-url-here/t3_mtl_nepali_final.safetensors]
- **Resume Checkpoint (`.pt`)**: [your-huggingface-url-here/t3_nepali_epoch_45.pt]

*Place these files in the root of the repository to get started.*

## ⚙️ Installation
```bash
conda create -n chatterbox_ne python=3.11
conda activate chatterbox_ne

# Clone and install dependencies
git clone https://github.com/officialuser/chatterbox-nepali.git
cd chatterbox-nepali
pip install -e .
```

## 🎙️ Inference & Usage

### 1. Web UI (Gradio)
Launch a graphical interface for easy generation:
```bash
python3 gradio_nepali.py
```

### 2. Command Line Test
Quickly test a specific checkpoint on a long sentence:
```bash
python3 test_nepali.py \
  --checkpoint "t3_mtl_nepali_final.safetensors" \
  --ref_audio "path/to/nepali_ref.wav" \
  --text "तपाईंलाई कस्तो छ? यो मेरो नयाँ नेपाली एआई मोडल हो।" \
  --output "output.wav"
```

## 🏋️ Training / Fine-tuning
If you want to continue training the model or fine-tune it on your own dataset:

1. **Prepare Data**: Place your `.wav` files and a `metadata.csv` (format: `file|text`) in `data/nepali/`.
2. **Launch Training**:
```bash
export PYTHONPATH=src
python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/metadata.csv \
  --device mps \
  --batch_size 4 \
  --accum_steps 4 \
  --epochs 50 \
  --save_every 5 \
  --resume_t3_weights "t3_nepali_epoch_45.pt"
```

## 🛠️ Important Fixes
This fork includes critical fixes for the Nepali language that are **not** present in the upstream repo:
* **Causal Shift Fix**: Corrected the autoregressive loss function in `t3.py` to prevent "garbage" output during fine-tuning.
* **Tokenizer Fix**: Prevented double-stacking of language tags (`[ne]`) when processing Nepali strings.
* **Alignment Safety Fix**: Relaxed the repetition analyzer in `alignment_stream_analyzer.py` from 2 tokens (too strict for Nepali vowels) to 15 tokens (~600ms) to prevent early audio cutoffs.

## 📄 License & Credits
* Original architecture by **Resemble AI**.
* Fine-tuning and Nepali optimization by **officialuser**.
* Distributed under the **MIT License**.
