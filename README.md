# 🇳🇵 Chatterbox-TTS (Nepali Edition)

Fine-tuned text-to-speech for the Nepali language, based on the **Chatterbox-Multilingual-500M** architecture. This repository contains the custom training logic, bug fixes for the Devanagari script, and tools for high-fidelity Nepali voice cloning.

![Nepali TTS Preview](./Chatterbox-Multilingual.png)

## 🚀 Key Features
* **Full Devanagari Support**: Patched tokenizer and alignment analyzer to handle complex Nepali character clusters without cutting off.
* **Seamless Voice Cloning**: High-quality zero-shot cloning of Nepali voices using a 5-10 second reference clip.
* **Optimized for Mac**: Pre-configured for **Apple Silicon (MPS)** acceleration and memory-efficient training on M2/M3 chips.
* **Clean Inference**: Dedicated Gradio UI and test scripts for rapid experimentation.

## 📦 Model Checkpoints
To ensure repository performance, large training checkpoints are hosted on Hugging Face:
- **Repo Link**: [https://huggingface.co/officialuser/chatterbox-nepali](https://huggingface.co/officialuser/chatterbox-nepali)

| File | Purpose | Recommendation |
| :--- | :--- | :--- |
| `t3_nepali_epoch_20.pt` | **Base Nepali Checkpoint** | Use for **quick testing** or as a starting point to **train further**. |

*Place this file in the root folder of this repository after downloading.*

---

## 🏋️ Training / Generation
This project encourages you to build the final model weights yourself! When you reach the end of your training cycle (e.g., 50 epochs), the script will automatically consolidate your efforts and generate a high-performance **`t3_mtl_nepali_final.safetensors`** file.

### How to Resume Training:
```bash
export PYTHONPATH=src
python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/metadata.csv \
  --device mps \
  --batch_size 4 \
  --accum_steps 4 \
  --epochs 50 \
  --save_every 5 \
  --resume_t3_weights "t3_nepali_epoch_20.pt"
```

Once your `.safetensors` is generated, it will enable **Faster Inference** and significantly smaller file sizes.

---

## 🎙️ Inference & Implementation

### 🛡️ Quick Testing (Using Checkpoint)
For rapid audio generation and proof-of-concept testing using the current checkpoint, run the following:

```bash
# Make sure your environment is active
conda activate chatterbox_ne
export PYTHONPATH=src

# Run the test script
python3 test_nepali.py \
  --checkpoint "t3_nepali_epoch_20.pt" \
  --ref_audio "data/nepali/wavs/nep_sample.wav" \
  --text "इन्द्रेणी वा इन्द्रधनुष प्रकाश र रंगबाट उत्पन्न भएको यस्तो घटना हो जसमा रंगीन प्रकाशको एउटा अर्धवृत आकाशमा देखिन्छ। जब सूर्यको प्रकाश पृथ्वीको वायुमण्डलमा भएको पानीको थोपा माथि पर्छ, पानीको थोपाले प्रकाशलाई परावर्तन, आवर्तन र डिस्पर्सन गर्दछ।" \
  --output "my_first_nepali_test.wav"
```

### 🏮 Custom Implementation (Safetensors)
After you have successfully trained your model and generated a `.safetensors` file, you can integrate it into your own apps with this optimized code:

```python
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from safetensors.torch import load_file

model = ChatterboxMultilingualTTS.from_pretrained(device="mps")

# Loading your custom safetensors is faster than .pt checkpoints!
weights = load_file("t3_mtl_nepali_final.safetensors", device="mps")
cleaned_weights = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in weights.items()}
model.t3.load_state_dict(cleaned_weights, strict=False)

wav = model.generate("तपाईंलाई कस्तो छ?", language_id="ne", audio_prompt_path="reference.wav")
ta.save("final_output.wav", wav, model.sr)
```

## 🛠️ Critical Bug Fixes (Patched in this Fork)
This fork includes essential fixes for Devanagari that are **not available** upstream:
* **Causal Shift Fix**: Fixed the next-token prediction loss in `t3.py`.
* **Tokenizer Logic**: Prevented double-prepending of `[ne]` tags.
* **Alignment Safety**: Increased repetition tolerance in `alignment_stream_analyzer.py` to stop early audio cutoffs on long Nepali vowels.

## 📄 License & Credits
* Original architecture by **Resemble AI**.
* Fine-tuning and Nepali optimization by **officialuser**.
* Distributed under the **MIT License**.
