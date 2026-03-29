# Training Guide: Adding Nepali Support to Chatterbox TTS

This guide explains how to fine-tune the Chatterbox Multilingual model for Nepali (`ne`) on your macOS system (optimized for M2 Max 64GB).

## 1. Project Directory Structure

Organize your data like this for efficient training:

```text
chatterbox/
├── data/
│   ├── nepali/
│   │   ├── wavs/               <-- Put your audio here
│   │   │   ├── train_001.wav
│   │   │   └── ...
│   │   └── manifest.jsonl      <-- Index file of (audio, text)
├── src/
│   └── chatterbox/
│       └── train_nepali.py     <-- The training script
└── src/
    └── ...
```

## 2. Audio & Text Requirements

For high-quality Nepali support, aim for the following:

- **Audio format**: Mono WAV files.
- **Sample Rate**: 16kHz (The script automatically resamples if needed, but 16kHz is native).
- **Duration**: Each clip should be between **3 to 10 seconds**. Too long or too short can destabilize training.
- **Text**: Standard Nepali Devanagari script. Ensure the text matches exactly what is being said in the audio.
- **Quantity**: 
    - **Minimum**: 1 hour of audio (approx. 500-800 clips) for basic accent.
    - **Recommended**: 5-10 hours for professional quality.

## 3. Creating the Manifest File

The training script requires a `.jsonl` (JSON Lines) file where each line is a JSON object mapping audio to text.

**File path**: `data/nepali/manifest.jsonl`
**Content example**:
```json
{"audio_path": "data/nepali/wavs/train_001.wav", "text": "नमस्ते, सबैजनालाई कस्तो छ?"}
{"audio_path": "data/nepali/wavs/train_002.wav", "text": "यस अनुप्रयोगमा स्वागत छ।"}
```

## 4. Training on Mac (M2 Max)

Your system is powerful! We've optimized the script to use **MPS (Metal Performance Shaders)** which utilizes the Apple Silicon GPU.

### How to start training

Run the following command in your terminal while the `chatterbox_ne` conda environment is active.

```bash
conda activate chatterbox_ne
export PYTHONPATH=src

# Run from the root directory
python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/manifest.jsonl \
  --batch_size 16 \
  --epochs 50 \
  --lr 5e-5 \
  --save_every 10
```

### Why these settings?
- `--batch_size 16`: With 64GB Unified Memory, you can easily handle 16 or even 32. Larger batches improve stability.
- `--device mps`: This is what makes it fast on your Mac.
- `--epochs 50`: Fine-tuning usually takes 20-50 epochs depending on your dataset size.

## 5. What happens during training?
1. The script expands the model vocabulary to include the `[ne]` language tag.
2. It initializes the new text embedding from existing ones.
3. It takes your Nepali text and converts it to tokens.
4. It takes your audio and converts it to "Speech Tokens" (at 25Hz).
5. It teaches the Transformer model to predict these specific speech tokens when it sees the `[ne]` prefix.

## 6. How to check if it's working?

Once training is finished, it will save a final model: `t3_mtl_nepali_final.safetensors`.

To test your new model:
1. Update your code to point to the new `.safetensors` file.
2. Use `language_id="ne"` for generation.

```python
# Create a test script
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch

model = ChatterboxMultilingualTTS.from_pretrained("mps") # ensure your locally updated src is used
wav = model.generate("नमस्ते, अब म नेपाली बोल्न सक्छु।", language_id="ne")
```

## Troubleshooting TIPS
1. **Memory Pressure**: If your Mac slows down a lot during training, reduce `--batch_size` to 8.
2. **Loss not decreasing**: Decrease `--lr` to 1e-5.
3. **Artifacts in voice**: Ensure the audio is clean and has no background noise.
