# DviTTS-Nano

[Weights on Hugging Face](https://huggingface.co/prakharkhatri/dvitts-nano) · [Demo](https://huggingface.co/spaces/prakharkhatri/DviTTS-demo) · [Android APK](https://github.com/codeprakhar25/dvitts-nano/releases/latest) · [Demo page](https://codeprakhar25.github.io/dvitts/)

Text-to-speech for English and six Indian languages that runs on a phone or a laptop CPU. No GPU, no PyTorch. One
folder of ONNX files plus this runner; the same files serve Android, desktop and servers through ONNX Runtime.

**Base model:** the 0.5B language model at the start of [DviTTS](https://huggingface.co/prakharkhatri/dvitts) is
[Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (FunAudioLLM / Alibaba,
Apache 2.0), a model that already knows how to produce speech. We retrained it
to emit [IndiCodec](https://huggingface.co/prakharkhatri/IndiCodec) ids and to speak six Indian languages.

This is the small sibling of [DviTTS](https://huggingface.co/prakharkhatri/dvitts): a 110M stage 1 distilled from it,
a one-step stage 2, and a quantized decoder. It speaks in 18 named preset voices. Cloning from your own recording
needs the full model.

## Key features

- **On-device:** ~351 MB of ONNX. Stage 1 8-bit, MeanFlow 4-bit, decoder 4/8-bit.
- **Laptops:** optional llama.cpp stage 1 on x86 (0.63x real time on a Ryzen 7 laptop vs 0.81x without it).
- **Languages:** English, Hindi, Marathi, Bengali, Punjabi, Gujarati, Odia.
- **18 presets:** 4 Hindi, 2 each other Indian language, 4 English. Labels, not the speakers' names.
- **Same runner on phone and laptop:** `python/dvitts_onnx.py` is the spec; `android/` is a line-by-line Kotlin port.
- **Watermark:** every clip carries an AudioSeal mark by default.

## Use (Python, any CPU)

Weights live on Hugging Face, not in this repo. The model folder already contains copies of the two Python files;
this repo is the source for those files and for the Android app.

```bash
pip install onnxruntime==1.32.0 tokenizers numpy num2words indic-numtowords
hf download prakharkhatri/dvitts-nano --local-dir dvitts-nano
python dvitts-nano/dvitts_onnx.py dvitts-nano "नमस्ते, आप कैसे हैं?" --voice ananya --out hello.wav
```

```python
import sys
sys.path.insert(0, "dvitts-nano")
from dvitts_onnx import DviTTS

tts = DviTTS("dvitts-nano", threads=4)
y, sr = tts.say("The meeting starts at 10:30, so please be on time.", voice="emma")
```

Laptops and desktops (x86 Windows / Linux), optional ~1.3x speed-up: stage 1 on llama.cpp. The runner uses it by
itself when `llama-cpp-python` is installed; without it, it prints one warning and stays on ONNX Runtime.

```bash
pip install --no-deps -r dvitts-nano/requirements-laptop.txt   # maintainer's prebuilt CPU wheel, sha256-pinned
pip install typing-extensions diskcache jinja2
```

ARM machines (Apple Silicon Macs, Windows on ARM, phones) stay on the ONNX path: the runner switches to llama.cpp by
itself only on x86. llama.cpp on an Apple Silicon Mac is opt-in and not yet measured:
`CMAKE_ARGS="-DGGML_CPU_REPACK=OFF" pip install llama-cpp-python==0.3.36 --no-binary llama-cpp-python --no-cache-dir`
(CPU repack gave wrong output on newer ARM chips in our tests), then `DviTTS(..., s1_gguf="s1_q8_0.gguf")`. Phone-only download:
`hf download prakharkhatri/dvitts-nano --local-dir dvitts-nano --exclude "*.gguf" --exclude "wm_fp32.onnx"`.

To run the copy in this repo against a downloaded model folder:

```bash
python python/dvitts_onnx.py /path/to/dvitts-nano "Hello." --voice emma --out hello.wav
```

## Android

`android/` is a Gradle project: `core` (pure Kotlin/JVM), `android` (AAR + onnxruntime-android, model download with
md5 check, playback), `demo` (voice picker, Speak, Share, bench). Open `android/` in Android Studio. Import the
ONNX folder from Hugging Face as a zip, or point the demo at a local copy.

On the same random draws the Kotlin port matches the Python spec (36,214 number words, 2,273 front-end texts, 2,277
tokenizer cases, 7 whole calls in four languages, two of them with the gap retry firing).

## Quality

Unseen speakers only. The phone build (these ONNX files and this runner, watermark and trim on) on AWS x86 CPUs,
4 threads. Whisper large-v3 for English WER, IndicConformer for Indian-language CER; each clip capped
at 100 %. Nano does not clone from a recording: for these benchmarks each unseen prompt was pre-encoded with the
IndiCodec encoder (part of DviTTS, not in the Nano folder), the same way the presets were made.

| Set | n | DviTTS-Nano | DviTTS (GPU) |
|---|---|---|---|
| Seed-TTS test-en, WER | 1,088 | 2.65 | 2.04 |
| CMU Arctic, WER | 350 | 2.74 | 1.45 |
| LibriTTS-R test, WER | 200 | 3.63 | 3.07 |
| IndicVoices Hindi, CER | 300 | 3.52 | 3.53 |
| FLEURS Hindi, CER | 100 | 5.16 | 4.52 |

FLEURS, 100 clips per language:

| Language | DviTTS-Nano CER | DviTTS (GPU) |
|---|---|---|
| Bengali | 3.54 | 2.69 |
| Odia | 3.75 | 3.11 |
| Marathi | 3.70 | 2.57 |
| Gujarati | 4.54 | 3.76 |
| Punjabi | 5.54 | 5.23 |

Clips that never ended: 0.

## Speed

| Device | Threads | Real-time factor | Peak memory |
|---|---|---|---|
| Phone, Snapdragon 8 Gen 3 (8-core CPU) | 4 | 0.61 | 877 MB |
| Mac, Apple M1 Pro | 4 | 0.27 | 973 MB |
| Laptop, Ryzen 7 (x86), llama.cpp stage 1\* | 8 | 0.63 | 770 MB |

Real-time factor is compute time per second of audio, watermark on; under 1.0 is faster than real time.
\* Without llama.cpp (ONNX Runtime only) the same laptop runs at 0.81.

## Limitations

- Preset voices only; no cloning in this folder (use [DviTTS](https://github.com/codeprakhar25/dvitts) for cloning).
- Cross-language (English text with an Indian preset, or the reverse) works, but sounds less like the picked voice.
- Indian languages other than Hindi learned from fewer speakers; Punjabi is the weakest.
- English text uses rule-based number reading (no NeMo), so unusual formats (times with am/pm, units) may be read
  plainly.
- No emotion or style control.
- No streaming; on a phone the first audio of one sentence takes about 3 s.

## Intended use and ethical considerations

For research and building speech products. Not for impersonation, fraud, or presenting synthetic speech as a real
person.

Preset voices are speakers from three public datasets (Rasa, IndicVoices-R, LibriTTS-R; CC BY 4.0), credited in
Acknowledgements and `voices/index.json`. Display names (Ananya, Emma, …) are labels, not the speakers' names.

**Do not use these voices to imitate the real speakers, or to make anyone believe they are hearing them.** Audio from
this model carries an AudioSeal watermark by default.

## Citation

If you use DviTTS-Nano, please cite:

```bibtex
@misc{khatri2026dvittsnano,
  title        = {DviTTS-Nano: On-Device Text-to-Speech for English and Six Indian Languages},
  author       = {Khatri, Prakhar},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/prakharkhatri/dvitts-nano}}
}
```

DviTTS-Nano builds on CosyVoice 3 (its base model) and AudioSeal (its watermark); please cite them too:

```bibtex
@article{du2025cosyvoice,
  title   = {CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author  = {Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal = {arXiv preprint arXiv:2505.17589},
  year    = {2025}
}

@article{sanroman2024proactive,
  title   = {Proactive Detection of Voice Cloning with Localized Watermarking},
  author  = {San Roman, Robin and Fernandez, Pierre and Elsahar, Hady and D{\'e}fossez, Alexandre and Furon, Teddy and Tran, Tuan},
  journal = {ICML},
  year    = {2024}
}
```

## Acknowledgements

- **Base model:** [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (FunAudioLLM, Alibaba; Apache 2.0) and its Qwen2 text tokenizer.
- **Watermark:** [AudioSeal](https://github.com/facebookresearch/audioseal) (Meta; MIT). **Silence trim:** [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).
- **Text front end:** [num2words](https://github.com/savoirfairelinux/num2words) (LGPL-2.1, used unmodified) and [indic-numtowords](https://github.com/AI4Bharat/indic-numtowords) (AI4Bharat; MIT).
- **Training data (CC BY 4.0):** Emilia-YODAS English subset (Amphion), LibriTTS and LibriTTS-R, VCTK, and from AI4Bharat Rasa, IndicVoices-R and IndicVoices (training split).
- **Preset voices:** speakers from Rasa, IndicVoices-R and LibriTTS-R (CC BY 4.0). They belong to those speakers and must not be used to impersonate them.
- The model folder on Hugging Face ships the MIT licence texts of AudioSeal, Silero VAD and indic-numtowords in `THIRD_PARTY_LICENSES`.

## Licence

Apache 2.0 (LICENSE).
