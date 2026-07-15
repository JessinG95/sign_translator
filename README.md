# Sign Language → Voice Translator

A real-time AI system that watches continuous sign language through a
webcam and speaks it aloud as natural, grammatical English sentences.

```
Webcam --> MediaPipe (hands+pose+face) --> Sentence Boundary Detector
        --> CNN+BiLSTM Sign Recognizer (gloss sequence)
        --> Seq2Seq Transformer (gloss -> natural sentence)
        --> Text-to-Speech (spoken aloud)
```

## 1. Install

```bash
pip install -r requirements.txt
```

Python 3.9–3.11 recommended (MediaPipe wheel availability varies by version).

## 2. Project layout

```
config.py                    all tunable settings in one place
main.py                      real-time webcam -> voice pipeline
utils/
  landmark_extractor.py      MediaPipe holistic -> fixed-length feature vector
  sentence_boundary.py       pause detection to segment continuous signing
  vocab.py                   tokenizer / vocabulary for gloss & sentence text
models/
  sign_recognizer.py         CNN + BiLSTM + attention pooling classifier
  seq2seq_transformer.py     Transformer encoder-decoder (gloss -> sentence)
data/
  collect_data.py            webcam tool to record labeled sign examples
  dataset.py                 PyTorch Dataset classes for both training stages
  gloss_sentence_pairs.tsv   TINY example file showing the training format
train/
  train_recognizer.py        trains the sign recognizer
  train_seq2seq.py           trains the gloss->sentence transformer
tts/
  speak.py                   non-blocking pyttsx3 / gTTS speech output
```

## 3. Collect training data

Record several samples per sign you want the system to recognize:

```bash
python -m data.collect_data --gloss HELLO --samples 30 --frames 24
python -m data.collect_data --gloss THANKYOU --samples 30 --frames 24
python -m data.collect_data --gloss WATER --samples 30 --frames 24
# ...repeat for every gloss/word in your vocabulary
```

This saves `.npy` landmark sequences under `dataset/<GLOSS>/`. Aim for
**30+ samples per gloss**, varying angle/speed/lighting a bit, for a
usable model. More data = better accuracy.

## 4. Train the sign recognizer (video → gloss)

```bash
python -m train.train_recognizer --epochs 50
```

Saves the best checkpoint to `checkpoints/sign_recognizer.pt` and the
class list to `checkpoints/gloss_vocab.json`.

## 5. Train the gloss → sentence translator

Build a TSV of `gloss_sequence <TAB> target_sentence` pairs — one line
per example, glosses lowercase and space-separated in the order you'd
sign them. `data/gloss_sentence_pairs.tsv` has 20 example rows showing
the format; **replace/expand it** with pairs that use the exact glosses
you trained the recognizer on. A few hundred pairs (even with repeated
sentence patterns) is enough to get grammatical output for a small
vocabulary; more is better.

```bash
python -m train.train_seq2seq --data data/gloss_sentence_pairs.tsv --epochs 100
```

Saves `checkpoints/gloss2sentence.pt`, `checkpoints/gloss_seq_vocab.json`,
and `checkpoints/sentence_vocab.json`.

## 6. Run live

```bash
python main.py
```

- Sign naturally and continuously.
- The system buffers frames while it sees hand motion, and cuts the
  "utterance" as soon as you pause (config: `PAUSE_FRAMES_THRESHOLD`).
- The buffered motion is scanned with a sliding window to produce a
  gloss sequence, which the transformer turns into a full sentence,
  which is then spoken aloud.
- Press `q` to quit.

## Tuning notes

- **Pause sensitivity**: `PAUSE_FRAMES_THRESHOLD` and `MOTION_EPSILON`
  in `config.py` control how quickly a pause is detected as "end of
  sentence." Lower = snappier but risks cutting off mid-sentence;
  higher = more patient but adds latency.
- **Window size**: `SLIDING_WINDOW_SIZE` / `SLIDING_WINDOW_STRIDE`
  control the gloss-recognition granularity. Smaller windows react
  faster to short signs; larger windows are more robust to noise.
- **TTS backend**: set `TTS_ENGINE = "pyttsx3"` (offline, low latency,
  default) or `"gtts"` (more natural voice, needs internet) in
  `config.py`.
- **GPU**: training and inference automatically use CUDA if available
  (`torch.cuda.is_available()`), otherwise fall back to CPU. The models
  are intentionally small so CPU real-time inference is workable.

## Design notes / honest limitations

- This ships **no pretrained weights or dataset** — sign language
  vocabularies, dialects, and regional variants differ, so you train it
  on your own recorded vocabulary. The training pipeline is intentionally
  simple to make that practical.
- The sentence-boundary heuristic (pause detection) is a lightweight,
  dependency-free approach. For production use, consider fusing it with
  the recognizer's confidence scores.
- The gloss recognizer uses sliding-window classification with
  consecutive-duplicate collapsing (CTC-style), which works well for a
  moderate vocabulary but isn't a full CTC/attention alignment model.
  If you outgrow it, swapping in `torch.nn.CTCLoss` on top of the same
  CNN+BiLSTM backbone is a natural upgrade path.
- `models/seq2seq_transformer.py` includes an optional `T5Seq2Seq`
  wrapper if you'd rather fine-tune a pretrained T5-small instead of the
  from-scratch Transformer — useful once you have a larger paired
  gloss/sentence dataset.
