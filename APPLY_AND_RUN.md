# Minimal surgery patch for `feature/cough`

This patch does four practical things:

1. Adds a simple VAD / candidate-event detector based on RMS + 600-3000 Hz band energy + positive spectral flux.
2. Fixes the BiLSTM bug where `nn.Linear(...)` was created inside `forward()`.
3. Changes `src/infer.py` from one-shot clip prediction to sliding-window event detection.
4. Adds a separate `tools/plot_logmel_peaks.py` script for your advisor's "show Mel + high points" idea.

## Files to replace/add

- `config/config.yaml`
- `src/preprocessing/vad.py`  (new)
- `src/dataset.py`
- `src/models/bilstm.py`
- `src/train.py`
- `src/infer.py`
- `tools/plot_logmel_peaks.py` (new)

## Training

```bash
python -m src.train --config config/config.yaml
```

## Event-level inference on a long recording

```bash
python -m src.infer --model checkpoints/model.pt --wav path/to/long.wav --config config/config.yaml
```

Outputs:
- `infer_outputs/<wav_stem>_frames.csv`
- `infer_outputs/<wav_stem>_events.csv`

## Visual check for advisor

```bash
python tools/plot_logmel_peaks.py --wav path/to/example.wav --seconds 3 --do_denoise
```

This saves a PNG showing:
- waveform
- candidate VAD regions
- log-Mel spectrogram
- score curve and detected peaks
