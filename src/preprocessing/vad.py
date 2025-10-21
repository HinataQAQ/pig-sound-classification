import numpy as np

def frame_signal(y, frame_len, hop_len):
    n = len(y)
    if n < frame_len:
        frames = np.expand_dims(np.pad(y, (0, frame_len - n)), axis=0)
        return frames
    num = 1 + (n - frame_len) // hop_len
    frames = np.stack([y[i*hop_len:i*hop_len+frame_len] for i in range(num)])
    return frames

def short_time_energy(frames):
    return np.sum(frames**2, axis=1) / frames.shape[1]

def zero_crossing_rate(frames):
    return np.mean((frames[:, 1:] * frames[:, :-1]) < 0, axis=1)

def double_threshold_vad(y, sr, frame_ms=25, hop_ms=10, energy_th=(0.1, 0.03), zcr_th=(0.1, 0.02)):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    frames = frame_signal(y, frame_len, hop_len)
    ste = short_time_energy(frames)
    zcr = zero_crossing_rate(frames)

    hi_e, lo_e = energy_th
    hi_z, lo_z = zcr_th

    voiced = np.zeros_like(ste, dtype=bool)
    state = False
    for i in range(len(ste)):
        if not state:
            if ste[i] > hi_e or zcr[i] > hi_z:
                state = True
        else:
            if ste[i] < lo_e and zcr[i] < lo_z:
                state = False
        voiced[i] = state

    segments = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            start = i
            while i < len(voiced) and voiced[i]:
                i += 1
            end = i
            segments.append((start*hop_len, min(len(y), start*hop_len + (end-start)*hop_len + frame_len)))
        i += 1
    return segments
