from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np
from scipy.signal import find_peaks


def _safe_float(value: float) -> float:
    return float(np.round(float(value), 4))


def _merge_segments(segments: List[Dict[str, float]], max_gap: float = 0.25) -> List[Dict[str, float]]:
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start"] - prev["end"] <= max_gap:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())
    return merged


def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr = np.asarray(arr, dtype=float)
    span = np.max(arr) - np.min(arr)
    if span <= 1e-9:
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / span


def _estimate_vocal_phrases(y: np.ndarray, sr: int) -> List[Dict[str, float]]:
    # Non-silent intervals are candidate phrase zones.
    intervals = librosa.effects.split(y, top_db=28)
    if len(intervals) == 0:
        return []

    n_fft = 2048
    hop = 512
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    total_energy = np.sum(spec, axis=0) + 1e-9
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Vocal-heavy speech/singing band approximation.
    vocal_band = (freqs >= 300) & (freqs <= 3400)
    band_energy = np.sum(spec[vocal_band, :], axis=0)
    vocal_ratio = band_energy / total_energy

    rms = librosa.feature.rms(S=spec)[0]
    time_frames = librosa.frames_to_time(np.arange(len(vocal_ratio)), sr=sr, hop_length=hop)

    min_rms = np.percentile(rms, 40) if rms.size else 0.0
    phrases: List[Dict[str, float]] = []

    for start_sample, end_sample in intervals:
        start_t = start_sample / sr
        end_t = end_sample / sr
        dur = end_t - start_t
        if dur < 0.45 or dur > 16:
            continue

        idx = np.where((time_frames >= start_t) & (time_frames <= end_t))[0]
        if idx.size == 0:
            continue

        ratio = float(np.mean(vocal_ratio[idx]))
        rms_seg = float(np.mean(rms[idx]))

        if ratio >= 0.42 and rms_seg >= min_rms:
            phrases.append({"start": _safe_float(start_t), "end": _safe_float(end_t)})

    return _merge_segments(phrases)


def _estimate_sections(y: np.ndarray, sr: int, duration: float) -> List[Dict[str, float | str]]:
    if duration <= 0.0:
        return []

    # Build low-dimensional descriptor over short windows.
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=hop)
    feat = np.vstack([mfcc, rms[np.newaxis, :]]).T

    frame_times = librosa.frames_to_time(np.arange(feat.shape[0]), sr=sr, hop_length=hop)
    win = 4.0
    step = 2.0
    starts = np.arange(0, max(duration - win, 0) + 1e-6, step)
    if starts.size == 0:
        starts = np.array([0.0])

    window_vecs = []
    for s in starts:
        e = min(s + win, duration)
        idx = np.where((frame_times >= s) & (frame_times < e))[0]
        if idx.size == 0:
            window_vecs.append(np.zeros(feat.shape[1], dtype=float))
        else:
            window_vecs.append(np.mean(feat[idx], axis=0))

    window_vecs = np.asarray(window_vecs)
    boundaries = [0.0]

    if len(window_vecs) > 1:
        diffs = np.linalg.norm(np.diff(window_vecs, axis=0), axis=1)
        if np.any(diffs > 0):
            thr = np.percentile(diffs, 75)
            for i, d in enumerate(diffs, start=1):
                if d >= thr and (starts[i] - boundaries[-1]) >= 6.0:
                    boundaries.append(float(starts[i]))

    if duration - boundaries[-1] >= 3.0:
        boundaries.append(duration)
    elif len(boundaries) == 1:
        boundaries = [0.0, duration]
    else:
        boundaries[-1] = duration

    # Deduplicate/cleanup boundaries.
    clean = [boundaries[0]]
    for b in boundaries[1:]:
        if b - clean[-1] >= 2.0:
            clean.append(b)
        else:
            clean[-1] = b

    # Section labels: intro + verse/chorus by relative energy.
    sec_energy = []
    for i in range(len(clean) - 1):
        s, e = clean[i], clean[i + 1]
        idx = np.where((frame_times >= s) & (frame_times < e))[0]
        sec_energy.append(float(np.mean(rms[idx])) if idx.size else 0.0)

    median_energy = np.median(sec_energy) if sec_energy else 0.0
    sections: List[Dict[str, float | str]] = []

    for i in range(len(clean) - 1):
        s = _safe_float(clean[i])
        e = _safe_float(clean[i + 1])
        if i == 0 and s <= 0.05 and (e - s) <= 12.0:
            sec_type = "intro"
        else:
            sec_type = "chorus" if sec_energy[i] >= median_energy else "verse"
        sections.append({"start": s, "end": e, "type": sec_type})

    return sections


def analyze_audio(path: str) -> dict:
    """Analyze an audio file and return rhythmic + structural metadata for video planning."""
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if y.size == 0:
        return {
            "duration": 0.0,
            "bpm": 0.0,
            "beats": [],
            "downbeats": [],
            "bars": [],
            "vocalPhrases": [],
            "energyPeaks": [],
            "sections": [],
        }

    duration = librosa.get_duration(y=y, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist() if len(beat_frames) else []

    beats = [_safe_float(t) for t in beats]
    bpm = _safe_float(float(tempo)) if np.isfinite(tempo) else 0.0

    bars = beats[::4] if beats else []
    downbeats = bars.copy()

    # Energy curve from RMS + spectral flux, then peak-picking.
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    flux = onset_env
    # Align lengths for combination.
    n = min(len(rms), len(flux))
    rms_n = _normalize(rms[:n])
    flux_n = _normalize(flux[:n])
    energy_curve = 0.6 * rms_n + 0.4 * flux_n

    min_peak_distance_frames = max(1, int(0.8 * sr / hop))
    peaks, _ = find_peaks(
        energy_curve,
        distance=min_peak_distance_frames,
        prominence=0.2,
        height=np.percentile(energy_curve, 65) if len(energy_curve) else None,
    )
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop).tolist()
    energy_peaks = [_safe_float(t) for t in peak_times]

    vocal_phrases = _estimate_vocal_phrases(y=y, sr=sr)
    sections = _estimate_sections(y=y, sr=sr, duration=float(duration))

    return {
        "duration": _safe_float(duration),
        "bpm": bpm,
        "beats": beats,
        "downbeats": downbeats,
        "bars": bars,
        "vocalPhrases": vocal_phrases,
        "energyPeaks": energy_peaks,
        "sections": sections,
    }
