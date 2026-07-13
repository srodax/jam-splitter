#!/usr/bin/env python3.9
"""
Jam Session Splitter

Splits multi-stem Jamulus WAV recordings into individual MP3 tracks and a
chronological "bloopers" compilation using a two-stage cascade:

  Stage 1: Conservative silence detection (chunked RMS scan)
  Stage 2: Librosa spectral classification + onset-backtracked boundary refinement

Medleys (back-to-back songs without silence) are intentionally kept as single tracks.

Memory model: stems stay on disk; ffmpeg streams mix/render; analysis reads the
mono mix and per-segment windows via soundfile only.
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Aggression preset table: aggression -> (silence_thresh_dBFS, min_silence_s, min_track_s)
# ---------------------------------------------------------------------------
AGGRESSION_PRESETS = {
    1:  (-20.0, 3.0, 180),
    2:  (-25.0, 2.5, 165),
    3:  (-30.0, 2.0, 150),
    4:  (-32.0, 1.8, 135),
    5:  (-35.0, 1.5, 120),
    6:  (-38.0, 1.2, 105),
    7:  (-42.0, 1.0, 90),
    8:  (-45.0, 0.8, 75),
    9:  (-48.0, 0.6, 60),
    10: (-50.0, 0.5, 60),
}

# Musicality score weights for Phase 2 classification
MUSICALITY_WEIGHTS = {
    "centroid": 1.0,
    "bandwidth": 1.0,
    "rms": 0.5,
    "flux_variance": -0.5,
}
MUSICALITY_THRESHOLD = 0.45  # segments scoring >= this are candidate tracks
MUSICALITY_LOW_THRESHOLD = 0.30  # segments scoring < this are likely talking even if long
MIN_LIBROSA_DURATION_S = 2.0  # skip librosa analysis on segments shorter than this

# Onset backtracking: how far to look inward from a boundary (seconds)
ONSET_BACKTRACK_WINDOW = 2.0

# Phase 1 scan step — matches former pydub seek_step (ms)
SILENCE_SEEK_STEP_MS = 10

# 16-bit PCM full-scale (pydub default for WAV)
MAX_POSSIBLE_AMPLITUDE = 32768.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A contiguous non-silent region of audio."""
    start_ms: float
    end_ms: float
    duration_s: float
    is_track: bool = False
    musicality_score: Optional[float] = None
    onset_refined_start_ms: Optional[float] = None
    onset_refined_end_ms: Optional[float] = None


@dataclass
class Config:
    """Resolved configuration after merging CLI args + aggression presets."""
    stem_paths: List[str]
    dbs: List[float]
    silence_thresh: float
    min_silence_len: float  # seconds
    min_track_length: float  # seconds
    output_dir: str
    bitrate: str
    dry_run: bool
    no_librosa: bool
    aggression: int


@dataclass
class AnalysisContext:
    """Paths and metadata for the downsampled mono analysis mix."""
    mono_mix_path: str
    sample_rate: int
    duration_s: float


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split multi-stem Jamulus recordings into individual MP3 tracks "
                    "and a bloopers compilation."
    )
    parser.add_argument(
        "--stems", nargs="+", required=True,
        help="One or more WAV stem files (must all have identical duration)."
    )
    parser.add_argument(
        "--dbs", nargs="*", type=float, default=None,
        help="Per-stem dB level in the final mix (e.g. 0 -3 2). "
             "Unmatched stems get 0 dB. Must not exceed stem count."
    )
    parser.add_argument(
        "--aggression", type=int, default=5, choices=range(1, 11),
        help="Splitting aggression 1-10 (default: 5). Sets defaults for "
             "--silence-thresh, --min-silence-len, and --min-track-length."
    )
    parser.add_argument(
        "--silence-thresh", type=float, default=None,
        help="Override: dBFS threshold below which audio is considered silence "
             "(e.g. -40). More negative = quieter threshold."
    )
    parser.add_argument(
        "--min-silence-len", type=float, default=None,
        help="Override: minimum silence duration in seconds to qualify as a split "
             "boundary (e.g. 1.5)."
    )
    parser.add_argument(
        "--min-track-length", type=float, default=None,
        help="Override: minimum segment duration in seconds to qualify as a track "
             "(e.g. 120). Shorter segments become bloopers."
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Directory for output MP3 files (default: ./output)."
    )
    parser.add_argument(
        "--bitrate", default="192k",
        help="MP3 export bitrate (default: 192k)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect and classify segments only; do not render MP3s."
    )
    parser.add_argument(
        "--no-librosa", action="store_true",
        help="Skip Phase 2 (librosa spectral analysis). Use silence + min-length only."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Aggression / config resolution
# ---------------------------------------------------------------------------

def resolve_config(args: argparse.Namespace) -> Config:
    """Merge CLI args with aggression presets.  Explicit overrides take precedence."""
    preset = AGGRESSION_PRESETS[args.aggression]

    silence_thresh = args.silence_thresh if args.silence_thresh is not None else preset[0]
    min_silence_len = args.min_silence_len if args.min_silence_len is not None else preset[1]
    min_track_length = args.min_track_length if args.min_track_length is not None else preset[2]

    # Handle dB values
    stem_count = len(args.stems)
    if args.dbs is None:
        dbs = [0.0] * stem_count
    else:
        dbs = list(args.dbs)
        if len(dbs) > stem_count:
            print(f"Warning: {len(dbs)} dB values given for {stem_count} stems; "
                  f"truncating to {stem_count}.", file=sys.stderr)
            dbs = dbs[:stem_count]
        elif len(dbs) < stem_count:
            print(f"Note: {len(dbs)} dB values given for {stem_count} stems; "
                  f"remaining stems get 0 dB.", file=sys.stderr)
            dbs.extend([0.0] * (stem_count - len(dbs)))

    return Config(
        stem_paths=list(args.stems),
        dbs=dbs,
        silence_thresh=silence_thresh,
        min_silence_len=min_silence_len,
        min_track_length=min_track_length,
        output_dir=args.output_dir,
        bitrate=args.bitrate,
        dry_run=args.dry_run,
        no_librosa=args.no_librosa,
        aggression=args.aggression,
    )


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

class _ElapsedIndicator:
    """Context manager that prints a running elapsed-time indicator in a background thread."""

    def __init__(self, label: str):
        self._label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        start = time.monotonic()
        while not self._stop.wait(1.0):
            elapsed = time.monotonic() - start
            sys.stderr.write(f"\r  {self._label} (elapsed: {elapsed:.0f}s)  ")
            sys.stderr.flush()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        sys.stderr.write("\r" + " " * 80 + "\r")  # clear the line
        sys.stderr.flush()


def _format_size(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: List[str], label: str = "") -> bool:
    """Run ffmpeg, printing output on failure. Returns True on success."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nffmpeg error ({label}):", file=sys.stderr)
        print(e.stderr[-2000:], file=sys.stderr)
        return False


def _get_duration_seconds(path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _get_sample_rate(path: str) -> int:
    """Get audio sample rate via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return int(float(result.stdout.strip()))


# ---------------------------------------------------------------------------
# Stem validation & analysis mix (memory-safe)
# ---------------------------------------------------------------------------

def prepare_analysis_mix(config: Config) -> AnalysisContext:
    """
    Validate stems via ffprobe, stream a full-rate mono mix to a temp WAV via ffmpeg.

    The mix uses amix normalize=0 to match pydub overlay (additive, no divide).
    """
    if not config.stem_paths:
        print("Error: No stem files provided.", file=sys.stderr)
        sys.exit(1)

    durations = []
    sample_rate = None
    for i, path in enumerate(config.stem_paths):
        if not os.path.exists(path):
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

        dur_s = _get_duration_seconds(path)
        durations.append(dur_s)
        file_size = os.path.getsize(path)
        print(f"  Stem {i+1}/{len(config.stem_paths)}: {os.path.basename(path)} "
              f"({_format_size(file_size)}, {dur_s/60:.1f} min)")

        if sample_rate is None:
            sample_rate = _get_sample_rate(path)

    ref_dur_s = durations[0]
    for i, dur_s in enumerate(durations[1:], start=2):
        diff_ms = abs(dur_s - ref_dur_s) * 1000
        if diff_ms > 100:
            print(f"Error: Stem {i} duration ({dur_s:.1f}s) differs from "
                  f"stem 1 ({ref_dur_s:.1f}s) by {diff_ms/1000:.1f}s. "
                  f"All stems must have identical duration.", file=sys.stderr)
            sys.exit(1)
        elif diff_ms > 0:
            print(f"Note: Stem {i} differs from stem 1 by {diff_ms:.0f}ms — "
                  f"within tolerance, continuing.")

    print(f"Validated {len(config.stem_paths)} stem(s), duration: "
          f"{ref_dur_s/60:.1f} min ({ref_dur_s:.1f}s) @ {sample_rate} Hz")

    fd, mono_path = tempfile.mkstemp(suffix=".wav", prefix="jam_splitter_mono_")
    os.close(fd)

    inputs: List[str] = []
    for path in config.stem_paths:
        inputs.extend(["-i", path])

    filter_parts: List[str] = []
    for i, db in enumerate(config.dbs):
        chain = f"[{i}:a]"
        if db != 0:
            chain += f"volume={db}dB,"
        chain += "aformat=channel_layouts=mono"
        filter_parts.append(f"{chain}[m{i}]")

    mix_inputs = "".join(f"[m{i}]" for i in range(len(config.stem_paths)))
    n_stems = len(config.stem_paths)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={n_stems}:duration=longest:normalize=0[aout]"
    )
    filter_graph = ";".join(filter_parts)

    cmd = (
        ["ffmpeg", "-y"] + inputs +
        ["-filter_complex", filter_graph,
         "-map", "[aout]", "-ac", "1", "-c:a", "pcm_s16le", mono_path]
    )

    print("  Creating mono analysis mix via ffmpeg ...")
    with _ElapsedIndicator("ffmpeg mixdown"):
        if not _run_ffmpeg(cmd, "mono mixdown"):
            os.remove(mono_path)
            print("Error: ffmpeg mixdown failed.", file=sys.stderr)
            sys.exit(1)

    mono_size = os.path.getsize(mono_path)
    print(f"  Mono mix: {_format_size(mono_size)} (on disk, not loaded into RAM)")

    return AnalysisContext(
        mono_mix_path=mono_path,
        sample_rate=sample_rate,
        duration_s=ref_dur_s,
    )


# ---------------------------------------------------------------------------
# Phase 1: Conservative silence detection (chunked, pydub-compatible)
# ---------------------------------------------------------------------------

def _amp_thresh_from_dbfs(dbfs: float) -> float:
    """Convert dBFS threshold to absolute RMS amplitude (pydub semantics)."""
    return (10.0 ** (dbfs / 20.0)) * MAX_POSSIBLE_AMPLITUDE


def _chunk_rms_amplitude(samples: np.ndarray) -> float:
    """RMS amplitude on the same scale as pydub/audioop (0 .. 32768)."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return rms * MAX_POSSIBLE_AMPLITUDE


def _detect_silent_ranges_ms(
    duration_ms: float,
    silence_thresh_dbfs: float,
    min_silence_len_ms: int,
    seek_step_ms: int,
    rms_at_ms,
) -> List[Tuple[float, float]]:
    """
    Replicate pydub.silence.detect_silence + invert for nonsilent ranges.

    rms_at_ms(start_ms) -> RMS amplitude for [start_ms, start_ms+seek_step_ms).
    """
    amp_thresh = _amp_thresh_from_dbfs(silence_thresh_dbfs)
    silent_ranges: List[Tuple[float, float]] = []
    in_silence = False
    silence_start = 0.0

    pos_ms = 0.0
    while pos_ms < duration_ms:
        rms = rms_at_ms(pos_ms)
        is_silent = rms <= amp_thresh

        if is_silent:
            if not in_silence:
                silence_start = pos_ms
                in_silence = True
        elif in_silence:
            silence_end = pos_ms
            if silence_end - silence_start >= min_silence_len_ms:
                silent_ranges.append((silence_start, silence_end))
            in_silence = False

        pos_ms += seek_step_ms

    if in_silence and duration_ms - silence_start >= min_silence_len_ms:
        silent_ranges.append((silence_start, duration_ms))

    return silent_ranges


def _invert_ranges(
    silent_ranges: List[Tuple[float, float]],
    duration_ms: float,
) -> List[Tuple[float, float]]:
    """Convert silent ranges to non-silent ranges (pydub detect_nonsilent)."""
    nonsilent: List[Tuple[float, float]] = []
    prev_end = 0.0
    for start_ms, end_ms in silent_ranges:
        if start_ms > prev_end:
            nonsilent.append((prev_end, start_ms))
        prev_end = end_ms
    if prev_end < duration_ms:
        nonsilent.append((prev_end, duration_ms))
    return nonsilent


def detect_super_segments(
    analysis: AnalysisContext,
    silence_thresh: float,
    min_silence_len_s: float,
) -> List[Segment]:
    """
    Detect non-silent regions using chunked RMS scan over the mono mix file.
    Returns list of Segments with start_ms, end_ms, duration_s.
    """
    min_silence_ms = int(min_silence_len_s * 1000)
    seek_step_ms = SILENCE_SEEK_STEP_MS
    duration_ms = analysis.duration_s * 1000.0

    print(f"Phase 1: Scanning for silence gaps "
          f"(thresh={silence_thresh} dBFS, min_silence={min_silence_len_s}s) ...", flush=True)

    # Block reader: load only small windows from disk
    block_ms = 60_000  # 1 minute blocks
    block_frames_cache: dict = {}

    def rms_at_ms(pos_ms: float) -> float:
        block_start_ms = int(pos_ms // block_ms) * block_ms
        if block_start_ms not in block_frames_cache:
            with sf.SoundFile(analysis.mono_mix_path, "r") as wav:
                sr = wav.samplerate
                start_frame = int(block_start_ms / 1000.0 * sr)
                n_frames = int(block_ms / 1000.0 * sr)
                wav.seek(start_frame)
                data = wav.read(n_frames, dtype="float32", always_2d=True)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                block_frames_cache.clear()  # keep one block at a time
                block_frames_cache[block_start_ms] = (sr, data)

        sr, block = block_frames_cache[block_start_ms]
        rel_ms = pos_ms - block_start_ms
        start_frame = int(rel_ms / 1000.0 * sr)
        end_frame = int((rel_ms + seek_step_ms) / 1000.0 * sr)
        end_frame = max(end_frame, start_frame + 1)
        chunk = block[start_frame:end_frame]
        return _chunk_rms_amplitude(chunk)

    with _ElapsedIndicator("Scanning"):
        silent_ranges = _detect_silent_ranges_ms(
            duration_ms,
            silence_thresh,
            min_silence_ms,
            seek_step_ms,
            rms_at_ms,
        )
        nonsilent_ranges = _invert_ranges(silent_ranges, duration_ms)

    block_frames_cache.clear()
    gc.collect()

    if not nonsilent_ranges:
        print("Warning: No non-silent audio detected in the recording.", file=sys.stderr)
        return []

    segments = []
    for start_ms, end_ms in nonsilent_ranges:
        duration_s = (end_ms - start_ms) / 1000.0
        segments.append(Segment(
            start_ms=start_ms,
            end_ms=end_ms,
            duration_s=duration_s,
        ))

    print(f"Phase 1: Detected {len(segments)} super-segment(s) from silence analysis "
          f"(thresh={silence_thresh} dBFS, min_silence={min_silence_len_s}s)")
    return segments


# ---------------------------------------------------------------------------
# Phase 2: Librosa spectral analysis
# ---------------------------------------------------------------------------

def _try_import_librosa() -> bool:
    """Return True if librosa is importable, False otherwise."""
    try:
        import librosa  # noqa: F401
        return True
    except ImportError:
        return False


def _read_mono_segment(path: str, start_ms: float, end_ms: float) -> Tuple[np.ndarray, int]:
    """Read a mono float32 segment from a WAV file via soundfile seek."""
    with sf.SoundFile(path, "r") as wav:
        sr = wav.samplerate
        start_frame = int(start_ms / 1000.0 * sr)
        end_frame = int(end_ms / 1000.0 * sr)
        n_frames = max(end_frame - start_frame, 0)
        wav.seek(start_frame)
        data = wav.read(n_frames, dtype="float32", always_2d=True)
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data.reshape(-1)
        return data, sr


def librosa_classify_segments(
    segments: List[Segment],
    analysis: AnalysisContext,
) -> List[Segment]:
    """
    Run librosa spectral analysis on each segment:
      - Compute musicality score (centroid, bandwidth, RMS, flux variance)
      - Classify as track or blooper
      - Refine boundaries with onset backtracking

    Modifies segments in-place and returns the updated list.
    """
    total = len(segments)

    for i, seg in enumerate(segments):
        pct = (i + 1) / total * 100
        sys.stderr.write(f"\r  Phase 2: [{i+1}/{total}] {pct:.0f}%  "
                         f"({seg.duration_s:.1f}s @ {seg.start_ms/1000:.1f})  ")
        sys.stderr.flush()

        if seg.duration_s < MIN_LIBROSA_DURATION_S:
            seg.musicality_score = 0.0
            seg.is_track = False
            seg.onset_refined_start_ms = seg.start_ms
            seg.onset_refined_end_ms = seg.end_ms
            continue

        samples, sr = _read_mono_segment(
            analysis.mono_mix_path, seg.start_ms, seg.end_ms
        )

        musicality = _compute_musicality(samples, sr)
        seg.musicality_score = musicality
        seg.is_track = musicality >= MUSICALITY_THRESHOLD

        refined_start, refined_end = _refine_boundaries(samples, sr, seg)
        seg.onset_refined_start_ms = refined_start + seg.start_ms
        seg.onset_refined_end_ms = refined_end + seg.start_ms

        del samples

    sys.stderr.write("\r" + " " * 80 + "\r")
    sys.stderr.flush()
    print(f"Phase 2: Analyzed {total} segment(s)")
    gc.collect()

    return segments


def _compute_musicality(samples: np.ndarray, sr: int) -> float:
    """
    Compute a musicality score for a segment using spectral features.
    Higher = more likely to be full-band music (not speech/fragment).

    Returns a score in [0, 1] where:
      ~0.0-0.35 = likely speech / isolated noodling
      ~0.35-0.55 = ambiguous
      ~0.55-1.0 = likely full-band music
    """
    import librosa

    w = MUSICALITY_WEIGHTS

    min_n_fft = 256
    desired_n_fft = 2048
    n_fft = min(desired_n_fft, max(min_n_fft, len(samples) // 2))
    n_fft = 2 ** int(np.log2(n_fft))
    hop_length = n_fft // 4

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_fft=.* is too large")

        centroid = librosa.feature.spectral_centroid(
            y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
    centroid_mean = float(np.mean(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(
        y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth_mean = float(np.mean(bandwidth))

    rms = librosa.feature.rms(y=samples, frame_length=n_fft, hop_length=hop_length)
    rms_mean = float(np.mean(rms))

    onset_env = librosa.onset.onset_strength(
        y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    flux_var = float(np.var(onset_env))

    norm_centroid = _sigmoid_normalize(centroid_mean, center=2000.0, scale=800.0)
    norm_bandwidth = _sigmoid_normalize(bandwidth_mean, center=2500.0, scale=1000.0)
    norm_rms = _sigmoid_normalize(rms_mean, center=0.06, scale=0.03)
    norm_flux_var = _sigmoid_normalize(flux_var, center=3.0, scale=2.0)

    score = (
        w["centroid"] * norm_centroid +
        w["bandwidth"] * norm_bandwidth +
        w["rms"] * norm_rms +
        w["flux_variance"] * norm_flux_var
    )
    score = score / sum(abs(v) for v in w.values())

    return float(max(0.0, min(1.0, score)))


def _sigmoid_normalize(value: float, center: float, scale: float) -> float:
    """
    Map a value to [0, 1] using a sigmoid centered at `center` with steepness `scale`.
    Values << center map to ~0, values >> center map to ~1.
    """
    return 1.0 / (1.0 + np.exp(-(value - center) / scale))


def _refine_boundaries(
    samples: np.ndarray,
    sr: int,
    seg: Segment,
) -> Tuple[float, float]:
    """
    Use onset detection to refine segment boundaries inward.
    Returns (refined_start_ms_relative, refined_end_ms_relative)
    where values are milliseconds relative to the segment start.
    """
    import librosa

    if seg.duration_s < 3.0:
        return 0.0, seg.duration_s * 1000

    min_n_fft = 256
    desired_n_fft = 2048
    n_fft = min(desired_n_fft, max(min_n_fft, len(samples) // 2))
    n_fft = 2 ** int(np.log2(n_fft))
    hop_length = n_fft // 4

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_fft=.* is too large")

        onset_env = librosa.onset.onset_strength(
            y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
        )

    onset_env = onset_env / (onset_env.max() + 1e-10)

    frame_time_ms = hop_length / sr * 1000

    backtrack_window_frames = int(ONSET_BACKTRACK_WINDOW * sr / hop_length)
    start_search_end = min(backtrack_window_frames, len(onset_env))

    start_idx = 0
    if start_search_end > 0:
        start_search = onset_env[:start_search_end]
        peaks_start = _find_onset_peaks(start_search)
        if len(peaks_start) > 0:
            start_idx = peaks_start[0]

    end_search_start = max(0, len(onset_env) - backtrack_window_frames)
    end_search = onset_env[end_search_start:]

    end_idx = len(onset_env)
    if len(end_search) > 0:
        peaks_end = _find_onset_peaks(end_search)
        if len(peaks_end) > 0:
            end_idx = end_search_start + peaks_end[-1]

    refined_start_ms = start_idx * frame_time_ms
    refined_end_ms = end_idx * frame_time_ms

    refined_start_ms = max(0.0, refined_start_ms)
    refined_end_ms = min(seg.duration_s * 1000, refined_end_ms)

    if refined_end_ms - refined_start_ms < 500:
        refined_start_ms = 0.0
        refined_end_ms = seg.duration_s * 1000

    return refined_start_ms, refined_end_ms


def _find_onset_peaks(onset_env: np.ndarray) -> np.ndarray:
    """Find peaks in onset envelope above a simple threshold."""
    import librosa

    if len(onset_env) < 3:
        return np.array([], dtype=int)

    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=3,
        delta=0.1,
        wait=3,
    )
    return peaks


# ---------------------------------------------------------------------------
# Post-processing: apply min-track-length and classification
# ---------------------------------------------------------------------------

def finalize_segments(segments: List[Segment], min_track_length: float) -> Tuple[List[Segment], List[Segment]]:
    """
    Classify segments as tracks or bloopers.

    When musicality_score is None (Phase 2 skipped or unavailable):
      - duration < min_track_length → blooper
      - duration >= min_track_length → track

    When musicality_score is set (librosa Phase 2):
      - duration < min_track_length → blooper (regardless of score)
      - duration >= min_track_length AND score >= MUSICALITY_THRESHOLD → TRACK
      - duration >= min_track_length AND score < MUSICALITY_LOW_THRESHOLD → blooper
        (long but clearly non-musical, e.g. extended talking)
      - duration >= min_track_length AND score in [LOW, THRESHOLD) → TRACK
        (ambiguous but long enough — err on the side of keeping it)
    """
    tracks = []
    bloopers = []
    for seg in segments:
        if seg.duration_s < min_track_length:
            seg.is_track = False
            bloopers.append(seg)
        elif seg.musicality_score is None:
            seg.is_track = True
            tracks.append(seg)
        elif seg.musicality_score >= MUSICALITY_THRESHOLD:
            seg.is_track = True
            tracks.append(seg)
        elif seg.musicality_score < MUSICALITY_LOW_THRESHOLD:
            seg.is_track = False
            bloopers.append(seg)
        else:
            seg.is_track = True
            tracks.append(seg)

    print(f"\nFinal: {len(tracks)} track(s), {len(bloopers)} blooper(s) "
          f"(min_track_length={min_track_length}s)")
    return tracks, bloopers


# ---------------------------------------------------------------------------
# Rendering (ffmpeg streaming)
# ---------------------------------------------------------------------------

def render_outputs(
    stem_paths: List[str],
    dbs: List[float],
    tracks: List[Segment],
    bloopers: List[Segment],
    config: Config,
):
    """Render tracks and bloopers using ffmpeg (streaming, low memory)."""
    os.makedirs(config.output_dir, exist_ok=True)

    for i, seg in enumerate(tracks):
        out_path = os.path.join(config.output_dir, f"track_{i+1:02d}.mp3")
        start_ms = seg.onset_refined_start_ms if seg.onset_refined_start_ms is not None else seg.start_ms
        end_ms = seg.onset_refined_end_ms if seg.onset_refined_end_ms is not None else seg.end_ms
        start_s = start_ms / 1000.0
        dur_s = (end_ms - start_ms) / 1000.0

        label = f"Rendering track {i+1}/{len(tracks)} ({dur_s:.1f}s)"
        sys.stderr.write(f"\r  {label} ...")
        sys.stderr.flush()

        with _ElapsedIndicator("Encoding MP3"):
            if not _render_range_ffmpeg(stem_paths, dbs, start_s, dur_s, out_path, config.bitrate):
                print(f"Error: failed to render {out_path}", file=sys.stderr)
                sys.exit(1)

        sys.stderr.write(f"\r  {label} → {out_path}\n")
        sys.stderr.flush()

    if bloopers:
        out_path = os.path.join(config.output_dir, "bloopers.mp3")
        total_blooper_s = sum(s.duration_s for s in bloopers)
        print(f"Rendering bloopers: {len(bloopers)} segments ({total_blooper_s:.1f}s total)")

        temp_files: List[str] = []
        try:
            for i, seg in enumerate(bloopers):
                fd, tmp = tempfile.mkstemp(suffix=".wav", prefix=f"jam_splitter_blooper_{i:04d}_")
                os.close(fd)
                temp_files.append(tmp)
                start_s = seg.start_ms / 1000.0
                dur_s = seg.duration_s
                sys.stderr.write(f"\r  Extracting blooper {i+1}/{len(bloopers)} ...")
                sys.stderr.flush()
                if not _render_range_ffmpeg(stem_paths, dbs, start_s, dur_s, tmp, "pcm_s16le"):
                    print(f"Error: failed to extract blooper segment {i+1}", file=sys.stderr)
                    sys.exit(1)

            sys.stderr.write("\r" + " " * 80 + "\r")
            sys.stderr.flush()

            fd, concat_list = tempfile.mkstemp(suffix=".txt", prefix="jam_splitter_concat_")
            os.close(fd)
            try:
                with open(concat_list, "w", encoding="utf-8") as f:
                    for tf in temp_files:
                        escaped = tf.replace("'", "'\\''")
                        f.write(f"file '{escaped}'\n")

                print(f"  Concatenating {len(temp_files)} blooper chunks ...")
                with _ElapsedIndicator("Encoding bloopers MP3"):
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                         "-i", concat_list, "-c:a", "libmp3lame",
                         "-b:a", config.bitrate, out_path],
                        check=True, capture_output=True, text=True,
                    )
            finally:
                if os.path.exists(concat_list):
                    os.remove(concat_list)

            print(f"  → {out_path}")
        finally:
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)
    else:
        print("No bloopers to render.")


def _render_range_ffmpeg(
    stem_paths: List[str],
    dbs: List[float],
    start_s: float,
    dur_s: float,
    out_path: str,
    bitrate_or_codec: str,
) -> bool:
    """Render a time range from all stems via ffmpeg amix (normalize=0, pydub-compatible)."""
    inputs: List[str] = []
    for path in stem_paths:
        inputs.extend(["-ss", f"{start_s:.6f}", "-i", path])

    filter_parts: List[str] = []
    for i, db in enumerate(dbs):
        if db != 0:
            filter_parts.append(f"[{i}:a]volume={db}dB[a{i}]")
        else:
            filter_parts.append(f"[{i}:a]anull[a{i}]")

    mix_inputs = "".join(f"[a{i}]" for i in range(len(stem_paths)))
    n_stems = len(stem_paths)
    # Force stereo output to match pydub set_channels(2) on mono mixes
    filter_parts.append(
        f"{mix_inputs}amix=inputs={n_stems}:duration=longest:normalize=0,"
        f"aformat=channel_layouts=stereo[out]"
    )
    filter_graph = ";".join(filter_parts)

    cmd = (
        ["ffmpeg", "-y"] + inputs +
        ["-t", f"{dur_s:.6f}",
         "-filter_complex", filter_graph,
         "-map", "[out]", "-map_metadata", "-1"]
    )

    if out_path.endswith(".mp3"):
        cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate_or_codec])
    else:
        cmd.extend(["-c:a", bitrate_or_codec])

    cmd.append(out_path)
    return _run_ffmpeg(cmd, f"render {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------

def print_dry_run(
    segments: List[Segment],
    tracks: List[Segment],
    bloopers: List[Segment],
    config: Config,
    total_duration_s: float,
):
    """Print detailed segment analysis without rendering."""
    print("\n" + "=" * 80)
    print("DRY RUN — Segment Analysis")
    print("=" * 80)
    print(f"Stems:          {len(config.stem_paths)} file(s)")
    print(f"Duration:       {total_duration_s:.1f}s")
    print(f"Silence thresh: {config.silence_thresh} dBFS")
    print(f"Min silence:    {config.min_silence_len}s")
    print(f"Min track len:  {config.min_track_length}s")
    print(f"Librosa:        {'enabled' if not config.no_librosa else 'disabled'}")
    print("-" * 80)
    print(f"{'#':>3} {'Start':>8} {'End':>8} {'Dur':>7} {'Class':>7} {'Score':>7} {'Refined':>8}")
    print("-" * 80)

    for i, seg in enumerate(segments):
        classification = "TRACK" if seg.is_track else "blooper"
        score_str = f"{seg.musicality_score:.3f}" if seg.musicality_score is not None else "N/A"
        if seg.onset_refined_start_ms is not None:
            r_start = seg.onset_refined_start_ms / 1000
            r_end = seg.onset_refined_end_ms / 1000
            refined_str = f"{r_start:.1f}-{r_end:.1f}"
        else:
            refined_str = "none"
        print(f"{i+1:>3} {seg.start_ms/1000:>8.1f} {seg.end_ms/1000:>8.1f} "
              f"{seg.duration_s:>7.1f} {classification:>7} {score_str:>7} {refined_str:>8}")

    print("-" * 80)
    total_track_s = sum(s.duration_s for s in tracks)
    total_blooper_s = sum(s.duration_s for s in bloopers)
    total_s = total_track_s + total_blooper_s
    print(f"Tracks:   {len(tracks)}  ({total_track_s/60:.1f} min)")
    print(f"Bloopers: {len(bloopers)}  ({total_blooper_s/60:.1f} min)")
    print(f"Total:    {total_s/60:.1f} min")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = resolve_config(args)

    print("=" * 60)
    print("Jam Session Splitter")
    print("=" * 60)
    print(f"Aggression: {config.aggression} → "
          f"silence_thresh={config.silence_thresh} dBFS, "
          f"min_silence={config.min_silence_len}s, "
          f"min_track={config.min_track_length}s")

    librosa_available = _try_import_librosa()
    if config.no_librosa:
        print("Phase 2 (librosa) disabled via --no-librosa.")
    elif not librosa_available:
        print("Warning: librosa not found. Falling back to --no-librosa mode. "
              "Install with: pip install librosa", file=sys.stderr)
        config.no_librosa = True

    print("\n--- Stem validation & analysis mix ---")
    analysis = prepare_analysis_mix(config)
    gc.collect()

    try:
        print("\n--- Phase 1: Silence Detection ---")
        segments = detect_super_segments(
            analysis,
            config.silence_thresh,
            config.min_silence_len,
        )

        if not segments:
            print("No audio segments detected. Exiting.", file=sys.stderr)
            sys.exit(0)

        if not config.no_librosa:
            print("\n--- Phase 2: Librosa Spectral Analysis ---")
            segments = librosa_classify_segments(segments, analysis)

        tracks, bloopers = finalize_segments(segments, config.min_track_length)

        if config.dry_run:
            print_dry_run(segments, tracks, bloopers, config, analysis.duration_s)
        else:
            if os.path.exists(analysis.mono_mix_path):
                os.remove(analysis.mono_mix_path)
                analysis.mono_mix_path = ""
                gc.collect()

            print("\n--- Rendering ---")
            render_outputs(config.stem_paths, config.dbs, tracks, bloopers, config)
            print("\nDone.")
            if tracks:
                print(f"Tracks saved to: {os.path.abspath(config.output_dir)}/track_*.mp3")
            if bloopers:
                print(f"Bloopers saved to: {os.path.abspath(config.output_dir)}/bloopers.mp3")
    finally:
        if analysis.mono_mix_path and os.path.exists(analysis.mono_mix_path):
            os.remove(analysis.mono_mix_path)


if __name__ == "__main__":
    main()
