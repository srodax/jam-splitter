# Jam Session Splitter

Splits multi-stem Jamulus WAV recordings into individual MP3 tracks and a chronological "bloopers" compilation.

## How It Works

**Two-stage cascade algorithm:**

1. **Phase 1 — Conservative Silence Detection** (`pydub.silence`): Splits on clear, unambiguous silences into "super-segments"
2. **Phase 2 — Librosa Spectral Analysis**: Classifies each super-segment as track or blooper based on spectral richness (centroid, bandwidth, RMS, flux), and refines cut boundaries with onset backtracking

Medleys (back-to-back songs without silence) are intentionally kept as single tracks.

## Requirements

- Python 3.8+
- ffmpeg (system install: `brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux)

```bash
pip install -r requirements.txt
```

## Usage

### Basic

```bash
python jam-splitter.py \
  --stems guitar.wav bass_drums.wav
```

### With per-stem balance control

`--dbs` sets **relative** levels between stems. The tool automatically measures
the summed mix and applies a global gain so the full session stays near **-1 dBFS**
(anti-clip). Relative offsets are preserved.

```bash
python jam-splitter.py \
  --stems guitar.wav bass_drums.wav vocals.wav \
  --dbs 0 -3 2 \
  --output-dir ./my_session
```

Without `--dbs`, all stems start at 0 dB relative balance; global gain is still
computed from the actual mix peak (not a simple `-20*log10(N)` default).

### Tuning sensitivity

```bash
# Aggressive splitting (more, shorter tracks)
python jam-splitter.py --stems file1.wav file2.wav --aggression 8

# Conservative (fewer, longer tracks)
python jam-splitter.py --stems file1.wav file2.wav --aggression 2

# Aggression preset + fine-tune one parameter
python jam-splitter.py --stems file1.wav file2.wav --aggression 5 --min-track-length 90
```

### Dry run (detect only, no rendering)

```bash
python jam-splitter.py --stems file1.wav file2.wav --dry-run
```

### Skip librosa (silence-only mode)

```bash
python jam-splitter.py --stems file1.wav file2.wav --no-librosa
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--stems` | (required) | One or more WAV files (must have identical duration) |
| `--dbs` | all 0 (relative) | Per-stem relative dB balance; global anti-clip gain applied automatically |
| `--aggression` | 5 | 1-10 splitting aggression preset |
| `--silence-thresh` | from aggression | Override: dBFS silence threshold |
| `--min-silence-len` | from aggression | Override: min silence seconds for a split |
| `--min-track-length` | from aggression | Override: min seconds to qualify as track |
| `--output-dir` | ./output | Output directory for MP3s |
| `--bitrate` | 192k | MP3 export bitrate |
| `--dry-run` | false | Detect and classify only, no rendering |
| `--no-librosa` | false | Skip Phase 2, use silence + min-length only |

## Aggression Presets

| Aggression | Silence Thresh | Min Silence | Min Track | Behavior |
|---|---|---|---|---|
| 1 | -20 dBFS | 3.0s | 180s | Only dead silence; very long tracks |
| 3 | -30 dBFS | 2.0s | 150s | Conservative |
| **5** | **-35 dBFS** | **1.5s** | **120s** | **Default — balanced** |
| 7 | -42 dBFS | 1.0s | 90s | Aggressive |
| 10 | -50 dBFS | 0.5s | 60s | Any quiet moment; short segments |

Explicit `--silence-thresh`, `--min-silence-len`, or `--min-track-length` override
the aggression-derived value for that parameter only.

## Output

- `output/track_01.mp3`, `output/track_02.mp3`, ... — detected songs
- `output/bloopers.mp3` — all inter-song fragments concatenated chronologically

Total output duration = total input duration (no audio is discarded).
