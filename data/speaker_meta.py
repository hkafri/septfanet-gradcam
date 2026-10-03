"""Speaker gender metadata for LibriSpeech, parsed from the corpus's SPEAKERS.TXT.

OpenSLR serves SPEAKERS.TXT only as part of a subset archive (e.g.
test-clean.tar.gz / train-clean-100.tar.gz); fetching it directly 404s. This
project therefore reads the copy already extracted on disk inside the full
LibriSpeech corpus (the torchaudio fallback download used by fetch scripts).
"""

from pathlib import Path
from typing import Dict, Optional

# Default location of an extracted LibriSpeech corpus on this machine.
_DEFAULT_SPEAKERS_TXT = (
    Path(__file__).resolve().parents[2]
    / "3s_tse" / "data" / "librispeech_torchaudio_download" / "LibriSpeech" / "SPEAKERS.TXT"
)


def load_speaker_genders(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse SPEAKERS.TXT into {speaker_id: 'F'|'M'}.

    SPEAKERS.TXT is a pipe-separated table:
        ;ID  |SEX| SUBSET           |MINUTES| NAME
        14   | F | train-clean-360  | 25.03 | Kristin LeMoine
    Comment lines start with ';'. Covers every LibriSpeech speaker across all
    subsets, so a single file resolves gender for test-clean, dev-clean, and
    train-clean-100 alike.
    """
    path = Path(path) if path is not None else _DEFAULT_SPEAKERS_TXT
    if not path.exists():
        raise FileNotFoundError(
            f"SPEAKERS.TXT not found at {path}. Extract any LibriSpeech subset "
            f"archive (it ships SPEAKERS.TXT) and pass its path explicitly."
        )

    genders: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        speaker_id, sex = parts[0], parts[1].upper()
        if speaker_id.isdigit() and sex in ("F", "M"):
            genders[speaker_id] = sex
    if not genders:
        raise RuntimeError(f"No speaker genders parsed from {path}")
    return genders
