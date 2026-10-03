"""Extend the LibriSpeech pool with NEW speakers from train-clean-100.

Streams the `train.100` split, collecting `utterances_per_speaker` utterances
for each of `num_speakers` NEW speakers (excluding every speaker already in
data/librispeech_samples/ and data/librispeech_holdout/), and writes each
utterance to disk as soon as it arrives so progress is visible and the run is
resumable. Utterances that fail to decode are skipped.
"""

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import soundfile as sf

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

from data.speaker_meta import load_speaker_genders

SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
PARQUET_DIR = gradcam_root / "data" / "librispeech_trainclean100_parquet"
TRAIN_CLEAN_100_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"


def _download_parquet_shards(shard_names):
    """Download specific train.100 parquet shards directly from the HF mirror."""
    import requests
    base = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/"
    paths = []
    for name in shard_names:
        dest = PARQUET_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_suffix(".parquet.part")
            with requests.get(base + name, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                with open(tmp, "wb") as out:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            out.write(chunk)
                            got += len(chunk)
                print(f"    ... downloaded {got/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
            tmp.replace(dest)
        paths.append(dest)
    return paths


def existing_speaker_ids():
    ids = set()
    for d in (SAMPLES_DIR, HOLDOUT_DIR):
        if d.exists():
            for f in d.glob("*.flac"):
                ids.add(f.name.split("-")[0])
    return ids


def _count_for(sid):
    return len(list(FULLSCALE_DIR.glob(f"{sid}-*.flac")))


def fetch_new_speakers(num_speakers, utterances_per_speaker=2, speakers_txt=None):
    from datasets import Audio, load_dataset

    existing = existing_speaker_ids()
    print(f"[*] Existing pool speakers to exclude: {len(existing)}", flush=True)
    load_speaker_genders(speakers_txt)  # validate gender metadata is resolvable up front

    FULLSCALE_DIR.mkdir(parents=True, exist_ok=True)
    prior_counts = {}
    for f in FULLSCALE_DIR.glob("*.flac"):
        sid = f.name.split("-")[0]
        prior_counts[sid] = prior_counts.get(sid, 0) + 1
    # Speakers that already have enough utterances on disk are skipped up-front
    # so the stream jumps straight to speakers that still need audio.
    finished = {s for s, n in prior_counts.items() if n >= utterances_per_speaker}
    if finished:
        print(f"[*] Resuming: {len(finished)} speakers already complete on disk", flush=True)

    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    new_spks = set(finished)
    saved = 0
    for ex in ds:
        if len(new_spks) >= num_speakers:
            break
        sid = str(ex["speaker_id"])
        if sid in existing or sid in finished:
            continue
        out_path = FULLSCALE_DIR / f"{ex['id']}.flac"
        if not out_path.exists():
            try:
                array, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            except Exception as e:
                print(f"    [!] skip undecodable {ex['id']}: {e}", flush=True)
                continue
            sf.write(out_path, array, sr, format="FLAC")
            saved += 1
        if _count_for(sid) >= utterances_per_speaker:
            new_spks.add(sid)
        if saved % 20 == 0 and saved > 0:
            print(f"    ... {saved} new utterances written, {len(new_spks)} speakers so far", flush=True)

    overlap = existing.intersection(new_spks)
    assert not overlap, f"Speaker overlap with existing pool: {overlap}"
    return saved, existing, new_spks


def fetch_new_speakers_parquet(num_speakers, utterances_per_speaker=2):
    """Select new speakers by reading train.100 parquet shards from the HF mirror.

    train.100 ships as 14 parquet shards (~440 MB each, ~2200 utterances / ~40
    speakers per shard). Shards are downloaded one at a time only until enough
    new speakers have been collected, so only ~3-4 shards are needed.
    """
    import pyarrow.parquet as pq

    existing = existing_speaker_ids()
    print(f"[*] Existing pool speakers to exclude: {len(existing)}", flush=True)
    FULLSCALE_DIR.mkdir(parents=True, exist_ok=True)

    prior_counts = {}
    for f in FULLSCALE_DIR.glob("*.flac"):
        s = f.name.split("-")[0]
        prior_counts[s] = prior_counts.get(s, 0) + 1
    finished = {s for s, n in prior_counts.items() if n >= utterances_per_speaker}
    if finished:
        print(f"[*] Resuming: {len(finished)} speakers already complete on disk", flush=True)

    new_spks = set(finished)
    saved = 0
    shard_idx = 0
    while len(new_spks) < num_speakers and shard_idx < 14:
        name = f"clean/train.100/{shard_idx:04d}.parquet"
        shard_idx += 1
        print(f"[*] Downloading shard {name} ...", flush=True)
        paths = _download_parquet_shards([name])
        table = pq.read_table(paths[0], columns=["speaker_id", "id", "audio"])
        spk_col = [str(v) for v in table["speaker_id"].to_pylist()]
        id_col = table["id"].to_pylist()
        audio_col = table["audio"].to_pylist()
        for sid, utt_id, audio in zip(spk_col, id_col, audio_col):
            if len(new_spks) >= num_speakers:
                break
            if sid in existing or sid in finished:
                continue
            if len(list(FULLSCALE_DIR.glob(f"{sid}-*.flac"))) >= utterances_per_speaker:
                new_spks.add(sid)
                continue
            out_path = FULLSCALE_DIR / f"{utt_id}.flac"
            if not out_path.exists():
                try:
                    array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                except Exception as e:
                    print(f"    [!] skip undecodable {utt_id}: {e}", flush=True)
                    continue
                sf.write(out_path, array, sr, format="FLAC")
                saved += 1
            if len(list(FULLSCALE_DIR.glob(f"{sid}-*.flac"))) >= utterances_per_speaker:
                new_spks.add(sid)
        print(f"    -> {len(new_spks)} speakers collected after shard {shard_idx-1} ({saved} new files)", flush=True)

    if len(new_spks) < num_speakers:
        raise RuntimeError(f"Only collected {len(new_spks)} new speakers after all shards")
    overlap = existing.intersection(new_spks)
    assert not overlap, f"Speaker overlap with existing pool: {overlap}"
    return saved, existing, new_spks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-speakers", type=int, default=130)
    ap.add_argument("--utterances-per-speaker", type=int, default=2)
    ap.add_argument("--speakers-txt", default=None)
    ap.add_argument("--source", choices=["parquet", "stream"], default="parquet",
                    help="parquet = read train.100 parquet shards from the HF mirror (fast, robust); "
                         "stream = HF streaming (slow, used only as fallback)")
    args = ap.parse_args()

    if args.source == "parquet":
        saved, existing, new = fetch_new_speakers_parquet(args.num_speakers, args.utterances_per_speaker)
    else:
        saved, existing, new = fetch_new_speakers(args.num_speakers, args.utterances_per_speaker, args.speakers_txt)
    total_files = len(list(FULLSCALE_DIR.glob("*.flac")))
    print("\n" + "=" * 60, flush=True)
    print("FULL-SCALE POOL EXTENSION VERIFICATION:", flush=True)
    print(f"  Existing pool speakers:       {len(existing)}", flush=True)
    print(f"  New train-clean-100 speakers: {len(new)}", flush=True)
    print(f"  Zero overlap verified:        {len(existing.intersection(new)) == 0}", flush=True)
    print(f"  Total .flac in fullscale dir: {total_files}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
