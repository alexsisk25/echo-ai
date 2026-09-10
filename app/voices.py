"""Voice fingerprinting: recognize recurring speakers across recordings.

When the user renames a diarized speaker (SPEAKER_00 -> "Sam"), that voice is
enrolled: an averaged ECAPA-TDNN embedding stored in the voiceprints
table. New recordings cosine-match each diarized speaker against the
enrolled prints and auto-tag confident matches. Everything runs local.

Matching is assistive, not perfect (noisy audio degrades it), so it is
deliberately conservative: a speaker is tagged only when two or more of
its segments individually match the same enrolled voice above the
threshold. Wrong tags are one click to fix in the UI.
"""

import logging
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from app import config

log = logging.getLogger("otter.voices")

MATCH_THRESHOLD = 0.5   # conservative per PLAN.md
MIN_MATCHING_SEGMENTS = 2
MIN_SEGMENT_SECONDS = 1.0
MAX_SEGMENTS_PER_SPEAKER = 10
SAMPLE_RATE = 16000

_classifier = None
_classifier_lock = threading.Lock()


def _get_classifier():
    global _classifier
    # The retag sweep runs on a background thread, so lazy init needs a lock.
    with _classifier_lock:
        if _classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier

            _classifier = EncoderClassifier.from_hparams(
                source=config.VOICE_MODEL,
                savedir=config.DATA_DIR / "models" / "ecapa",
            )
    return _classifier


def _load_audio(audio_path: str | Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 samples via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
             "-ar", str(SAMPLE_RATE), "-ac", "1", tmp.name],
            check=True,
        )
        import torchaudio

        wave, _ = torchaudio.load(tmp.name)
    return wave[0].numpy()


def _pick_segments(segments: list[dict]) -> list[dict]:
    """The longest few segments make the cleanest voice samples."""
    usable = [s for s in segments
              if s["end"] - s["start"] >= MIN_SEGMENT_SECONDS]
    usable.sort(key=lambda s: s["end"] - s["start"], reverse=True)
    return usable[:MAX_SEGMENTS_PER_SPEAKER]


def embed_segments(audio: np.ndarray, segments: list[dict]) -> np.ndarray:
    """One normalized ECAPA embedding per segment: shape (n, 192)."""
    import torch

    clf = _get_classifier()
    out = []
    for s in segments:
        lo = int(s["start"] * SAMPLE_RATE)
        hi = min(int(s["end"] * SAMPLE_RATE), len(audio))
        if hi - lo < SAMPLE_RATE * MIN_SEGMENT_SECONDS:
            continue
        clip = torch.tensor(audio[lo:hi]).unsqueeze(0)
        emb = clf.encode_batch(clip).squeeze().detach().numpy()
        out.append(emb / np.linalg.norm(emb))
    return np.array(out)


def speaker_segments(conn, transcript_id: int) -> dict[str, list[dict]]:
    rows = conn.execute(
        "SELECT speaker, start_seconds, end_seconds FROM segments "
        "WHERE transcript_id = ? AND speaker IS NOT NULL",
        (transcript_id,),
    ).fetchall()
    by_speaker = {}
    for speaker, start, end in rows:
        by_speaker.setdefault(speaker, []).append(
            {"start": start, "end": end}
        )
    return by_speaker


def enroll(conn, name: str, embeddings: np.ndarray) -> None:
    """Add samples to a voiceprint, averaging with what is already there."""
    if len(embeddings) == 0:
        return
    new_mean = embeddings.mean(axis=0)
    row = conn.execute(
        "SELECT embedding, num_samples FROM voiceprints WHERE name = ?",
        (name,),
    ).fetchone()
    if row:
        old = np.frombuffer(row[0], dtype=np.float32)
        n_old = row[1]
        merged = (old * n_old + new_mean * len(embeddings)) / (
            n_old + len(embeddings)
        )
        merged = merged / np.linalg.norm(merged)
        conn.execute(
            "UPDATE voiceprints SET embedding = ?, num_samples = ? "
            "WHERE name = ?",
            (merged.astype(np.float32).tobytes(),
             n_old + len(embeddings), name),
        )
    else:
        mean = new_mean / np.linalg.norm(new_mean)
        conn.execute(
            "INSERT INTO voiceprints (name, embedding, num_samples) "
            "VALUES (?, ?, ?)",
            (name, mean.astype(np.float32).tobytes(), len(embeddings)),
        )
    conn.commit()


def enroll_from_recording(conn, transcript_id: int, audio_path: str,
                          speaker_label: str, name: str) -> int:
    """Enroll a voice from one recording's segments for one speaker.
    Returns the number of segment samples used."""
    segments = speaker_segments(conn, transcript_id).get(speaker_label, [])
    picked = _pick_segments(segments)
    if not picked:
        return 0
    audio = _load_audio(audio_path)
    embeddings = embed_segments(audio, picked)
    enroll(conn, name, embeddings)
    return len(embeddings)


def load_voiceprints(conn) -> dict[str, np.ndarray]:
    return {
        name: np.frombuffer(blob, dtype=np.float32)
        for name, blob in conn.execute(
            "SELECT name, embedding FROM voiceprints"
        ).fetchall()
    }


def match_speaker(segment_embeddings: np.ndarray,
                  voiceprints: dict[str, np.ndarray]) -> str | None:
    """Name whose print at least two segment embeddings match, else None.
    Ties go to the name with the highest mean similarity."""
    if len(segment_embeddings) == 0 or not voiceprints:
        return None
    best_name, best_score = None, 0.0
    for name, print_vec in voiceprints.items():
        sims = segment_embeddings @ print_vec
        n_matching = int((sims >= MATCH_THRESHOLD).sum())
        if n_matching >= MIN_MATCHING_SEGMENTS:
            score = float(sims.mean())
            if score > best_score:
                best_name, best_score = name, score
    return best_name


# Guided-split thresholds. Deliberately stricter than auto-tagging:
# splitting a cluster rewrites who said what, so it needs sustained
# evidence, not a couple of lucky frames.
SPLIT_MATCH_THRESHOLD = 0.55
SPLIT_MIN_SEGMENTS = 4
SPLIT_MIN_SECONDS = 20.0
SPLIT_MAX_SEGMENTS_SCANNED = 120


def guided_split(conn, transcript_id: int, audio_path: str) -> dict:
    """Split a diarized cluster that clearly holds more than one voice.

    Diarization in a noisy room can collapse two people into one label.
    Auto-tagging only ever renames a whole cluster, so it then names the
    merged blob after whichever voice it matched, which is how a
    two-person conversation ends up entirely as the user.

    This looks inside each generic cluster: every usable segment is
    embedded and scored against the enrolled voiceprints. When a
    cluster's segments divide into a sustained group matching an
    enrolled voice and a sustained remainder that does not, the matching
    group is relabelled to that name and the rest keeps the original
    label. Segments the evidence does not cover are never moved.

    Conservative by construction: only unpinned, machine-labelled
    segments are considered, both sides of a split must clear
    SPLIT_MIN_SEGMENTS and SPLIT_MIN_SECONDS, moved segments record
    auto_original so the split is revertible, and a rejected
    (label, name) pairing is respected.

    Returns {"split": {label: {name: n_segments}}}.
    """
    voiceprints = load_voiceprints(conn)
    if not voiceprints:
        return {"split": {}}

    rows = conn.execute(
        "SELECT id, speaker, start_seconds, end_seconds FROM segments "
        "WHERE transcript_id = ? AND pinned = 0 AND speaker IS NOT NULL "
        "AND speaker LIKE 'SPEAKER^_%' ESCAPE '^'",
        (transcript_id,),
    ).fetchall()
    if not rows:
        return {"split": {}}

    by_label = {}
    for seg_id, speaker, start, end in rows:
        by_label.setdefault(speaker, []).append(
            {"id": seg_id, "start": start, "end": end})

    rejected = set(conn.execute(
        "SELECT label, name FROM rejected_tags WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchall())

    audio = _load_audio(audio_path)
    split = {}
    for label, segments in by_label.items():
        candidates = {n: v for n, v in voiceprints.items()
                      if (label, n) not in rejected}
        if not candidates:
            continue
        # Mirror exactly what embed_segments will keep, so segment i
        # always corresponds to embedding i. It drops clips that run
        # past the end of the audio, and a silent skip in the middle
        # would shift the pairing and label people wrongly.
        audio_seconds = len(audio) / SAMPLE_RATE
        usable = [
            s for s in segments
            if min(s["end"], audio_seconds) - s["start"] >= MIN_SEGMENT_SECONDS
        ]
        # Two sustained groups have to fit inside the cluster at all.
        if len(usable) < SPLIT_MIN_SEGMENTS * 2:
            continue
        # Longest first, so a scan cap still sees the best evidence.
        usable.sort(key=lambda s: s["end"] - s["start"], reverse=True)
        usable = usable[:SPLIT_MAX_SEGMENTS_SCANNED]

        embeddings = embed_segments(audio, usable)
        if len(embeddings) != len(usable):
            # Should not happen now, but never risk a shifted pairing.
            log.warning("Embedding count %d != segment count %d for %s; "
                        "skipping the split", len(embeddings), len(usable),
                        label)
            continue
        if len(embeddings) < SPLIT_MIN_SEGMENTS * 2:
            continue
        scanned = usable

        # Each segment goes to its best-matching enrolled voice, or to
        # the unmatched pile.
        groups = {}
        unmatched = []
        for seg, emb in zip(scanned, embeddings):
            best_name, best_sim = None, 0.0
            for name, print_vec in candidates.items():
                sim = float(emb @ print_vec)
                if sim > best_sim:
                    best_name, best_sim = name, sim
            if best_name and best_sim >= SPLIT_MATCH_THRESHOLD:
                groups.setdefault(best_name, []).append(seg)
            else:
                unmatched.append(seg)

        def sustained(segs):
            return (len(segs) >= SPLIT_MIN_SEGMENTS
                    and sum(s["end"] - s["start"] for s in segs)
                    >= SPLIT_MIN_SECONDS)

        named = {n: segs for n, segs in groups.items() if sustained(segs)}
        if not named:
            continue
        # A split needs a sustained other side too: either a second
        # named voice, or a sustained unmatched remainder. Otherwise the
        # whole cluster is just that one person, which auto_tag handles.
        others = sustained(unmatched) or len(named) >= 2
        if not others:
            continue

        moved = {}
        for name, segs in named.items():
            ids = [s["id"] for s in segs]
            conn.executemany(
                "UPDATE segments SET speaker = ?, auto_original = ? "
                "WHERE id = ? AND pinned = 0",
                [(name, label, i) for i in ids],
            )
            moved[name] = len(ids)
        split[label] = moved
        log.info("Guided split of %s in transcript %d: %s",
                 label, transcript_id, moved)
    conn.commit()
    return {"split": split}


def split_pairs(split_result: dict) -> set[tuple[str, str]]:
    """The (label, name) pairs guided_split just separated out.

    Handed to auto_tag so it cannot immediately undo the split: a
    segment scoring just under the split threshold still clears the
    looser tagging threshold, so without this the remainder of a
    freshly split cluster could be renamed to the very person who was
    split out of it, re-merging two people under one name.
    """
    return {(label, name)
            for label, names in (split_result or {}).get("split", {}).items()
            for name in names}


def auto_tag(conn, transcript_id: int, audio_path: str,
             only_names: set[str] | None = None,
             exclude: set[tuple[str, str]] | None = None) -> dict[str, str]:
    """Rename diarized speakers that match enrolled voices.
    Returns {old_label: new_name} for what was tagged.

    Machine tags record the label they replaced in auto_original so they
    can be reverted, and a rejected (transcript, label, name) combination
    is never re-applied. only_names restricts matching to those enrolled
    voices (the retag sweep passes the one just enrolled)."""
    voiceprints = load_voiceprints(conn)
    if only_names is not None:
        voiceprints = {n: v for n, v in voiceprints.items()
                       if n in only_names}
    if not voiceprints:
        return {}
    by_speaker = speaker_segments(conn, transcript_id)
    generic = {s: segs for s, segs in by_speaker.items()
               if s.startswith("SPEAKER_")}
    if not generic:
        return {}
    rejected = set(conn.execute(
        "SELECT label, name FROM rejected_tags WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchall())
    # A pairing just split apart is off limits here, or tagging would
    # put the two people straight back under one name.
    rejected |= (exclude or set())

    audio = _load_audio(audio_path)
    tagged = {}
    for label, segments in generic.items():
        candidates = {n: v for n, v in voiceprints.items()
                      if (label, n) not in rejected}
        if not candidates:
            continue
        picked = _pick_segments(segments)
        if len(picked) < MIN_MATCHING_SEGMENTS:
            continue
        embeddings = embed_segments(audio, picked)
        name = match_speaker(embeddings, candidates)
        if name:
            # Pinned lines were assigned by a human and are never
            # moved by machine tagging.
            conn.execute(
                "UPDATE segments SET speaker = ?, auto_original = speaker "
                "WHERE transcript_id = ? AND speaker = ? AND pinned = 0",
                (name, transcript_id, label),
            )
            tagged[label] = name
    conn.commit()
    return tagged


def reset_speakers(conn, recording_id: int) -> dict | None:
    """Revert one recording's speaker labels to fresh diarization output.

    Once clusters have been renamed over each other the original
    SPEAKER_XX labels are not recoverable from the database, so this
    re-runs diarization on the audio and realigns every segment. All
    human and machine labels for this recording are cleared, including
    pinned lines and stored tag rejections; other recordings and
    enrolled voices are untouched. Enrolled voices are then re-applied
    with the usual conservative auto-tagging.

    Returns {"speakers": [...], "tagged": {label: name}} or None when
    the recording, its transcript, or its audio file is missing.
    """
    from app import diarize

    row = conn.execute(
        "SELECT r.file_path, t.id FROM recordings r "
        "JOIN transcripts t ON t.recording_id = r.id WHERE r.id = ?",
        (recording_id,),
    ).fetchone()
    if row is None or not Path(row[0]).exists():
        return None
    path, transcript_id = row

    turns = diarize.diarize(path)
    seg_dicts = [
        {"id": s[0], "start": s[1], "end": s[2]}
        for s in conn.execute(
            "SELECT id, start_seconds, end_seconds FROM segments "
            "WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchall()
    ]
    diarize.assign_speakers(seg_dicts, turns)
    for s in seg_dicts:
        conn.execute(
            "UPDATE segments SET speaker = ?, auto_original = NULL, "
            "pinned = 0 WHERE id = ?",
            (s["speaker"], s["id"]),
        )
    conn.execute(
        "DELETE FROM rejected_tags WHERE transcript_id = ?", (transcript_id,)
    )
    conn.commit()

    tagged = auto_tag(conn, transcript_id, path)
    return {
        "speakers": sorted({s["speaker"] for s in seg_dicts if s["speaker"]}),
        "tagged": tagged,
    }


_retag_lock = threading.Lock()


def retag_archive(name: str, db_path=None, key=None) -> dict[int, dict]:
    """Sweep every done recording that still has generic SPEAKER_XX labels
    and auto-tag the given enrolled voice wherever it matches.

    Meant for a background thread after enrollment, so it opens its own
    connection (SQLite connections cannot cross threads) and the caller
    resolves db_path and key up front. One sweep at a time.
    Returns {transcript_id: {old_label: name}} for what was tagged."""
    from app import db

    with _retag_lock:
        conn = db.connect(db_path=db_path, key=key)
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT t.id, r.file_path
                FROM segments s
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE r.status = 'done'
                  AND s.speaker LIKE 'SPEAKER^_%' ESCAPE '^'
                """
            ).fetchall()
            results = {}
            for transcript_id, path in rows:
                if not Path(path).exists():
                    continue
                try:
                    tagged = auto_tag(conn, transcript_id, path,
                                      only_names={name})
                except Exception:
                    log.exception("Retroactive tag failed for transcript %d",
                                  transcript_id)
                    continue
                if tagged:
                    results[transcript_id] = tagged
                    log.info("Retroactively tagged transcript %d: %s",
                             transcript_id, tagged)
            log.info("Retag sweep for %s: %d recording(s) checked, "
                     "%d tagged", name, len(rows), len(results))
            return results
        finally:
            conn.close()


def list_voices(conn) -> list[dict]:
    """Every enrolled voice with how widely it appears in the archive."""
    rows = conn.execute(
        """
        SELECT v.name, v.num_samples, v.created_at,
               count(DISTINCT t.recording_id),
               count(s.id),
               count(s.auto_original)
        FROM voiceprints v
        LEFT JOIN segments s ON s.speaker = v.name
        LEFT JOIN transcripts t ON t.id = s.transcript_id
        GROUP BY v.name
        ORDER BY v.name
        """
    ).fetchall()
    return [
        {"name": r[0], "num_samples": r[1], "created_at": r[2],
         "recordings": r[3], "segments": r[4], "auto_segments": r[5]}
        for r in rows
    ]


def rename_voice(conn, old: str, new: str) -> int | None:
    """Rename an enrollment and every segment carrying that name (the
    name came from this enrollment, so human and machine tags follow).
    Returns segments changed, None when the voice does not exist.
    Raises ValueError when the new name is already enrolled."""
    if conn.execute("SELECT 1 FROM voiceprints WHERE name = ?",
                    (old,)).fetchone() is None:
        return None
    if conn.execute("SELECT 1 FROM voiceprints WHERE name = ?",
                    (new,)).fetchone() is not None:
        raise ValueError(f"A voice named {new} is already enrolled")
    conn.execute("UPDATE voiceprints SET name = ? WHERE name = ?",
                 (new, old))
    cur = conn.execute("UPDATE segments SET speaker = ? WHERE speaker = ?",
                       (new, old))
    conn.execute("UPDATE rejected_tags SET name = ? WHERE name = ?",
                 (new, old))
    # The cached dossier is keyed by the old name; drop it, it rebuilds.
    conn.execute("DELETE FROM dossiers WHERE name = ?", (old,))
    conn.commit()
    return cur.rowcount


def delete_voice(conn, name: str) -> int | None:
    """Delete an enrollment and revert its machine-applied tags back to
    their SPEAKER_XX labels. Labels a human set stay untouched.
    Returns segments reverted, None when the voice does not exist."""
    if conn.execute("SELECT 1 FROM voiceprints WHERE name = ?",
                    (name,)).fetchone() is None:
        return None
    cur = conn.execute(
        "UPDATE segments SET speaker = auto_original, auto_original = NULL "
        "WHERE speaker = ? AND auto_original IS NOT NULL",
        (name,),
    )
    conn.execute("DELETE FROM voiceprints WHERE name = ?", (name,))
    conn.commit()
    return cur.rowcount


def confirm_tag(conn, transcript_id: int, label: str) -> int:
    """Promote a machine tag to human-confirmed. Returns segments changed."""
    cur = conn.execute(
        "UPDATE segments SET auto_original = NULL "
        "WHERE transcript_id = ? AND speaker = ? "
        "AND auto_original IS NOT NULL",
        (transcript_id, label),
    )
    conn.commit()
    return cur.rowcount


def reject_tag(conn, transcript_id: int, label: str) -> int:
    """Revert a machine tag to its SPEAKER_XX label and remember the
    rejection as a negative example so this voice is never re-applied
    to that speaker. Returns segments changed."""
    originals = [r[0] for r in conn.execute(
        "SELECT DISTINCT auto_original FROM segments "
        "WHERE transcript_id = ? AND speaker = ? "
        "AND auto_original IS NOT NULL",
        (transcript_id, label),
    ).fetchall()]
    for original in originals:
        conn.execute(
            "INSERT OR IGNORE INTO rejected_tags (transcript_id, label, name) "
            "VALUES (?, ?, ?)",
            (transcript_id, original, label),
        )
    cur = conn.execute(
        "UPDATE segments SET speaker = auto_original, auto_original = NULL "
        "WHERE transcript_id = ? AND speaker = ? "
        "AND auto_original IS NOT NULL",
        (transcript_id, label),
    )
    conn.commit()
    return cur.rowcount
