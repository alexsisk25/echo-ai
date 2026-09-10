"""Detect two diarization clusters living under one speaker name.

The failure this catches (recording 87): diarization separated a
restaurant conversation into a 61% cluster and a 35% cluster, voice
tagging named one of them, and a "Rename everywhere" folded the other
one in. Both people then sat under a single name, which reads as "this
whole meeting was me" and is invisible in the UI.

Finding it is not as simple as grouping by the current label, because
the merge erases the distinction. Two stages:

1. Structural. Segments under one real name are grouped by where they
   came from: each distinct auto_original value is a group (a machine
   tag records the SPEAKER_XX label it replaced), and every segment
   with no recorded origin forms one more group. A name covering two
   groups that each hold a meaningful share of the speech is a
   candidate.

2. Acoustic. A candidate is only a merge if the groups are actually
   different people. This matters because confirming an auto-tagged
   line clears auto_original, so a heavily confirmed cluster looks
   structurally identical to a merge while being one voice. Comparing
   group centroids separates the two cases. Measured on the real
   archive: recording 25's two halves sit at 0.10 and recording 87's at
   0.39, against a 0.45 threshold. 87 is the tighter of the two, which
   is expected (both speakers recorded on one phone mic in a noisy
   restaurant), so the margin there is real but not generous. If a
   merge is ever missed, this is the number to revisit.

Nothing here changes a label. It only sets a flag the UI turns into a
dismissible suggestion.
"""

import logging
from pathlib import Path

import numpy as np

from app import voices

log = logging.getLogger("otter.merged")

# Each side of a suspected merge must hold at least this share of the
# recording's speech. Below it, the "cluster" is a fragment and not
# worth interrupting anyone about.
MIN_SHARE = 0.15
# Centroid cosine above this means the two groups are the same person
# (a confirmed cluster), so it is not a merge.
SAME_VOICE_SIMILARITY = 0.45
# Segments sampled per group for the acoustic check.
SAMPLE_PER_GROUP = 8


def _origin_groups(conn, transcript_id: int) -> dict:
    """{name: {origin_key: [segment rows]}} for real (non-generic) names.

    origin_key is the recorded auto_original, or "" when the segment
    carries no history.
    """
    rows = conn.execute(
        "SELECT speaker, auto_original, start_seconds, end_seconds "
        "FROM segments WHERE transcript_id = ? AND speaker IS NOT NULL "
        "AND speaker NOT LIKE 'SPEAKER^_%' ESCAPE '^'",
        (transcript_id,),
    ).fetchall()
    by_name = {}
    for speaker, origin, start, end in rows:
        by_name.setdefault(speaker, {}).setdefault(origin or "", []).append(
            {"start": start, "end": end})
    return by_name


def _speech_seconds(segments) -> float:
    return sum(s["end"] - s["start"] for s in segments)


def _different_voices(audio, group_a, group_b) -> bool | None:
    """Are these two groups different people? None when unmeasurable."""
    def centroid(group):
        usable = [s for s in group
                  if s["end"] - s["start"] >= voices.MIN_SEGMENT_SECONDS]
        usable.sort(key=lambda s: s["end"] - s["start"], reverse=True)
        embeddings = voices.embed_segments(audio, usable[:SAMPLE_PER_GROUP])
        if len(embeddings) == 0:
            return None
        mean = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm else None

    a, b = centroid(group_a), centroid(group_b)
    if a is None or b is None:
        return None
    return float(a @ b) < SAME_VOICE_SIMILARITY


def detect(conn, recording_id: int, verify_audio: bool = True) -> dict | None:
    """Find a name covering two substantial, acoustically distinct
    clusters. Returns {"name", "shares", "verified"} or None.

    Reads only; the caller decides whether to record the finding.
    """
    row = conn.execute(
        "SELECT t.id, r.file_path FROM transcripts t "
        "JOIN recordings r ON r.id = t.recording_id "
        "WHERE t.recording_id = ?", (recording_id,),
    ).fetchone()
    if row is None:
        return None
    transcript_id, file_path = row

    total = conn.execute(
        "SELECT COALESCE(SUM(end_seconds - start_seconds), 0) FROM segments "
        "WHERE transcript_id = ?", (transcript_id,),
    ).fetchone()[0] or 0.0
    if total <= 0:
        return None

    audio = None
    findings = []
    for name, groups in _origin_groups(conn, transcript_id).items():
        substantial = sorted(
            ((key, segs) for key, segs in groups.items()
             if _speech_seconds(segs) / total >= MIN_SHARE),
            key=lambda kv: _speech_seconds(kv[1]), reverse=True,
        )
        if len(substantial) < 2:
            continue

        shares = {
            (key or "unrecorded origin"):
                round(_speech_seconds(segs) / total, 3)
            for key, segs in substantial
        }
        verified = None
        if verify_audio and Path(file_path).exists():
            try:
                if audio is None:
                    audio = voices._load_audio(file_path)
                verified = _different_voices(
                    audio, substantial[0][1], substantial[1][1])
            except Exception:
                log.exception("Acoustic check failed for recording %d",
                              recording_id)
                verified = None
            # Same voice on both sides: a confirmed cluster, not a merge.
            if verified is False:
                continue
        findings.append({"name": name, "shares": shares,
                         "verified": verified})
    if not findings:
        return None
    # A recording can hold more than one merged name; the flag is a
    # single suggestion, so report the most substantial one (largest
    # smaller-side share) and stay deterministic about which.
    findings.sort(key=lambda f: sorted(f["shares"].values())[-2],
                  reverse=True)
    return findings[0]


_UNSET = object()


def refresh(conn, recording_id: int, verify_audio: bool = True,
            finding=_UNSET) -> str | None:
    """Run the detector and record the result. Returns the flagged name.

    A recording that comes back clean has any previous flag and its
    dismissal cleared, so a merge introduced later is surfaced again.
    A flag naming a DIFFERENT speaker than the dismissed one is a new
    finding too, so it clears the dismissal rather than inheriting it.

    finding lets a caller that already ran detect() pass the result in
    instead of paying for the audio work twice.
    """
    if finding is _UNSET:
        finding = detect(conn, recording_id, verify_audio=verify_audio)
    name = finding["name"] if finding else None
    previous = conn.execute(
        "SELECT merged_name_suspect FROM recordings WHERE id = ?",
        (recording_id,)).fetchone()
    previous = previous[0] if previous else None
    if name:
        if name == previous:
            conn.execute(
                "UPDATE recordings SET merged_name_suspect = ? WHERE id = ?",
                (name, recording_id))
        else:
            conn.execute(
                "UPDATE recordings SET merged_name_suspect = ?, "
                "merged_name_dismissed = 0 WHERE id = ?",
                (name, recording_id))
    else:
        conn.execute(
            "UPDATE recordings SET merged_name_suspect = NULL, "
            "merged_name_dismissed = 0 WHERE id = ?", (recording_id,))
    conn.commit()
    return name


def scan_archive(conn, dry_run: bool = False,
                 verify_audio: bool = True) -> list[dict]:
    """One-time sweep of every processed recording.

    dry_run reports without touching the database at all, which is how
    the archive was first surveyed.
    """
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM recordings WHERE status = 'done' ORDER BY id"
    ).fetchall()]
    found = []
    for rec_id in ids:
        try:
            finding = detect(conn, rec_id, verify_audio=verify_audio)
        except Exception:
            log.exception("Merged-name scan failed for recording %d", rec_id)
            continue
        if not dry_run:
            refresh(conn, rec_id, verify_audio=verify_audio,
                    finding=finding)
        if finding:
            finding["recording_id"] = rec_id
            found.append(finding)
    return found
