from app import diarize


def test_segment_gets_speaker_with_most_overlap():
    segments = [{"start": 0.0, "end": 4.0, "text": "hello there"}]
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    diarize.assign_speakers(segments, turns)
    assert segments[0]["speaker"] == "SPEAKER_01"


def test_segment_with_no_overlap_stays_unlabeled():
    segments = [{"start": 10.0, "end": 12.0, "text": "late words"}]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    diarize.assign_speakers(segments, turns)
    assert segments[0]["speaker"] is None


def test_no_turns_at_all():
    segments = [{"start": 0.0, "end": 2.0, "text": "hi"}]
    diarize.assign_speakers(segments, [])
    assert segments[0]["speaker"] is None


def test_multiple_segments_alternating_speakers():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "question"},
        {"start": 2.0, "end": 4.0, "text": "answer"},
        {"start": 4.0, "end": 6.0, "text": "follow up"},
    ]
    turns = [
        {"start": 0.0, "end": 2.1, "speaker": "SPEAKER_00"},
        {"start": 2.1, "end": 4.1, "speaker": "SPEAKER_01"},
        {"start": 4.1, "end": 6.0, "speaker": "SPEAKER_00"},
    ]
    diarize.assign_speakers(segments, turns)
    assert [s["speaker"] for s in segments] == [
        "SPEAKER_00", "SPEAKER_01", "SPEAKER_00",
    ]


def test_touching_but_not_overlapping_turn_is_ignored():
    # Zero-length overlap (turn ends exactly where the segment starts)
    # must not count as a match.
    segments = [{"start": 2.0, "end": 3.0, "text": "hm"}]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    diarize.assign_speakers(segments, turns)
    assert segments[0]["speaker"] is None
