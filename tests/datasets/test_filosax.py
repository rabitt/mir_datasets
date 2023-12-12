"""Tests for Filosax_Lite."""
import os

import jams
import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import filosax
from tests.test_utils import run_multitrack_tests, run_track_tests


def test_track():
    default_trackid = "multitrack_02_sax_1"
    data_home = os.path.normpath("tests/resources/mir_datasets/filosax")
    dataset = filosax.Dataset(data_home, version="test")
    filosax_data = dataset.load_tracks()
    default_track = filosax_data[default_trackid]

    expected_attributes = {
        "track_id": "multitrack_02_sax_1",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/filosax/"),
            "Participant 1/02/Sax.wav",
        ),
        "annotation_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/filosax/"),
            "Participant 1/02/annotations.json",
        ),
        "midi_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/filosax/"),
            "Participant 1/02/Sax.mid",
        ),
        "musicXML_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/filosax/"),
            "Participant 1/02/Sax.musicxml",
        ),
        "pdf_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/filosax/"),
            "Participant 1/02/Sax.pdf",
        ),
    }

    expected_property_types = {"notes": list, "audio": tuple}
    assert default_track._track_paths == {
        "audio": ["Participant 1/02/Sax.wav", "c62cd9d2ec8085461cb2ffc9deeda176"],
        "annotation": [
            "Participant 1/02/annotations.json",
            "8424df983f30e927c5a72b3d9bb94466",
        ],
        "midi": ["Participant 1/02/Sax.mid", "007beb3b84dfaafbc8767148bc00c0ff"],
        "musicXML": [
            "Participant 1/02/Sax.musicxml",
            "3846ef79efc36448ea0a8059c43935ac",
        ],
        "pdf": ["Participant 1/02/Sax.pdf", "69b20b5b95b8807137cc86e7c72cc861"],
    }

    run_track_tests(default_track, expected_attributes, expected_property_types)

    audio, sr = default_track.audio
    assert sr == 44100
    assert audio.shape == (44100 * 5,)


def test_multitrack():
    default_trackid = "multitrack_01"
    data_home = "tests/resources/mir_datasets/filosax"
    dataset = filosax.Dataset(data_home, version="test")
    default_track = dataset.multitrack(default_trackid)

    run_multitrack_tests(default_track)


def test_to_jams():
    default_trackid = "multitrack_01"
    data_home = "tests/resources/mir_datasets/filosax"
    dataset = filosax.Dataset(data_home, version="test")
    default_track = dataset.multitrack(default_trackid)
    jam = default_track.to_jams()

    beats = jam.search(namespace="beat")[0]["data"]
    chords = jam.search(namespace="chord")[0]["data"]
    segments = jam.search(namespace="segment_open")[0]["data"]

    assert [beat.time for beat in beats] == [0.0, 1.0, 2.672448, 4.279448]
    assert [beat.duration for beat in beats] == [
        1.0,
        1.6724480000000002,
        1.6070000000000002,
        1.6764999999999999,
    ]
    assert [beat.value for beat in beats] == [1, 1, 1, 1]

    assert [chord.time for chord in chords] == [1.0, 2.672448, 4.279448]
    assert [chord.duration for chord in chords] == [
        1.6724480000000002,
        1.6070000000000002,
        1.6764999999999999,
    ]
    assert [chord.value for chord in chords] == ["C#:7(b9)", "C#:7(b9)", "C:7(b9)"]

    assert [segment.time for segment in segments] == [
        1.0,
        14.046292,
        72.127318,
        137.51004,
    ]
    assert [segment.duration for segment in segments] == [
        13.046292,
        58.081026,
        65.382722,
        188.931246,
    ]
    assert [segment.value for segment in segments] == [
        "improvised solo",
        "head",
        "written solo",
        "improvised solo",
    ]


def test_load_annotation():
    annotation_path = "tests/resources/mir_datasets/filosax/Participant 1/01/annotations.json"
    annotation_data = filosax.load_annotation(annotation_path)

    # check types
    assert type(annotation_data) == list
    assert type(annotation_data[0]) is filosax.Note

    n = annotation_data[0]

    # check values
    assert n.a_start_time == 1.078095238095238
    assert n.a_end_time == 1.1941950113378685
    assert n.a_duration == 0.11609977324263054
    assert n.midi_pitch == 57
    assert n.crochet_num == 24
    assert n.musician == "Musician_A"
    assert n.bar_num == 2
    assert n.s_start_time == 1.0
    assert n.s_duration == 0.2090559999999999
    assert n.s_end_time == 1.209056
    assert n.s_rhythmic_duration == 12
    assert n.s_rhythmic_position == 0
    assert n.tempo == 143.502219500995
    assert n.bar_type == 2
    assert n.is_grace == 0
    assert n.chord_changes["0"] == "C#:7(b9)"
    assert n.num_chord_changes == 1
    assert n.main_chord_num == 0
    assert n.scale_changes[0] == 8
    assert n.loudness_max_val == -31.652
    assert n.loudness_max_time == 0.083
    assert len(n.loudness_curve) == 117
    assert n.loudness_curve[0] == -53.117
    assert n.pitch_average_val == 57.06023931623931
    assert n.pitch_average_time == 0.087
    assert len(n.pitch_curve) == 117
    assert n.pitch_curve[0] == 57.222
    assert n.pitch_vib_freq == 0.0
    assert n.pitch_vib_ext == 0.0
    assert n.spec_cent == 246.21
    assert n.spec_flux == 0.783
    assert len(n.spec_cent_curve) == 121
    assert n.spec_cent_curve[0] == 261.21
    assert len(n.spec_flux_curve) == 121
    assert n.spec_flux_curve[0] == 1.358


def test_metadata():
    default_multitrackid = "multitrack_01"
    data_home = "tests/resources/mir_datasets/filosax"
    dataset = filosax.Dataset(data_home, version="test")
    filosax_data = dataset.load_multitracks()
    default_multitrack = filosax_data[default_multitrackid]

    name = default_multitrack.name
    duration = default_multitrack.duration
    beats = default_multitrack.beats
    chords = default_multitrack.chords
    segments = default_multitrack.segments
    bass_drums = default_multitrack.bass_drums
    piano_drums = default_multitrack.piano_drums

    assert name == "All The Things You Are"
    assert duration == 5.0

    assert [beat.time for beat in beats] == [0.0, 1.0, 2.672448, 4.279448]
    assert [beat.duration for beat in beats] == [
        1.0,
        1.6724480000000002,
        1.6070000000000002,
        1.6764999999999999,
    ]
    assert [beat.value for beat in beats] == [1, 1, 1, 1]

    assert [chord.time for chord in chords] == [1.0, 2.672448, 4.279448]
    assert [chord.duration for chord in chords] == [
        1.6724480000000002,
        1.6070000000000002,
        1.6764999999999999,
    ]
    assert [chord.value for chord in chords] == ["C#:7(b9)", "C#:7(b9)", "C:7(b9)"]

    assert [segment.time for segment in segments] == [
        1.0,
        14.046292,
        72.127318,
        137.51004,
    ]
    assert [segment.duration for segment in segments] == [
        13.046292,
        58.081026,
        65.382722,
        188.931246,
    ]
    assert [segment.value for segment in segments] == [
        "improvised solo",
        "head",
        "written solo",
        "improvised solo",
    ]

    assert bass_drums.track_id == "multitrack_01_bass_drums"
    assert piano_drums.track_id == "multitrack_01_piano_drums"

    annotation_data = bass_drums.notes
    assert type(annotation_data) == list
    assert type(annotation_data[0]) is filosax.Note
    assert len(annotation_data) == 1
    assert annotation_data[0].a_start_time == 0.0
