"""Tests for Jazz Trio Database (JTD)
"""

import os
from typing import Tuple

import numpy as np

import pytest
from mirdata import annotations
from mirdata.datasets import jtd
from tests.test_utils import run_track_tests, run_multitrack_tests


def test_track():
    default_trackid = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano"
    data_home = "tests/resources/mir_datasets/jtd"
    dataset = jtd.Dataset(data_home, version="sample")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano",
        "audio_path": os.path.join(  # using os.path.join allows this to work properly on both Windows and Linux
            "tests/resources/mir_datasets/jtd",
            "processed/barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano.wav",
        ),
        "midi_path": os.path.join(
            "tests/resources/mir_datasets/jtd",
            "annotations/barronk-allgodschildren-drummondrrileyb-1990-8b77c067/piano_midi.mid",
        ),
        "onsets_path": os.path.join(
            "tests/resources/mir_datasets/jtd",
            "annotations/barronk-allgodschildren-drummondrrileyb-1990-8b77c067/piano_onsets.csv",
        ),
        "beats_path": os.path.join(
            "tests/resources/mir_datasets/jtd",
            "annotations/barronk-allgodschildren-drummondrrileyb-1990-8b77c067/beats.csv",
        ),
        "instrument": "piano",
    }

    expected_property_types = {
        "audio": Tuple,
        "midi": annotations.NoteData,
        "onsets": annotations.EventData,
        "musician": str,
        "beats": annotations.BeatData,
    }

    assert track._track_paths == {
        "audio": [
            "processed/barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano.wav",
            "fe5f025813675f301b458dcad361db19",
        ],
        "midi": [
            "annotations/barronk-allgodschildren-drummondrrileyb-1990-8b77c067/piano_midi.mid",
            "c4e6659f654c1e28c12d31db5e29d5d2",
        ],
        "beats": [
            "annotations/barronk-allgodschildren-drummondrrileyb-1990-8b77c067/beats.csv",
            "25f4ad5092b0381bb7472cde86db86cf",
        ],
        "onsets": [
            "annotations/barronk-allgodschildren-drummondrrileyb-1990-8b77c067/piano_onsets.csv",
            "ad03567467f6c2212c98b2c02d6569cf",
        ],
        "metadata": [
            "annotations/barronk-allgodschildren-drummondrrileyb-1990-8b77c067/metadata.json",
            "685a44e51b0f2991f84577de575a7186",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (44100 * 5,)  # five seconds of audio


def test_multitrack():
    default_trackid = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067"
    data_home = "tests/resources/mir_datasets/jtd"
    dataset = jtd.Dataset(data_home, version="sample")
    multitrack = dataset.multitrack(default_trackid)
    run_multitrack_tests(multitrack)
    # test track audio property
    assert multitrack.track_audio_property == "audio"
    # test audio loading functions
    audio, sr = multitrack.audio
    assert sr == 44100
    assert audio.shape == (44100 * 5,)  # five seconds of audio
    # test getting individual tracks
    # testing bass
    bass = multitrack.bass
    assert isinstance(bass, jtd.Track)
    assert bass.instrument == "bass"
    assert bass.musician == "Ray Drummond"
    # testing piano
    drums = multitrack.drums
    assert isinstance(drums, jtd.Track)
    assert drums.instrument == "drums"
    assert drums.musician == "Ben Riley"
    # testing piano
    piano = multitrack.piano
    assert isinstance(piano, jtd.Track)
    assert piano.instrument == "piano"
    assert piano.musician == "Kenny Barron"


def test_track_to_jams():
    default_trackid = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano"
    data_home = "tests/resources/mir_datasets/jtd"
    dataset = jtd.Dataset(data_home, version="sample")
    track = dataset.track(default_trackid)
    jam = track.to_jams()
    # testing MIDI
    midi_annot = jam.search(namespace="note_hz")[0]["data"]
    expected_pitches = [
        74,
        69,
        54,
    ]  # first two notes are D5, A4, F#3 (confirmed by checking MIDI in REAPER)
    assert all(
        [
            round(annotation.value) == expected
            for annotation, expected in zip(midi_annot, expected_pitches)
        ]
    )
    # testing onsets
    onset_annot = jam.search(namespace="tag_open")[0]["data"]
    expected_onsets = [0.0, 0.08, 0.29]  # taken directly from source data .csv
    assert all(
        [
            round(annotation.time, 2) == expected
            for annotation, expected in zip(onset_annot, expected_onsets)
        ]
    )
    # testing inter-onset intervals
    expected_durations = [0.08, 0.21]
    assert all(
        [
            round(annotation.duration, 2) == expected
            for annotation, expected in zip(onset_annot, expected_durations)
        ]
    )
    # testing beats
    expected_beats = [0.08, 0.29, 0.51]
    beat_annot = jam.search(namespace="beat")[0]["data"]
    assert all(
        [
            round(annotation.time, 2) == expected
            for annotation, expected in zip(beat_annot, expected_beats)
        ]
    )


def test_multitrack_to_jams():
    default_trackid = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067"
    data_home = "tests/resources/mir_datasets/jtd"
    dataset = jtd.Dataset(data_home, version="sample")
    track = dataset.multitrack(default_trackid)
    jam = track.to_jams()

    # testing tempo
    tempo_annot = jam.search(namespace="tempo")[0]["data"]
    expected_tempo = 297.62
    assert round(tempo_annot[0].value, 2) == expected_tempo

    # testing beats
    beat_annot = jam.search(namespace="beat")[0]["data"]
    expected_beats = [0.08, 0.28, 0.5, 0.7, 0.91, 1.11]
    assert all(
        [
            round(annotation.time, 2) == expected
            for annotation, expected in zip(beat_annot, expected_beats)
        ]
    )


def test_timestamp_to_seconds():
    # test case 1
    ts1 = "01:05"
    expected = 65
    actual = jtd.timestamp_to_seconds(ts1)
    assert actual == expected
    # test case 2
    ts2 = "03:03"
    expected = 183
    actual = jtd.timestamp_to_seconds(ts2)
    assert actual == expected
    # test case 3
    ts3 = "15:45"
    expected = 945
    actual = jtd.timestamp_to_seconds(ts3)
    assert actual == expected
    # test case 4
    ts4 = "01:02:03"
    expected = 3723
    actual = jtd.timestamp_to_seconds(ts4)
    assert actual == expected
    # test case 5
    ts5 = "asdfasgsegrijeaowfxgjflkawe"
    with pytest.raises(ValueError):
        _ = jtd.timestamp_to_seconds(ts5)


def test_load_onsets():
    # load an onsets file which exists
    annotation_path = (
        "tests/resources/mir_datasets/jtd/annotations/"
        "barronk-allgodschildren-drummondrrileyb-1990-8b77c067/bass_onsets.csv"
    )
    annotation_data = jtd.load_onsets(annotation_path)
    # check types
    assert isinstance(annotation_data, annotations.EventData)
    assert isinstance(annotation_data.intervals, np.ndarray)
    # check values
    expected = np.array(
        [
            0.09,
            0.29,
            0.5,
            0.72,
            0.91,
        ]
    )
    assert np.array_equal(annotation_data.intervals[:5, 0], expected)


def test_load_beats():
    # load a beats file which exists
    annotation_path = (
        "tests/resources/mir_datasets/jtd/annotations/"
        "barronk-allgodschildren-drummondrrileyb-1990-8b77c067/beats.csv"
    )
    # get the beats for the overall mixture
    annotation_data = jtd.load_beats(annotation_path, 0)
    # check types
    assert isinstance(annotation_data, annotations.BeatData)
    assert isinstance(annotation_data.times, np.ndarray)
    # check values
    expected = np.array([0.08, 0.28, 0.5, 0.7, 0.91, 1.11])
    assert np.array_equal(annotation_data.times[:6], expected)

    # get the beats for the drummer
    drums_data = jtd.load_beats(annotation_path, 3)
    # check types
    assert isinstance(drums_data, annotations.BeatData)
    assert isinstance(drums_data.times, np.ndarray)
    # check values
    expected = np.array([0.06, 0.28, 0.49, 0.7, 0.91, 1.11])
    assert np.array_equal(drums_data.times[:6], expected)


def test_track_properties():
    default_trackid = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano"
    data_home = "tests/resources/mir_datasets/jtd"
    dataset = jtd.Dataset(data_home, version="sample")
    track = dataset.track(default_trackid)
    assert track.instrument == "piano"
    assert track.musician == "Kenny Barron"


def test_multitrack_properties():
    default_trackid = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067"
    data_home = "tests/resources/mir_datasets/jtd"
    dataset = jtd.Dataset(data_home, version="sample")
    track = dataset.multitrack(default_trackid)
    assert track.name == "All Gods Children"
    assert track.album == "The Only One"
    assert track.year == 1990
    assert track.bandleader == "Kenny Barron"
    assert track.musicbrainz_id == "8b77c067-7620-4705-bb2d-c33e15e1912f"
    assert track.time_signature == 4
    assert track.jtd_300 is False
    assert track.start == 64
    assert track.stop == 69
    assert track.duration == 5
    assert round(track.tempo, 2) == 297.62
    assert track.audio_rchan is None
    assert track.audio_lchan is None


def test_no_midi():
    default_trackid = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067_bass"
    data_home = "tests/resources/mir_datasets/jtd"
    dataset = jtd.Dataset(data_home, version="sample")
    track = dataset.track(default_trackid)
    midi = track.midi
    assert midi is None
