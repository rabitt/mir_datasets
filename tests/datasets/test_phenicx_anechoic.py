import os

import numpy as np
import pytest

from mirdata import annotations, download_utils
from mirdata.datasets import phenicx_anechoic
from tests.test_utils import run_multitrack_tests, run_track_tests


def test_track():
    default_trackid = "beethoven-violin"
    data_home = os.path.normpath("tests/resources/mir_datasets/phenicx_anechoic")
    dataset = phenicx_anechoic.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "beethoven-violin",
        "audio_paths": [
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/phenicx_anechoic/"),
                "audio/beethoven/violin1.wav",
            ),
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/phenicx_anechoic/"),
                "audio/beethoven/violin2.wav",
            ),
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/phenicx_anechoic/"),
                "audio/beethoven/violin3.wav",
            ),
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/phenicx_anechoic/"),
                "audio/beethoven/violin4.wav",
            ),
        ],
        "notes_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/phenicx_anechoic/"),
            "annotations/beethoven/violin.txt",
        ),
        "notes_original_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/phenicx_anechoic/"),
            "annotations/beethoven/violin_o.txt",
        ),
        "instrument": "violin",
        "piece": "beethoven",
        "n_voices": 4,
    }

    expected_property_types = {
        "notes": annotations.NoteData,
        "notes_original": annotations.NoteData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100,)


def test_get_audio_voice():
    default_trackid = "beethoven-violin"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    track = dataset.track(default_trackid)

    y, sr = track.get_audio_voice(1)
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100,)

    with pytest.raises(ValueError):
        y, sr = track.get_audio_voice(5)


def test_to_jams():
    default_trackid = "beethoven-violin"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam.validate()

    notes = jam.annotations[0]["data"]
    assert [note.time for note in notes] == [4.284082, 4.284082, 4.284082]
    assert [note.duration for note in notes] == [
        0.9872560000000004,
        0.9872560000000004,
        0.9872560000000004,
    ]
    assert [note.value for note in notes] == [
        220.0,
        329.6275569128699,
        554.3652619537442,
    ]


def test_load_score():
    # load a file which exists
    score_path = "tests/resources/mir_datasets/phenicx_anechoic/annotations/beethoven/violin.txt"
    note_data = phenicx_anechoic.load_score(score_path)

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.pitches) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array([[4.284082, 5.271338], [4.284082, 5.271338], [4.284082, 5.271338]]),
    )
    assert np.allclose(note_data.pitches, np.array([220.0, 329.62755691, 554.36526195]))


def test_multitrack():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)
    # import pdb;pdb.set_trace()
    expected_attributes = {
        "mtrack_id": "beethoven",
        "track_audio_property": "audio",
        "track_ids": [
            "beethoven-horn",
            "beethoven-doublebass",
            "beethoven-violin",
            "beethoven-bassoon",
            "beethoven-flute",
            "beethoven-clarinet",
            "beethoven-viola",
            "beethoven-oboe",
            "beethoven-cello",
            "beethoven-trumpet",
        ],
        "instruments": {
            "horn": "beethoven-horn",
            "doublebass": "beethoven-doublebass",
            "violin": "beethoven-violin",
            "bassoon": "beethoven-bassoon",
            "flute": "beethoven-flute",
            "clarinet": "beethoven-clarinet",
            "viola": "beethoven-viola",
            "oboe": "beethoven-oboe",
            "cello": "beethoven-cello",
            "trumpet": "beethoven-trumpet",
        },
        "sections": {
            "brass": ["beethoven-horn", "beethoven-trumpet"],
            "strings": [
                "beethoven-doublebass",
                "beethoven-violin",
                "beethoven-viola",
                "beethoven-cello",
            ],
            "woodwinds": [
                "beethoven-bassoon",
                "beethoven-flute",
                "beethoven-clarinet",
                "beethoven-oboe",
            ],
        },
        "piece": "beethoven",
    }

    expected_property_types = {"tracks": dict, "track_audio_property": str}

    run_track_tests(mtrack, expected_attributes, expected_property_types)
    run_multitrack_tests(mtrack)


def test_get_audio_for_instrument():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    y = mtrack.get_audio_for_instrument("violin")
    assert y.shape == (44100,)

    with pytest.raises(ValueError):
        y = mtrack.get_audio_for_instrument("guitar")


def test_get_audio_for_section():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    y = mtrack.get_audio_for_section("strings")
    assert y.shape == (1, 44100)

    with pytest.raises(ValueError):
        y = mtrack.get_audio_for_section("synths")


def test_get_notes_target():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    track_keys = ["beethoven-viola", "beethoven-violin"]
    note_data = mtrack.get_notes_target(track_keys, notes_property="notes")

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.pitches) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array(
            [
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.310204, 4.910204],
                [4.310204, 4.910204],
                [8.359184, 12.004082],
            ]
        ),
    )
    assert np.allclose(
        note_data.pitches,
        np.array([220.0, 329.62755691, 554.36526195, 220.0, 329.62755691, 220.0]),
    )


def test_get_notes_for_instrument():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    note_data = mtrack.get_notes_for_instrument(instrument="violin", notes_property="notes")

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.pitches) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array([[4.284082, 5.271338], [4.284082, 5.271338], [4.284082, 5.271338]]),
    )
    assert np.allclose(note_data.pitches, np.array([220.0, 329.62755691, 554.36526195]))


def test_get_notes_for_section():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/phenicx_anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    note_data = mtrack.get_notes_for_section(section="strings", notes_property="notes")

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.pitches) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array(
            [
                [4.260862, 6.780091],
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.310204, 4.910204],
                [4.310204, 4.910204],
                [4.331995, 6.621655],
                [8.359184, 12.004082],
                [12.167256, 14.038594],
                [12.213696, 13.862268],
                [19.783401, 21.656599],
                [19.841451, 21.462971],
            ]
        ),
    )
    assert np.allclose(
        note_data.pitches,
        np.array(
            [
                55.0,
                220.0,
                329.62755691,
                554.36526195,
                220.0,
                329.62755691,
                110.0,
                220.0,
                51.9130872,
                103.82617439,
                48.9994295,
                97.998859,
            ]
        ),
    )
