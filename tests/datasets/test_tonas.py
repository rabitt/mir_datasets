import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import tonas
from tests.test_utils import run_track_tests

TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets/tonas")


def test_track():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "singer": "En el barrio de Triana",
        "style": "Debla",
        "title": "Antonio Mairena",
        "tuning_frequency": 451.0654725341684,
        "f0_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonas/"),
            "Deblas/01-D_AMairena.f0.Corrected",
        ),
        "notes_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonas/"),
            "Deblas/01-D_AMairena.notes.Corrected",
        ),
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonas/"),
            "Deblas/01-D_AMairena.wav",
        ),
        "track_id": "01-D_AMairena",
    }

    expected_property_types = {
        "f0": annotations.F0Data,
        "f0_automatic": annotations.F0Data,
        "f0_corrected": annotations.F0Data,
        "notes": annotations.NoteData,
        "audio": tuple,
        "singer": str,
        "style": str,
        "title": str,
        "tuning_frequency": float,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate cante100 jam schema
    assert jam.validate()

    # Validate melody
    f0 = jam.search(namespace="pitch_contour")[0]["data"]
    assert [note.time for note in f0] == [0.197, 0.209, 0.221, 0.232]
    assert [note.duration for note in f0] == [0.0, 0.0, 0.0, 0.0]
    assert [note.value for note in f0] == [
        {"index": 0, "frequency": 0.0, "voiced": False},
        {"index": 0, "frequency": 379.299, "voiced": True},
        {"index": 0, "frequency": 379.299, "voiced": True},
        {"index": 0, "frequency": 379.299, "voiced": True},
    ]
    print([note.confidence for note in f0])
    assert [note.confidence for note in f0] == [3.09e-06, 2.86e-06, 7.15e-06, 1.545e-05]

    # Validate note transciption
    notes = jam.search(namespace="note_hz")[0]["data"]
    assert [note.time for note in notes] == [0.216667, 0.65, 2.183333, 2.566667]
    assert [note.duration for note in notes] == [
        0.433333,
        1.016667,
        0.3833329999999999,
        0.3333330000000001,
    ]
    assert [note.value for note in notes] == [
        388.8382625732775,
        411.9597888711769,
        388.8382625732775,
        411.9597888711769,
    ]
    assert [note.confidence for note in notes] == [None, None, None, None]


def test_load_melody():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    f0_path = track.f0_path
    f0_data_corrected = tonas.load_f0(f0_path, True)
    f0_data_automatic = tonas.load_f0(f0_path, False)

    # check types
    assert type(f0_data_corrected) == annotations.F0Data
    assert type(f0_data_corrected.times) is np.ndarray
    assert type(f0_data_corrected.frequencies) is np.ndarray
    assert type(f0_data_corrected.voicing) is np.ndarray
    assert type(f0_data_corrected._confidence) is np.ndarray

    assert type(f0_data_automatic) == annotations.F0Data
    assert type(f0_data_automatic.times) is np.ndarray
    assert type(f0_data_automatic.frequencies) is np.ndarray
    assert type(f0_data_corrected.voicing) is np.ndarray
    assert type(f0_data_automatic._confidence) is np.ndarray

    # check values
    assert np.array_equal(f0_data_corrected.times, np.array([0.197, 0.209, 0.221, 0.232]))
    assert np.array_equal(
        f0_data_corrected.frequencies, np.array([0.000, 379.299, 379.299, 379.299])
    )
    assert np.array_equal(f0_data_corrected.voicing, np.array([0.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(
        f0_data_corrected._confidence,
        np.array([3.090e-06, 0.00000286, 0.00000715, 0.00001545]),
    )

    # check values
    assert np.array_equal(f0_data_automatic.times, np.array([0.197, 0.209, 0.221, 0.232]))
    assert np.array_equal(f0_data_automatic.frequencies, np.array([0.000, 0.000, 143.918, 143.918]))
    assert np.array_equal(f0_data_automatic.voicing, np.array([0.0, 0.0, 1.0, 1.0]))
    assert np.array_equal(
        f0_data_automatic._confidence,
        np.array([3.090e-06, 2.860e-06, 0.00000715, 0.00001545]),
    )


def test_load_notes():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    notes_path = track.notes_path
    notes_data = tonas.load_notes(notes_path)
    tuning_frequency = tonas._load_tuning_frequency(notes_path)

    # check types
    assert type(notes_data) == annotations.NoteData
    assert type(notes_data.intervals) is np.ndarray
    assert type(notes_data.pitches) is np.ndarray
    assert type(notes_data.confidence) is np.ndarray
    assert type(tuning_frequency) is float

    # check tuning frequency
    assert tuning_frequency == 451.0654725341684

    # check values
    assert np.array_equal(
        notes_data.intervals[:, 0], np.array([0.216667, 0.65, 2.183333, 2.566667])
    )
    assert np.array_equal(notes_data.intervals[:, 1], np.array([0.65, 1.666667, 2.566666, 2.9]))
    assert np.array_equal(
        notes_data.pitches,
        np.array([388.8382625732775, 411.9597888711769, 388.8382625732775, 411.9597888711769]),
    )
    assert np.array_equal(notes_data.confidence, np.array([0.018007, 0.010794, 0.00698, 0.03265]))


def test_load_audio():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    audio_path = track.audio_path
    audio, sr = tonas.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray


def test_metadata():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    metadata = dataset._metadata
    assert metadata[default_trackid] == {
        "title": "En el barrio de Triana",
        "style": "Debla",
        "singer": "Antonio Mairena",
    }
