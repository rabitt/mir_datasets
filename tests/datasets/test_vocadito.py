import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import vocadito
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "1"
    data_home = os.path.normpath("tests/resources/mir_datasets/vocadito")
    dataset = vocadito.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "1",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/vocadito/"),
            "Audio/vocadito_1.wav",
        ),
        "singer_id": "S1",
        "language": "Tagalog",
        "average_pitch_midi": 50,
        "f0_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/vocadito/"),
            "Annotations/F0/vocadito_1_f0.csv",
        ),
        "lyrics_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/vocadito/"),
            "Annotations/Lyrics/vocadito_1_lyrics.txt",
        ),
        "notes_a1_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/vocadito/"),
            "Annotations/Notes/vocadito_1_notesA1.csv",
        ),
        "notes_a2_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/vocadito/"),
            "Annotations/Notes/vocadito_1_notesA2.csv",
        ),
    }

    expected_property_types = {
        "f0": annotations.F0Data,
        "notes_a1": annotations.NoteData,
        "notes_a2": annotations.NoteData,
        "lyrics": list,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["Audio/vocadito_1.wav", "4df70945fd1765eae8c70d252f06c13a"],
        "f0": ["Annotations/F0/vocadito_1_f0.csv", "cfd6617c82cc2fc8911cdac767b8d046"],
        "notesA1": [
            "Annotations/Notes/vocadito_1_notesA1.csv",
            "9634dcad9d13c65b6753d381502bb813",
        ],
        "notesA2": [
            "Annotations/Notes/vocadito_1_notesA2.csv",
            "c13a0248850c262626fe5662c91287b8",
        ],
        "lyrics": [
            "Annotations/Lyrics/vocadito_1_lyrics.txt",
            "37132cbc2daa7a5fb285e34e45eb3376",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (1464660,)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/vocadito"
    default_trackid = "1"
    dataset = vocadito.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    f0s = jam.search(namespace="pitch_contour")[0]["data"][:2]
    assert [f0.time for f0 in f0s] == [0.0, 0.005804988662131519]
    assert [f0.duration for f0 in f0s] == [0.0, 0.0]
    assert [f0.value for f0 in f0s] == [
        {"frequency": 0.0, "index": 0, "voiced": False},
        {"frequency": 0.0, "index": 0, "voiced": False},
    ]
    assert [f0.confidence for f0 in f0s] == [None, None]


def test_load_f0():
    f0_path = "tests/resources/mir_datasets/vocadito/Annotations/F0/vocadito_1_f0.csv"
    f0_data = vocadito.load_f0(f0_path)

    # check types
    assert type(f0_data) == annotations.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.voicing) is np.ndarray

    # check values
    assert len(f0_data.times) == 5722
    assert np.allclose(f0_data.times[:2], np.array([0.0, 0.00580499]), atol=1e-5, rtol=0)
    assert len(f0_data.frequencies) == 5722
    assert np.allclose(f0_data.frequencies[:2], np.array([0.0, 0.0]), atol=1e-5, rtol=0)
    assert len(f0_data.voicing) == 5722
    assert np.allclose(f0_data.voicing[:2], np.array([0.0, 0.0]), atol=1e-5, rtol=0)


def test_load_notes():
    notes_path = "tests/resources/mir_datasets/vocadito/Annotations/Notes/vocadito_1_notesA1.csv"
    note_data = vocadito.load_notes(notes_path)

    # check types
    assert type(note_data) == annotations.NoteData

    # check values
    assert len(note_data.intervals) == 59
    assert np.allclose(
        note_data.intervals[:2],
        np.array([[0.66176871, 0.95201814], [1.01006803, 1.31192744]]),
        atol=1e-5,
        rtol=0.0,
    )
    assert len(note_data.pitches) == 59
    assert np.allclose(note_data.pitches[:2], np.array([143.742, 158.441]))
    assert note_data.confidence is None


def test_load_lyrics():
    lyrics_path = "tests/resources/mir_datasets/vocadito/Annotations/Lyrics/vocadito_1_lyrics.txt"
    lyrics_data = vocadito.load_lyrics(lyrics_path)

    assert lyrics_data == [
        ["ako", "ay", "may", "lobo"],
        ["lumipad", "sa", "langit"],
        ["di", "ko", "na", "nakita"],
        ["pumutok", "na", "pala"],
        [],
        ["sayang", "ang", "pera", "ko"],
        ["binili", "ng", "lobo"],
        ["sa", "pagkain", "sana"],
        ["nabusog", "pa", "ako"],
        [],
        ["sa", "pagkain", "sana"],
        ["nabusog", "pa", "ako"],
    ]


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/vocadito"
    dataset = vocadito.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["1"] == {
        "singer_id": "S1",
        "average_pitch_midi": 50,
        "language": "Tagalog",
    }
