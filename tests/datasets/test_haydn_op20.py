import logging
import os

try:
    import music21
except ImportError:
    logging.error(
        "In order to test haydn_op20 you must have music21 installed. "
        "Please reinstall mirdata using `pip install 'mirdata[haydn_op20] and re-run the tests."
    )
    raise ImportError

import numpy as np

from mirdata.annotations import ChordData, KeyData
from mirdata.datasets import haydn_op20
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0"
    data_home = os.path.normpath("tests/resources/mir_datasets/haydn_op20")
    dataset = haydn_op20.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "humdrum_annotated_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/haydn_op20/"),
            "op20n1-01.hrm",
        ),
        "title": "op20n1-01",
        "track_id": "0",
    }

    expected_property_types = {
        "duration": int,
        "chords": ChordData,
        "chords_music21": list,
        "roman_numerals": list,
        "keys": KeyData,
        "keys_music21": list,
        "score": music21.stream.Score,
        "midi_path": str,
    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jam():
    data_home = os.path.normpath("tests/resources/mir_datasets/haydn_op20")
    dataset = haydn_op20.Dataset(data_home)
    track = dataset.track("0")
    jam = track.to_jams()
    assert jam["file_metadata"]["title"] == "op20n1-01", "title does not match expected"
    assert jam["file_metadata"]["duration"] == 644, "duration does not match expected"
    assert jam["sandbox"]["humdrum_annotated_path"] == os.path.join(
        os.path.normpath("tests/resources/mir_datasets/haydn_op20/"), "op20n1-01.hrm"
    ), "duration does not match expected"
    assert jam["sandbox"]["midi_path"] == os.path.join(
        os.path.normpath("tests/resources/mir_datasets/haydn_op20/"), "op20n1-01.midi"
    ), "duration does not match expected"
    assert isinstance(jam["sandbox"]["chords_music21"], list)
    assert jam["sandbox"]["chords_music21"][0]["time"] == 0
    assert jam["sandbox"]["chords_music21"][0]["chord"] == "Eb-major triad"
    assert isinstance(jam["sandbox"]["keys_music21"], list)
    assert jam["sandbox"]["keys_music21"][0]["time"] == 0
    assert isinstance(jam["sandbox"]["keys_music21"][0]["key"], music21.key.Key)
    assert isinstance(jam["sandbox"]["roman_numerals"], list)
    assert jam["sandbox"]["roman_numerals"][0]["time"] == 0
    assert jam["sandbox"]["roman_numerals"][0]["roman_numeral"] == "I"

    chord_data = jam["sandbox"]["chord"]
    assert type(chord_data) == ChordData
    assert type(chord_data.intervals) == np.ndarray
    assert type(chord_data.labels) == list
    assert np.array_equal(chord_data.intervals[:, 0], np.array([0.0, 364.0, 392.0, 644.0]))
    assert np.array_equal(chord_data.intervals[:, 1], np.array([363.0, 391.0, 643.0, 644.0]))
    assert np.array_equal(
        chord_data.labels,
        np.array(
            [
                "Eb-major triad",
                "Bb-dominant seventh chord",
                "Eb-major triad",
                "F-dominant seventh chord",
            ]
        ),
    )
    assert haydn_op20.load_chords(None) is None

    key_data = jam["sandbox"]["key"]
    assert type(key_data) == KeyData
    assert type(key_data.intervals) == np.ndarray

    assert np.array_equal(key_data.intervals[:, 0], np.array([0.0, 644.0]))
    assert np.array_equal(key_data.intervals[:, 1], np.array([643.0, 644.0]))
    assert np.array_equal(key_data.keys, ["Eb:major", "Bb:major"])

    assert haydn_op20.load_key(None) is None


def test_load_score():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    score = haydn_op20.load_score(path)
    assert isinstance(score, music21.stream.Score)
    assert len(score.parts) == 4


def test_load_key():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"

    key_data = haydn_op20.load_key(path)
    assert type(key_data) == KeyData
    assert type(key_data.intervals) == np.ndarray

    assert np.array_equal(key_data.intervals[:, 0], np.array([0.0, 644.0]))
    assert np.array_equal(key_data.intervals[:, 1], np.array([643.0, 644.0]))
    assert np.array_equal(key_data.keys, ["Eb:major", "Bb:major"])

    assert haydn_op20.load_key(None) is None

    key_music21 = haydn_op20.load_key_music21(path)
    assert isinstance(key_music21, list)
    assert len(key_music21) == 4
    assert key_music21[0]["time"] == 0
    assert key_music21[-1]["time"] == 644
    assert isinstance(key_music21[0]["key"], music21.key.Key)


def test_load_chords():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"

    chord_data = haydn_op20.load_chords(path)

    assert type(chord_data) == ChordData
    assert type(chord_data.intervals) == np.ndarray
    assert type(chord_data.labels) == list
    assert np.array_equal(chord_data.intervals[:, 0], np.array([0.0, 364.0, 392.0, 644.0]))
    assert np.array_equal(chord_data.intervals[:, 1], np.array([363.0, 391.0, 643.0, 644.0]))
    assert np.array_equal(
        chord_data.labels,
        np.array(
            [
                "Eb-major triad",
                "Bb-dominant seventh chord",
                "Eb-major triad",
                "F-dominant seventh chord",
            ]
        ),
    )
    assert haydn_op20.load_chords(None) is None

    chords = haydn_op20.load_chords_music21(path)
    assert isinstance(chords, list)
    assert len(chords) == 4
    assert chords[0]["time"] == 0
    assert chords[-1]["time"] == 644
    assert chords[0]["chord"] == "Eb-major triad"
    assert chords[-1]["chord"] == "F-dominant seventh chord"


def test_load_roman_numerals():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    roman_numerals = haydn_op20.load_roman_numerals(path)
    assert isinstance(roman_numerals, list)
    assert len(roman_numerals) == 4
    assert roman_numerals[0]["time"] == 0
    assert roman_numerals[-1]["time"] == 644
    assert roman_numerals[0]["roman_numeral"] == "I"
    assert roman_numerals[-1]["roman_numeral"] == "V43/V"


def test_load_midi_path():
    path = "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm"
    midi_path = haydn_op20.convert_and_save_to_midi(path)
    assert isinstance(midi_path, str)
    assert midi_path == "tests/resources/mir_datasets/haydn_op20/op20n1-01.midi"
