import os
from typing import List

import numpy as np

from mirdata.datasets import fma_keys
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "10"
    data_home = os.path.normpath("tests/resources/mir_datasets/fma_keys")
    dataset = fma_keys.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "10",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/fma_keys/"),
            "000/000010.mp3",
        ),
        "key": "F#",
        "mode": "Major",
        "key_number": 6,
        "mode_number": 1,
        "spotify_uri": "spotify:track:66381EvBZ6e3RXzYATpGmN",
    }

    expected_property_types = {
        "spotify_uri": str,
        "key": str,
        "mode": str,
        "key_number": int,
        "mode_number": int,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["000/000010.mp3", "b1ca8926d40bbb97fb1f3a728ca55aa6"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (88200,)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/fma_keys"
    default_trackid = "10"
    dataset = fma_keys.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam["sandbox"]["spotify_uri"] == "spotify:track:66381EvBZ6e3RXzYATpGmN"
    assert jam["sandbox"]["key"] == "F#"
    assert jam["sandbox"]["mode"] == "Major"
    assert jam["sandbox"]["key_number"] == 6
    assert jam["sandbox"]["mode_number"] == 1


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/fma_keys"
    dataset = fma_keys.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["10"] == {
        "spotify_uri": "spotify:track:66381EvBZ6e3RXzYATpGmN",
        "key": "F#",
        "mode": "Major",
        "key_number": 6,
        "mode_number": 1,
    }

    assert metadata["141"] == {
        "spotify_uri": "spotify:track:7f0KQDOB9khm9ZtuWjjtre",
        "key": "F",
        "mode": "Major",
        "key_number": 5,
        "mode_number": 1,
    }
