"""Tests for Candombe dataset."""
import numpy as np

from mirdata import annotations
from mirdata.datasets import candombe
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "csic.1995_ansina1_01"
    data_home = "tests/resources/mir_datasets/candombe/"
    dataset = candombe.Dataset(data_home, version="1.0")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "csic.1995_ansina1_01",
        "audio_path": "tests/resources/mir_datasets/candombe/"
        "candombe_audio/csic.1995_ansina1_01.flac",
        "beats_path": "tests/resources/mir_datasets/candombe/"
        "candombe_annotations/with_bar_number/csic.1995_ansina1_01.csv",
    }

    expected_property_types = {"beats": annotations.BeatData, "audio": tuple}

    assert track._track_paths == {
        "audio": [
            "candombe_audio/csic.1995_ansina1_01.flac",
            "fe9bb8edaa46892e4f094a07583ecfb7",
        ],
        "beats": [
            "candombe_annotations/with_bar_number/csic.1995_ansina1_01.csv",
            "3abfa7a3a13225738b39769a5ef0726a",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loadversg functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (12288,)


def test_to_jams():
    default_trackid = "csic.1995_ansina1_01"
    data_home = "tests/resources/mir_datasets/candombe"
    dataset = candombe.Dataset(data_home, version="1.0")
    track = dataset.track(default_trackid)
    jam = track.to_jams()
    beats = jam.search(namespace="beat")[0]["data"]

    assert [beat.time for beat in beats] == [
        0.548571428,
        0.993877551,
        1.461405895,
        1.895328798,
        2.332653061,
        2.809024943,
        3.253650793,
        3.684126984,
        4.115306122,
        4.581587301,
        5.019863945,
        5.473469387,
    ]


def test_load_beats():
    # load a file which exists
    beats_path = (
        "tests/resources/mir_datasets/candombe/candombe_annotations/"
        "with_bar_number/csic.1995_ansina1_01.csv"
    )
    beats_data = candombe.load_beats(beats_path)

    # check types
    assert type(beats_data) == annotations.BeatData
    assert type(beats_data.times) is np.ndarray
    # ... etc

    # check values
    assert np.array_equal(
        beats_data.times,
        np.array(
            [
                0.548571428,
                0.993877551,
                1.461405895,
                1.895328798,
                2.332653061,
                2.809024943,
                3.253650793,
                3.684126984,
                4.115306122,
                4.581587301,
                5.019863945,
                5.473469387,
            ]
        ),
    )
