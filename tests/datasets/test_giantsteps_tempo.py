import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import giantsteps_tempo
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "113"
    data_home = os.path.normpath("tests/resources/mir_datasets/giantsteps_tempo")
    dataset = giantsteps_tempo.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/giantsteps_tempo/"),
            "audio/28952.LOFI.mp3",
        ),
        "annotation_v1_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/giantsteps_tempo/"),
            "giantsteps-tempo-dataset-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations/jams/28952.LOFI.jams",
        ),
        "annotation_v2_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/giantsteps_tempo/"),
            "giantsteps-tempo-dataset-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations_v2/jams/28952.LOFI.jams",
        ),
        "title": "28952",
        "track_id": "113",
    }

    expected_property_types = {
        "tempo": annotations.TempoData,
        "tempo_v2": annotations.TempoData,
        "genre": str,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 22050, "sample rate {} is not 22050".format(sr)
    print(audio.shape)


def test_load_genre():
    genre_path = (
        "tests/resources/mir_datasets/giantsteps_tempo/giantsteps-tempo-dataset"
        "-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations/jams/28952.LOFI.jams"
    )

    genre_data = giantsteps_tempo.load_genre(genre_path)

    assert type(genre_data) == str

    assert genre_data == "trance"

    assert giantsteps_tempo.load_genre(None) is None


def test_load_tempo():
    tempo_path = (
        "tests/resources/mir_datasets/giantsteps_tempo/giantsteps-tempo-dataset"
        "-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations/jams/28952.LOFI.jams"
    )

    tempo_data = giantsteps_tempo.load_tempo(tempo_path)

    assert type(tempo_data) == annotations.TempoData
    print(tempo_data.intervals)
    assert np.array_equal(tempo_data.intervals[:, 0], np.array([0.0]))
    assert np.array_equal(tempo_data.intervals[:, 1], np.array([120.0]))
    assert np.array_equal(tempo_data.value, np.array([137.6]))
    assert giantsteps_tempo.load_tempo(None) is None
