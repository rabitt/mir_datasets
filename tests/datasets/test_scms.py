import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import scms
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "Enta_Bhagyamu_2"
    data_home = "tests/resources/mir_datasets/scms"
    dataset = scms.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "Enta_Bhagyamu_2",
        "audio_path": "tests/resources/mir_datasets/" + "scms/SCMS/audio/Enta_Bhagyamu_2.wav",
        "pitch_path": "tests/resources/mir_datasets/"
        + "scms/SCMS/annotations/melody/Enta_Bhagyamu_2.csv",
        "activations_path": "tests/resources/mir_datasets/"
        + "scms/SCMS/annotations/activations/Enta_Bhagyamu_2.lab",
    }

    expected_property_types = {
        "tonic": float,
        "gender": str,
        "artist": str,
        "title": str,
        "train": bool,
        "pitch": annotations.F0Data,
        "activations": annotations.EventData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100,)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/scms"
    dataset = scms.Dataset(data_home)
    track = dataset.track("Enta_Bhagyamu_2")
    jam = track.to_jams()

    f0s = jam.search(namespace="pitch_contour")[0]["data"]
    assert [f0.time for f0 in f0s] == [0.0, 0.0029024943310657597, 0.005804988662131519]
    assert [f0.duration for f0 in f0s] == [0.0, 0.0, 0.0]
    assert [f0.value for f0 in f0s] == [
        {"frequency": 205.34705622484543, "index": 0, "voiced": True},
        {"frequency": 205.5702056921407, "index": 0, "voiced": True},
        {"frequency": 205.5604865955533, "index": 0, "voiced": True},
    ]
    assert [f0.confidence for f0 in f0s] == [None, None, None]

    assert jam["file_metadata"]["artist"] == "Manda Sudharani"
    assert jam["file_metadata"]["title"] == "Enta Bhagyamu"


def test_load_pitch():
    # load a file which exists
    pitch_path = (
        "tests/resources/mir_datasets/scms/SCMS/" + "annotations/melody/Enta_Bhagyamu_2.csv"
    )
    pitch_data = scms.load_pitch(pitch_path)

    # check types
    assert isinstance(pitch_data, annotations.F0Data)
    assert isinstance(pitch_data.times, np.ndarray)
    assert isinstance(pitch_data.frequencies, np.ndarray)
    assert isinstance(pitch_data.voicing, np.ndarray)

    # check values
    assert np.array_equal(
        pitch_data.times, np.array([0.0, 0.0029024943310657597, 0.005804988662131519])
    )
    assert np.array_equal(
        pitch_data.frequencies,
        np.array([205.34705622484543, 205.5702056921407, 205.5604865955533]),
    )
    assert np.array_equal(pitch_data.voicing, np.array([1.0, 1.0, 1.0]))


def test_load_activations():
    activations_path = (
        "tests/resources/mir_datasets/scms/SCMS/" + "annotations/activations/Enta_Bhagyamu_2.lab"
    )
    activations_data = scms.load_activations(activations_path)

    # check types
    assert isinstance(activations_data, annotations.EventData)

    # check values
    assert np.allclose(
        activations_data.intervals,
        np.array([[0.0, 2.5164625850340134], [2.681904761904762, 3.213061224489796]]),
    )
    assert activations_data.events[0] == "singer"
    assert activations_data.events[1] == "singer"
    assert activations_data.event_unit == "open"

    activations_path = (
        "tests/resources/mir_datasets/scms/SCMS/" + "annotations/activations/Enta_Bhagyamu_3.lab"
    )
    activations_data = scms.load_activations(activations_path)
    assert activations_data is None


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/scms"
    dataset = scms.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["Enta_Bhagyamu_2"] == {
        "tonic": 165.1176,
        "artist": "Manda Sudharani",
        "title": "Enta Bhagyamu",
        "gender": "female",
        "train": True,
    }
    with pytest.raises(FileNotFoundError):
        data_home = "a/fake/path"
        dataset = scms.Dataset(data_home)
        metadata = dataset._artists_to_track_mapping


def test_load_audio():
    data_home = "tests/resources/mir_datasets/scms"
    dataset = scms.Dataset(data_home)
    track = dataset.track("Enta_Bhagyamu_2")
    audio_path = track.audio_path
    audio, sr = scms.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
