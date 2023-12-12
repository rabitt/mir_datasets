"""Tests for example dataset."""
import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import example
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "some_id"
    data_home = "tests/resources/mir_datasets/dataset"
    dataset = example.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "some_id",
        "audio_path": "tests/resources/mir_datasets/example/" + "Wavfile/some_id.wav",
        "song_id": "some_id",
        "annotation_path": "tests/resources/mir_datasets/example/annotation/some_id.pv",
    }

    expected_property_types = {"annotation": annotations.XData}

    assert track._track_paths == {
        "audio": ["Wavfile/some_id.wav", "278ae003cb0d323e99b9a643c0f2eeda"],
        "annotation": ["Annotation/some_id.pv", "0d93a011a9e668fd80673049089bbb14"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (44100 * 2,)


def test_to_jams():
    default_trackid = "some_id"
    data_home = "tests/resources/mir_datasets/dataset"
    dataset = example.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    annotations = jam.search(namespace="annotation")[0]["data"]
    assert [annotation.time for annotation in annotations] == [0.027, 0.232]
    assert [annotation.duration for annotation in annotations] == [
        0.20500000000000002,
        0.736,
    ]
    # ... etc


def test_load_annotation():
    # load a file which exists
    annotation_path = "tests/resources/mir_datasets/dataset/Annotation/some_id.pv"
    annotation_data = example.load_annotation(annotation_path)

    # check types
    assert type(annotation_data) == annotations.XData
    assert type(annotation_data.times) is np.ndarray
    # ... etc

    # check values
    assert np.array_equal(annotation_data.times, np.array([0.016, 0.048]))
    # ... etc


def test_metadata():
    data_home = "tests/resources/mir_datasets/dataset"
    dataset = example.Dataset(data_home, version="test")
    metadata = dataset._metadata
    assert metadata["some_id"] == "something"
