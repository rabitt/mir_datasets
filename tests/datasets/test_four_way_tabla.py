import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import four_way_tabla
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "AHK_solo-tintal-1"
    data_home = os.path.normpath("tests/resources/mir_datasets/four_way_tabla")
    dataset = four_way_tabla.Dataset(data_home)
    track = dataset.track(default_trackid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/four_way_tabla/"),
            "4way-tabla-ismir21-dataset/train/audios/AHK_solo-tintal-1.wav",
        ),
        "onsets_b_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/four_way_tabla/"),
            "4way-tabla-ismir21-dataset/train/onsets/b/AHK_solo-tintal-1.onsets",
        ),
        "onsets_d_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/four_way_tabla/"),
            "4way-tabla-ismir21-dataset/train/onsets/d/AHK_solo-tintal-1.onsets",
        ),
        "onsets_rb_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/four_way_tabla/"),
            "4way-tabla-ismir21-dataset/train/onsets/rb/AHK_solo-tintal-1.onsets",
        ),
        "onsets_rt_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/four_way_tabla/"),
            "4way-tabla-ismir21-dataset/train/onsets/rt/AHK_solo-tintal-1.onsets",
        ),
        "track_id": "AHK_solo-tintal-1",
        "train": True,
    }

    expected_property_types = {
        "onsets_b": annotations.BeatData,
        "onsets_d": annotations.BeatData,
        "onsets_rb": annotations.BeatData,
        "onsets_rt": annotations.BeatData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (69152,)


def test_get_onsets():
    default_trackid = "AHK_solo-tintal-1"
    data_home = "tests/resources/mir_datasets/four_way_tabla"
    dataset = four_way_tabla.Dataset(data_home)
    track = dataset.track(default_trackid)

    loaded_b = track.onsets_b
    parsed_b = four_way_tabla.load_onsets(track.onsets_b_path)

    # Check types
    assert type(parsed_b) == annotations.BeatData
    assert type(parsed_b.times) is np.ndarray
    assert type(parsed_b.positions) is np.ndarray
    assert type(loaded_b) == annotations.BeatData
    assert type(loaded_b.times) is np.ndarray
    assert type(loaded_b.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_b.times, np.array([2.395, 2.885, 65.635]))
    assert np.array_equal(parsed_b.positions, np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(loaded_b.times, np.array([2.395, 2.885, 65.635]))
    assert np.array_equal(loaded_b.positions, np.array([0.0, 0.0, 0.0]))
    assert four_way_tabla.load_onsets(None) is None

    track = dataset.track("binati_SRC")
    parsed_onsets = track.onsets_rt
    assert parsed_onsets is None


def test_to_jams():
    default_trackid = "AHK_solo-tintal-1"
    data_home = "tests/resources/mir_datasets/four_way_tabla"
    dataset = four_way_tabla.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate Four-Way Tabla schema
    assert jam.validate()

    # Sama
    onsets = jam.search(namespace="beat")[0]["data"]
    assert len(onsets) == 3
    assert [onset.time for onset in onsets] == [2.395, 2.885, 65.635]
    assert [onset.duration for onset in onsets] == [0.0, 0.0, 0.0]
    assert [onset.value for onset in onsets] == [0.0, 0.0, 0.0]
    assert [onset.confidence for onset in onsets] == [None, None, None]


def test_load_audio():
    default_trackid = "AHK_solo-tintal-1"
    data_home = "tests/resources/mir_datasets/four_way_tabla"
    dataset = four_way_tabla.Dataset(data_home)
    track = dataset.track(default_trackid)
    audio_path = track.audio_path
    audio, sr = four_way_tabla.load_audio(audio_path)
    assert sr == 44100
    assert audio.shape == (69152,)
    assert type(audio) is np.ndarray
