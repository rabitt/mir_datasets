import os
import pytest
from mirdata.datasets import emvd
from tests.test_utils import run_track_tests

DEFAULT_TRACK_ID = "Singer9_BlackShriek_Mid_i"
DATA_HOME = os.path.normpath("tests/resources/mir_datasets/emvd")


def test_track():
    dataset = emvd.Dataset(DATA_HOME, version="test")
    track = dataset.track(DEFAULT_TRACK_ID)

    expected_attributes = {
        "track_id": DEFAULT_TRACK_ID,
        "audio_path": os.path.join(
            DATA_HOME,
            f"audio/{DEFAULT_TRACK_ID}.wav",
        ),
    }

    expected_property_types = {
        "audio": tuple,
        "singer_id": str,
        "vocalization_type": str,
        "name": str,
        "range": str,
        "vowel": str,
        "rank": str,
        "duration": float,
    }

    assert track._track_paths == {
        "audio": [f"audio/{DEFAULT_TRACK_ID}.wav", "f94c181391edb98b2177db6c0bb02a2d"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (95979,)


def test_to_jams():
    dataset = emvd.Dataset(DATA_HOME, version="test")
    track = dataset.track(DEFAULT_TRACK_ID)
    jam = track.to_jams()

    assert jam["file_metadata"]["duration"] == track.duration


def test_load_metadata():
    dataset = emvd.Dataset(DATA_HOME, version="test")
    metadata = dataset._metadata
    assert metadata[DEFAULT_TRACK_ID] == {
        "singer_id": "9",
        "type": "Technique",
        "name": "BlackShriek",
        "range": "Mid",
        "vowel": "i",
        "authors_rank": "2",
        "duration(s)": "1,9995625",
    }

    with pytest.raises(FileNotFoundError):
        dataset = emvd.Dataset(data_home="invalid", version="test")
        dataset._metadata()


def test_load_singer_metadata():
    dataset = emvd.Dataset(DATA_HOME, version="test")
    track = dataset.track(DEFAULT_TRACK_ID)

    assert dataset.get_singer_info(track.singer_id) == {
        "gender": "M",
        "status": "Non-professional",
        "recording": "Onsite",
        "distance_to_microphone(cm)": "2",
        "microphone": "SM58",
        "audio_interface": "Focusrite 6i6",
        "DAW": "Protools",
        "ClearVoice_High": "2",
        "ClearVoice_Mid": "3",
        "ClearVoice_Low": "2",
        "BlackShriek_High": "4",
        "BlackShriek_Mid": "4",
        "DeathGrowl_Mid": "4",
        "DeathGrowl_Low": "3",
        "HardcoreScream_High": "1",
        "HardcoreScream_Mid": "2",
        "HardcoreScream_Low": "3",
        "GrindInhale": "3",
        "PigSqueal": "1",
        "DeepGutturals": "1",
        "TunnelThroat": "1",
    }

    with pytest.raises(FileNotFoundError):
        dataset = emvd.Dataset(data_home="invalid", version="test")
        dataset.get_singer_info(track.singer_id)


def test_load_split_kfolds():
    dataset = emvd.Dataset(DATA_HOME, version="test")

    expected_splits = {
        "split0": "train",
        "split1": "eval",
        "split2": "unused",
        "split3": "valid",
    }

    for split_id in expected_splits:
        target = {
            "train": list(),
            "eval": list(),
            "valid": list(),
            "unused": list(),
        }

        target[expected_splits[split_id]].append(DEFAULT_TRACK_ID)

        assert dataset.get_track_splits(split_id) == target

    with pytest.raises(ValueError):
        dataset.get_track_splits("split4")

    with pytest.raises(FileNotFoundError):
        dataset = emvd.Dataset(data_home="invalid", version="test")
        dataset.get_track_splits()
