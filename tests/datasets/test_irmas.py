import os

from mirdata.datasets import irmas
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "1"
    default_trackid_train = "0189__2"
    data_home = os.path.normpath("tests/resources/mir_datasets/irmas")
    dataset = irmas.Dataset(data_home)
    track = dataset.track(default_trackid)
    track_train = dataset.track(default_trackid_train)
    expected_attributes = {
        "annotation_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/irmas/"),
            "IRMAS-TestingData-Part1/Part1/02 - And The Body Will Die-8.txt",
        ),
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/irmas/"),
            "IRMAS-TestingData-Part1/Part1/02 - And The Body Will Die-8.wav",
        ),
        "track_id": "1",
        "predominant_instrument": None,
        "genre": None,
        "drum": None,
        "split": "test",
        "train": False,
    }
    expected_attributes_train = {
        "annotation_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/irmas/"),
            "IRMAS-TrainingData/cla/[cla][cla]0189__2.wav",
        ),
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/irmas/"),
            "IRMAS-TrainingData/cla/[cla][cla]0189__2.wav",
        ),
        "track_id": "0189__2",
        "predominant_instrument": "cla",
        "genre": "cla",
        "drum": None,
        "split": "train",
        "train": True,
    }

    expected_property_test_types = {"instrument": list, "audio": tuple}

    run_track_tests(track, expected_attributes, expected_property_test_types)
    run_track_tests(track_train, expected_attributes_train, expected_property_test_types)

    audio, sr = track.audio
    assert sr == 44100
    assert len(audio) == 2
    assert len(audio[1, :]) == 88200


def test_to_jams():
    # Training samples
    default_trackid_train = "0189__2"
    data_home = "tests/resources/mir_datasets/irmas"
    dataset = irmas.Dataset(data_home)
    track_train = dataset.track(default_trackid_train)
    jam_train = track_train.to_jams()

    # Validate Mridangam schema
    assert jam_train.validate()

    # Test the training data parsers
    assert jam_train.sandbox["instrument"] == ["cla"]
    assert jam_train.sandbox["genre"] == "cla"
    assert jam_train.sandbox["train"] is True

    # Testing samples
    default_trackid_test = "1"
    data_home = "tests/resources/mir_datasets/irmas"
    dataset = irmas.Dataset(data_home)
    track_test = dataset.track(default_trackid_test)
    jam_test = track_test.to_jams()

    # Validate Mridangam schema
    assert jam_test.validate()

    # Test the testing genre parser
    assert jam_test.sandbox["instrument"] == ["gel", "voi"]
    assert jam_test.sandbox["train"] is False


def test_load_pred_inst():
    # Training samples
    pred_inst_audio_train = (
        "tests/resources/mir_datasets/irmas/IRMAS-TrainingData/cla/" + "[cla][cla]0189__2.wav"
    )

    pred_inst_train = os.path.basename(os.path.dirname(pred_inst_audio_train))
    assert pred_inst_train == "cla"

    # Testing samples
    pred_inst_ann_path_test = (
        "tests/resources/mir_datasets/irmas/IRMAS-TestingData-Part1/Part1/"
        + "02 - And The Body Will Die-8.txt"
    )
    pred_inst_data_test = irmas.load_pred_inst(pred_inst_ann_path_test)
    assert type(pred_inst_data_test) is list
    assert type(pred_inst_data_test[0]) is str
    assert pred_inst_data_test == ["gel", "voi"]
    assert irmas.load_pred_inst(None) is None
