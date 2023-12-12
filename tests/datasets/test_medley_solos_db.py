import os

from mirdata.datasets import medley_solos_db
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "d07b1fc0-567d-52c2-fef4-239f31c9d40e"
    data_home = os.path.normpath("tests/resources/mir_datasets/medley_solos_db")
    dataset = medley_solos_db.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "d07b1fc0-567d-52c2-fef4-239f31c9d40e",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/medley_solos_db/"),
            "audio/Medley-solos-DB_validation-3_d07b1fc0-567d-52c2-fef4-239f31c9d40e.wav",
        ),
        "instrument": "flute",
        "instrument_id": 3,
        "song_id": 210,
        "subset": "validation",
    }

    expected_property_types = {"audio": tuple}

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert y.shape == (65536,)
    assert sr == 22050


def test_to_jams():
    data_home = "tests/resources/mir_datasets/medley_solos_db"
    dataset = medley_solos_db.Dataset(data_home)
    track = dataset.track("d07b1fc0-567d-52c2-fef4-239f31c9d40e")
    jam = track.to_jams()

    assert jam["sandbox"]["instrument"] == "flute"
    assert jam["sandbox"]["instrument_id"] == 3
    assert jam["sandbox"]["song_id"] == 210
    assert jam["sandbox"]["subset"] == "validation"
