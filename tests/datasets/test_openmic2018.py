import os

from mirdata import download_utils
from mirdata.datasets import openmic2018
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "000046_3840"
    data_home = os.path.normpath("tests/resources/mir_datasets/openmic2018")

    dataset = openmic2018.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/openmic2018/"),
            "audio/000/000046_3840.ogg",
        ),
        "vggish_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/openmic2018/"),
            "vggish/000/000046_3840.json",
        ),
        "track_id": "000046_3840",
        "title": "Yosemite",
        "artist": "Nicky Cook",
        "instruments": {"clarinet": 0.1710499999999999, "flute": 0.0, "trumpet": 0.0},
        "url": "http://freemusicarchive.org/music/Chris_and_Nicky_Andrews/Niris/Yosemite",
        "start_time": 3.84,
        "genres": ["Experimental Pop", "Singer-Songwriter"],
        "split": "split01_train",
    }

    expected_property_types = {
        "artist": str,
        "title": str,
        "audio_path": str,
        "audio": tuple,
        "vggish": tuple,
        "instruments": dict,
        "url": str,
        "start_time": float,
        "genres": list,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_classmap():
    data_home = "tests/resources/mir_datasets/openmic2018"
    dataset = openmic2018.Dataset(data_home, version="test")

    ref_instruments = {
        "accordion": 0,
        "banjo": 1,
        "bass": 2,
        "cello": 3,
        "clarinet": 4,
        "cymbals": 5,
        "drums": 6,
        "flute": 7,
        "guitar": 8,
        "mallet_percussion": 9,
        "mandolin": 10,
        "organ": 11,
        "piano": 12,
        "saxophone": 13,
        "synthesizer": 14,
        "trombone": 15,
        "trumpet": 16,
        "ukulele": 17,
        "violin": 18,
        "voice": 19,
    }

    # verify that the loader works
    assert ref_instruments == dataset._class_map

    # and that our baked in mapping works
    assert ref_instruments == openmic2018.INSTRUMENTS
