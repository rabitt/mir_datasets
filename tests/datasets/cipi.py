import logging
import os

try:
    import music21
except ImportError:
    logging.error(
        "In order to test haydn_op20 you must have music21 installed. "
        "Please reinstall mirdata using `pip install 'mirdata[haydn_op20] and re-run the tests."
    )
    raise ImportError

    raise ImportError
from mirdata.annotations import ChordData, KeyData
from mirdata.datasets import cipi
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "c-1"
    data_home = os.path.normpath("tests/resources/mir_datasets/cipi")
    dataset = cipi.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "c-1",
        "annotations": {
            "lh_fingering": [
                "ArGNNThumb-s/lh/c-1.pt",
                "ecb8a0dcb70a4331708baf9141cedfd1",
            ],
            "rh_fingering": [
                "ArGNNThumb-s/rh/c-1.pt",
                "cdb1ba84bc21572c35d299291fca8442",
            ],
            "expressiviness": ["virtuoso/c-1.pt", "fb6eb47c9ee21051559325e217b9b0a1"],
            "notes": ["k/c-1.pt", "9816f5c88488925c019283e29f00b536"],
        },
    }

    expected_property_types = {
        "title": str,
        "book": str,
        "URI": str,
        "composer": str,
        "musicxml_paths": list,
        "difficulty_annotation": int,
        "scores": list,
        "fingering": tuple,
        "expressiviness": str,
        "notes": str,
    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jam():
    data_home = os.path.normpath("tests/resources/mir_datasets/cipi")
    dataset = cipi.Dataset(data_home)
    track = dataset.track("c-1")
    jam = track.to_jams()
    assert (
        jam["file_metadata"]["title"] == "Piano Sonata f minor op. 2,1"
    ), "title does not match expected"
    assert (
        jam["file_metadata"]["artist"] == "LUDWIG VAN BEETHOVEN"
    ), "artist does not match expected"
    assert jam["sandbox"]["composer"] == "LUDWIG VAN BEETHOVEN", "composer does not match expected"
    assert jam["sandbox"]["book"] == "Piano Sonatas, Volume I", "book does not match expected"
    assert (
        jam["sandbox"]["URI"]
        == "https://www.henle.de/en/detail/?Title=Piano+Sonatas%2C+Volume+I_32"
    ), "book does not match expected"
    assert (
        jam["sandbox"]["difficulty_annotation"] == 6
    ), "difficulty_annotation does not match expected"
    assert jam["file_metadata"]["duration"] == 0, "duration does not match expected"
    assert jam["sandbox"]["musicxml_paths"] == [
        "craig_files/beethoven-piano-sonatas-master/kern/sonata01-1.musicxml",
        "craig_files/beethoven-piano-sonatas-master/kern/sonata01-2.musicxml",
        "craig_files/beethoven-piano-sonatas-master/kern/sonata01-3.musicxml",
        "craig_files/beethoven-piano-sonatas-master/kern/sonata01-4.musicxml",
    ], "musicxml_paths does not match expected"


def test_load_score():
    path = os.path.normpath("craig_files/beethoven-piano-sonatas-master/kern/sonata01-1.musicxml")
    data_home = os.path.normpath("tests/resources/mir_datasets/cipi")
    score = cipi.load_score(path, data_home)
    assert isinstance(score, music21.stream.Score)
    assert len(score.parts) == 2
