import os

from mirdata import annotations
from mirdata.datasets import rwc_jazz
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "RM-J004"
    data_home = os.path.normpath("tests/resources/mir_datasets/rwc_jazz")
    dataset = rwc_jazz.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "RM-J004",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "audio/rwc-j-m01/4.wav",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "annotations/AIST.RWC-MDB-J-2001.CHORUS/RM-J004.CHORUS.TXT",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "annotations/AIST.RWC-MDB-J-2001.BEAT/RM-J004.BEAT.TXT",
        ),
        "piece_number": "No. 4",
        "suffix": "M01",
        "track_number": "Tr. 04",
        "title": "Crescent Serenade (Piano Solo)",
        "artist": "Makoto Nakamura",
        "duration": 167,
        "variation": "Instrumentation 1",
        "instruments": "Pf",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "sections": annotations.SectionData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/rwc_jazz"
    dataset = rwc_jazz.Dataset(data_home)
    track = dataset.track("RM-J004")
    jam = track.to_jams()

    beats = jam.search(namespace="beat")[0]["data"]
    assert [beat.time for beat in beats] == [
        0.05,
        0.86,
        1.67,
        2.48,
        3.29,
        4.1,
        4.91,
        5.72,
        6.53,
        7.34,
    ]
    assert [beat.duration for beat in beats] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert [beat.value for beat in beats] == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    assert [beat.confidence for beat in beats] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    segments = jam.search(namespace="segment")[0]["data"]
    assert [segment.time for segment in segments] == [0.05, 6.53, 152.06]
    assert [segment.duration for segment in segments] == [
        6.48,
        13.099999999999998,
        13.319999999999993,
    ]
    assert [segment.value for segment in segments] == [
        "nothing",
        "chorus A",
        "chorus B",
    ]
    assert [segment.confidence for segment in segments] == [None, None, None]

    assert jam["file_metadata"]["title"] == "Crescent Serenade (Piano Solo)"
    assert jam["file_metadata"]["artist"] == "Makoto Nakamura"


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/rwc_jazz"
    dataset = rwc_jazz.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["RM-J004"] == {
        "piece_number": "No. 4",
        "suffix": "M01",
        "track_number": "Tr. 04",
        "title": "Crescent Serenade (Piano Solo)",
        "artist": "Makoto Nakamura",
        "duration": 167,
        "variation": "Instrumentation 1",
        "instruments": "Pf",
    }
    assert metadata["RM-J044"] == {
        "piece_number": "No. 44",
        "suffix": "M04",
        "track_number": "Tr. 09",
        "title": "Joyful, Joyful, We Adore Thee",
        "artist": ("K’s Band"),
        "duration": 270,
        "variation": "Style (Free jazz)",
        "instruments": "Pf & Bs & Dr & Gt & Ts & Fl & Bar",
    }
