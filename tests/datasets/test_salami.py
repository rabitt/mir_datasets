import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import salami
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "2"
    data_home = os.path.normpath("tests/resources/mir_datasets/salami")
    dataset = salami.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "2",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/salami/"), "audio/2.mp3"
        ),
        "sections_annotator1_uppercase_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/salami/"),
            "salami-data-public-hierarchy-corrections/annotations/2/parsed/textfile1_uppercase.txt",
        ),
        "sections_annotator1_lowercase_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/salami/"),
            "salami-data-public-hierarchy-corrections/annotations/2/parsed/textfile1_lowercase.txt",
        ),
        "sections_annotator2_uppercase_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/salami/"),
            "salami-data-public-hierarchy-corrections/annotations/2/parsed/textfile2_uppercase.txt",
        ),
        "sections_annotator2_lowercase_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/salami/"),
            "salami-data-public-hierarchy-corrections/annotations/2/parsed/textfile2_lowercase.txt",
        ),
        "source": "Codaich",
        "annotator_1_id": "5",
        "annotator_2_id": "8",
        "duration": 264,
        "title": "For_God_And_Country",
        "artist": "The_Smashing_Pumpkins",
        "annotator_1_time": "37",
        "annotator_2_time": "45",
        "broad_genre": "popular",
        "genre": "Alternative_Pop___Rock",
    }

    expected_property_types = {
        "sections_annotator_1_uppercase": annotations.SectionData,
        "sections_annotator_1_lowercase": annotations.SectionData,
        "sections_annotator_2_uppercase": annotations.SectionData,
        "sections_annotator_2_lowercase": annotations.SectionData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (89856,)

    # Test file with missing annotations
    track = dataset.track("192")

    # test attributes
    assert track.source == "Codaich"
    assert track.annotator_1_id == "16"
    assert track.annotator_2_id == "14"
    assert track.duration == 209
    assert track.title == "Sull__aria"
    assert track.artist == "Compilations"
    assert track.annotator_1_time == "20"
    assert track.annotator_2_time == ""
    assert track.broad_genre == "classical"
    assert track.genre == "Classical_-_Classical"
    assert track.track_id == "192"
    assert track._data_home == data_home

    assert track._track_paths == {
        "audio": ["audio/192.mp3", "d954d5dc9f17d66155d3310d838756b8"],
        "annotator_1_uppercase": [
            "salami-data-public-hierarchy-corrections/annotations/192/parsed/textfile1_uppercase.txt",
            "4d268cfd27fe011dbe579f25f8d125ce",
        ],
        "annotator_1_lowercase": [
            "salami-data-public-hierarchy-corrections/annotations/192/parsed/textfile1_lowercase.txt",
            "6640237e7844d0d9d37bf21cf96a2690",
        ],
        "annotator_2_uppercase": [None, None],
        "annotator_2_lowercase": [None, None],
    }

    # test that cached properties don't fail and have the expected type
    assert type(track.sections_annotator_1_uppercase) is annotations.SectionData
    assert type(track.sections_annotator_1_lowercase) is annotations.SectionData
    assert track.sections_annotator_2_uppercase is None
    assert track.sections_annotator_2_lowercase is None

    # Test file with missing annotations
    track = dataset.track("1015")

    assert track._track_paths == {
        "audio": ["audio/1015.mp3", "811a4a6b46f0c15a61bfb299b21ebdc4"],
        "annotator_1_uppercase": [None, None],
        "annotator_1_lowercase": [None, None],
        "annotator_2_uppercase": [
            "salami-data-public-hierarchy-corrections/annotations/1015/parsed/textfile2_uppercase.txt",
            "e4a268342a45fdffd8ec9c3b8287ad8b",
        ],
        "annotator_2_lowercase": [
            "salami-data-public-hierarchy-corrections/annotations/1015/parsed/textfile2_lowercase.txt",
            "201642fcea4a27c60f7b48de46a82234",
        ],
    }

    # test that cached properties don't fail and have the expected type
    assert track.sections_annotator_1_uppercase is None
    assert track.sections_annotator_1_lowercase is None
    assert type(track.sections_annotator_2_uppercase) is annotations.SectionData
    assert type(track.sections_annotator_2_lowercase) is annotations.SectionData


def test_to_jams():
    data_home = "tests/resources/mir_datasets/salami"
    dataset = salami.Dataset(data_home)
    track = dataset.track("2")
    jam = track.to_jams()

    segments = jam.search(namespace="multi_segment")[0]["data"]
    assert [segment.time for segment in segments] == [
        0.0,
        0.0,
        0.464399092,
        0.464399092,
        5.191269841,
        14.379863945,
        254.821632653,
        258.900453514,
        263.205419501,
        263.205419501,
    ]
    assert [segment.duration for segment in segments] == [
        0.464399092,
        0.464399092,
        13.915464853,
        4.726870749000001,
        249.630362812,
        248.82555555599998,
        4.078820860999997,
        4.304965987000003,
        1.6797959180000248,
        1.6797959180000248,
    ]
    assert [segment.value for segment in segments] == [
        {"label": "Silence", "level": 0},
        {"label": "Silence", "level": 1},
        {"label": "A", "level": 0},
        {"label": "b", "level": 1},
        {"label": "b", "level": 1},
        {"label": "B", "level": 0},
        {"label": "ab", "level": 1},
        {"label": "ab", "level": 1},
        {"label": "Silence", "level": 0},
        {"label": "Silence", "level": 1},
    ]
    assert [segment.confidence for segment in segments] == [
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

    assert jam["file_metadata"]["title"] == "For_God_And_Country"
    assert jam["file_metadata"]["artist"] == "The_Smashing_Pumpkins"


def test_load_sections():
    # load a file which exists
    sections_path = (
        "tests/resources/mir_datasets/salami/"
        + "salami-data-public-hierarchy-corrections/annotations/2/parsed/textfile1_uppercase.txt"
    )
    section_data = salami.load_sections(sections_path)

    # check types
    assert type(section_data) == annotations.SectionData
    assert type(section_data.intervals) is np.ndarray
    assert type(section_data.labels) is list

    # check valuess
    assert np.array_equal(
        section_data.intervals[:, 0],
        np.array([0.0, 0.464399092, 14.379863945, 263.205419501]),
    )
    assert np.array_equal(
        section_data.intervals[:, 1],
        np.array([0.464399092, 14.379863945, 263.205419501, 264.885215419]),
    )
    assert np.array_equal(section_data.labels, np.array(["Silence", "A", "B", "Silence"]))

    # load none
    section_data_none = salami.load_sections(None)
    assert section_data_none is None


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/salami"
    dataset = salami.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["2"] == {
        "source": "Codaich",
        "annotator_1_id": "5",
        "annotator_2_id": "8",
        "duration": 264,
        "title": "For_God_And_Country",
        "artist": "The_Smashing_Pumpkins",
        "annotator_1_time": "37",
        "annotator_2_time": "45",
        "class": "popular",
        "genre": "Alternative_Pop___Rock",
    }
