import os
from collections import deque

import numpy as np
import pandas as pd
import pytest

from mirdata.datasets import baf
from tests.test_utils import run_track_tests

TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets/baf")
TRACK_ID = "query_0001"
TRACK_ID2 = "query_0002"


def test_track():
    default_trackid = TRACK_ID
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/baf/queries"),
            TRACK_ID + ".wav",
        ),
        "track_id": TRACK_ID,
    }

    expected_property_types = {
        "audio": tuple,
        "country": str,
        "channel": str,
        "datetime": str,
        "matches": baf.EventDataExtended,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    default_trackid = TRACK_ID
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()
    assert jam.validate()


def test_load_audio():
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(TRACK_ID)
    audio_path = track.audio_path
    audio, sr = baf.load_audio(audio_path)
    assert sr == 8000
    assert type(audio) is np.ndarray


def test_load_matches():
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(TRACK_ID)
    matches = baf.load_matches(track._track_metadata)

    assert type(matches) == baf.EventDataExtended
    assert type(matches.intervals) == np.ndarray
    assert type(matches.interval_unit) == str
    assert type(matches.events) == list
    assert type(matches.event_unit) == str
    assert type(matches.tags) == list
    assert type(matches.tag_unit) == str

    intervals_list = deque(
        [
            [40.44, 59.936],
            [40.0, 40.44],
            [0.0, 33.0],
            [33.0, 34.49],
            [34.49, 34.61],
        ]
    )
    intervals = np.array(intervals_list, dtype=float)
    interval_unit = "s"
    events = ["ref_0027", "ref_0027", "ref_1072", "ref_1072", "ref_1072"]
    event_unit = "open"
    tags = ["unanimity", "majority", "unanimity", "majority", "single"]
    tag_unit = "open"

    assert isinstance(matches, baf.EventDataExtended)
    assert matches.intervals.all() == intervals.all()
    assert matches.interval_unit == interval_unit
    assert matches.events == events
    assert matches.event_unit == event_unit
    assert matches.tags == tags
    assert matches.tag_unit == tag_unit

    track2 = dataset.track(TRACK_ID2)
    assert baf.load_matches(track2._track_metadata) is None


def test_metadata():
    dataset = baf.Dataset(TEST_DATA_HOME)
    metadata = dataset._metadata
    assert metadata[TRACK_ID] == {
        "country": "Norway",
        "channel": "Discovery Channel",
        "datetime": "2021-02-26 14:45:26",
        "annotations": [
            {
                "reference": "ref_0027",
                "query_start": 40.44,
                "query_end": 59.936,
                "tag": "unanimity",
            },
            {
                "reference": "ref_0027",
                "query_start": 40.0,
                "query_end": 40.44,
                "tag": "majority",
            },
            {
                "reference": "ref_1072",
                "query_start": 0.0,
                "query_end": 33.0,
                "tag": "unanimity",
            },
            {
                "reference": "ref_1072",
                "query_start": 33.0,
                "query_end": 34.49,
                "tag": "majority",
            },
            {
                "reference": "ref_1072",
                "query_start": 34.49,
                "query_end": 34.61,
                "tag": "single",
            },
        ],
    }
    assert metadata[TRACK_ID2] == {
        "country": "Norway",
        "channel": "TV2 Norway",
        "datetime": "2021-01-09 16:08:50",
        "annotations": [],
    }

    with pytest.raises(FileNotFoundError):
        dataset = baf.Dataset("/a/fake/path")
        metadata = dataset._metadata


def test_csv_to_pandas():
    # Test case where queries_info file exist
    queries_info_csv = os.path.join(TEST_DATA_HOME, "queries_info.csv")
    queries_info_df = baf.csv_to_pandas(queries_info_csv)
    assert type(queries_info_df) is pd.DataFrame
    # Test case where cross_annotaitons file exist
    xann_csv = os.path.join(TEST_DATA_HOME, "cross_annotations.csv")
    xannotations_df = baf.csv_to_pandas(xann_csv)
    assert type(xannotations_df) is pd.DataFrame
    # Test case where a file doesn't exist
    queries_info_csv = "/a/fake/path"
    with pytest.raises(FileNotFoundError):
        queries_info_df = baf.csv_to_pandas(queries_info_csv)
