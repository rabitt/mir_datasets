# -*- coding: utf-8 -*-

import os

from tests.test_utils import run_track_tests

from mirdata import mridangam_stroke
from tests.test_utils import DEFAULT_DATA_HOME

TEST_DATA_HOME = "tests/resources/mir_datasets/Mridangam-Stroke"


def test_track_default_data_home():
    # test data home None
    track_default = mridangam_stroke.Track("224030")
    assert track_default._data_home == os.path.join(
        DEFAULT_DATA_HOME, "Mridangam-Stroke"
    )


def test_track():
    default_trackid = "224030"
    track = mridangam_stroke.Track(default_trackid, data_home=TEST_DATA_HOME)
    expected_attributes = {
        'audio_path': "tests/resources/mir_datasets/Mridangam-Stroke/mridangam_stroke_1.5/"
        + "B/224030__akshaylaya__bheem-b-001.wav",
        'stroke_name': 'bheem',
        'tonic': 'B',
        'track_id': "224030",
    }
    run_track_tests(track, expected_attributes, {})

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (35841,)


def test_to_jams():
    default_trackid = "224030"
    track = mridangam_stroke.Track(default_trackid, data_home=TEST_DATA_HOME)
    jam = track.to_jams()
    # print(jam)

    # Validate Mridangam schema
    assert jam.validate()

    # Test the stroke parser
    assert jam.annotations["tag_open"][0].data[0].value == "bheem"
    assert jam.sandbox.tonic == "B"


def test_get_tonic():
    default_trackid = "224030"
    track = mridangam_stroke.Track(default_trackid, data_home=TEST_DATA_HOME)
    tonic = mridangam_stroke.get_tonic(track.audio_path)
    assert tonic == 'B'


def test_get_stroke_name():
    default_trackid = "224030"
    track = mridangam_stroke.Track(default_trackid, data_home=TEST_DATA_HOME)
    stroke_name = mridangam_stroke.get_stroke_name(track.audio_path)
    assert stroke_name == 'bheem'
