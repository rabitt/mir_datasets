import os

import numpy as np

from mirdata.datasets import giantsteps_key
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "3"
    data_home = os.path.normpath("tests/resources/mir_datasets/giantsteps_key")
    dataset = giantsteps_key.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/giantsteps_key/"),
            "audio/10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix).mp3",
        ),
        "keys_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/giantsteps_key/"),
            "keys_gs+/10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix).txt",
        ),
        "metadata_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/giantsteps_key/"),
            "meta/10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix).json",
        ),
        "title": "10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)",
        "track_id": "3",
    }

    expected_property_types = {
        "key": str,
        "genres": dict,
        "artists": list,
        "tempo": int,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (88200,), "audio shape {} was not (88200,)".format(audio.shape)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/giantsteps_key"
    dataset = giantsteps_key.Dataset(data_home)
    track = dataset.track("3")
    jam = track.to_jams()
    assert jam["sandbox"]["key"] == "D major", "key does not match expected"

    assert (
        jam["file_metadata"]["title"]
        == "10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)"
    ), "title does not match expected"
    sand_box = {
        "artists": ["Jason Sparks"],
        "genres": {"genres": ["Breaks"], "sub_genres": []},
        "tempo": 150,
        "key": "D major",
    }
    assert dict(jam["sandbox"]) == sand_box, "title does not match expected"


def test_load_key():
    key_path = (
        "tests/resources/mir_datasets/giantsteps_key/keys_gs+/10089 Jason Sparks - Close My Eyes feat. J. "
        + "Little (Original Mix).txt"
    )
    key_data = giantsteps_key.load_key(key_path)

    assert type(key_data) == str

    assert key_data == "D major"

    assert giantsteps_key.load_key(None) is None


def test_load_meta():
    meta_path = (
        "tests/resources/mir_datasets/giantsteps_key/meta/10089 Jason Sparks - Close My Eyes feat. J. "
        + "Little (Original Mix).json"
    )
    genres = {"genres": ["Breaks"], "sub_genres": []}
    artists = ["Jason Sparks"]
    tempo = 150

    assert type(giantsteps_key.load_genre(meta_path)) == dict
    assert type(giantsteps_key.load_artist(meta_path)) == list
    assert type(giantsteps_key.load_tempo(meta_path)) == int

    assert giantsteps_key.load_genre(meta_path) == genres
    assert giantsteps_key.load_artist(meta_path) == artists
    assert giantsteps_key.load_tempo(meta_path) == tempo

    assert giantsteps_key.load_genre(None) is None
    assert giantsteps_key.load_artist(None) is None
    assert giantsteps_key.load_tempo(None) is None
