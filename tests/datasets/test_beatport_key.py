import os

import numpy as np

from mirdata.datasets import beatport_key
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "1"
    data_home = os.path.normpath("tests/resources/mir_datasets/beatport_key")
    dataset = beatport_key.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatport_key/"),
            "audio/100066 Lindstrom - Monsteer (Original Mix).mp3",
        ),
        "keys_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatport_key/"),
            "keys/100066 Lindstrom - Monsteer (Original Mix).txt",
        ),
        "metadata_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatport_key/"),
            "meta/100066 Lindstrom - Monsteer (Original Mix).json",
        ),
        "title": "100066 Lindstrom - Monsteer (Original Mix)",
        "track_id": "1",
    }

    expected_property_types = {
        "key": list,
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
    data_home = "tests/resources/mir_datasets/beatport_key"
    dataset = beatport_key.Dataset(data_home)
    track = dataset.track("1")
    jam = track.to_jams()
    assert jam["sandbox"]["key"] == ["D minor"], "key does not match expected"

    assert (
        jam["file_metadata"]["title"] == "100066 Lindstrom - Monsteer (Original Mix)"
    ), "title does not match expected"
    sand_box = {
        "artists": ["Lindstrom"],
        "genres": {"genres": ["Electronica / Downtempo"], "sub_genres": []},
        "tempo": 115,
        "key": ["D minor"],
    }
    assert dict(jam["sandbox"]) == sand_box, "sandbox does not match expected"


def test_load_key():
    key_path = "tests/resources/mir_datasets/beatport_key/keys/100066 Lindstrom - Monsteer (Original Mix).txt"
    key_data = beatport_key.load_key(key_path)

    assert type(key_data) == list

    assert key_data == ["D minor"]

    assert beatport_key.load_key(None) is None


def test_load_meta():
    meta_path = "tests/resources/mir_datasets/beatport_key/meta/100066 Lindstrom - Monsteer (Original Mix).json"
    genres = {"genres": ["Electronica / Downtempo"], "sub_genres": []}
    artists = ["Lindstrom"]
    tempo = 115

    assert type(beatport_key.load_genre(meta_path)) == dict
    assert type(beatport_key.load_artist(meta_path)) == list
    assert type(beatport_key.load_tempo(meta_path)) == int

    assert beatport_key.load_genre(meta_path) == genres
    assert beatport_key.load_artist(meta_path) == artists
    assert beatport_key.load_tempo(meta_path) == tempo

    assert beatport_key.load_genre(None) is None
    assert beatport_key.load_artist(None) is None
    assert beatport_key.load_tempo(None) is None


def test_find_replace():
    with open("tests/resources/mir_datasets/beatport_key/find_replace.json", "w") as the_file:
        the_file.write('{"probando": nan}')
    dataset = beatport_key.Dataset()
    dataset._find_replace("tests/resources/mir_datasets/beatport_key", ": nan", ": null", "*.json")
    f = open("tests/resources/mir_datasets/beatport_key/find_replace.json", "r")
    content = f.read()
    assert content == '{"probando": null}'
