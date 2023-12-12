import os

import pytest

from mirdata.datasets import compmusic_indian_tonic
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180"
    data_home = os.path.normpath("tests/resources/mir_datasets/compmusic_indian_tonic")
    dataset = compmusic_indian_tonic.Dataset(data_home)
    track = dataset.track(default_trackid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_indian_tonic/"),
            "indian_art_music_tonic_1.0/CM/audio/0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180.mp3",
        ),
        "track_id": "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180",
    }

    run_track_tests(
        track,
        expected_attributes,
        {
            "audio": tuple,
            "tonic": float,
            "artist": str,
            "gender": str,
            "mbid": str,
            "type": str,
            "tradition": str,
        },
    )

    _, sr = track.audio
    assert sr == 44100


def test_to_jams():
    default_trackid = "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180"
    data_home = "tests/resources/mir_datasets/compmusic_indian_tonic"
    dataset = compmusic_indian_tonic.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam.validate()

    # Test annotations
    assert jam.sandbox.tonic == 131.436
    assert jam.file_metadata.artist == "T. N. Seshagopalan"


def test_load_audio():
    default_trackid = "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180"
    data_home = "tests/resources/mir_datasets/compmusic_indian_tonic"
    dataset = compmusic_indian_tonic.Dataset(data_home)
    track = dataset.track(default_trackid)

    _, sr = track.audio
    assert sr == 44100


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/compmusic_indian_tonic"
    dataset = compmusic_indian_tonic.Dataset(data_home)

    default_trackid = "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180"
    track = dataset.track(default_trackid)
    track.tonic == 131.436
    track.artist == "T. N. Seshagopalan"
    track.gender == "Male"
    track.mbid == "0a6ebaa4-87cc-452d-a7af-a2006e96f16a"
    track.type == "Instrumental"
    track.tradition == "Carnatic"

    default_trackid_iitm = "01-Varnam"
    track_iitm = dataset.track(default_trackid_iitm)
    track_iitm.tonic == 148.32
    track_iitm.artist == "NA"
    track_iitm.gender == "Male"
    track_iitm.mbid == -1
    track_iitm.type == "NA"
    track_iitm.tradition == "NA"

    meta = dataset._metadata
    assert meta[default_trackid_iitm]["tonic"] == track_iitm.tonic
    assert meta[default_trackid_iitm]["artist"] == track_iitm.artist
    assert meta[default_trackid_iitm]["gender"] == track_iitm.gender
    assert meta[default_trackid_iitm]["mbid"] == track_iitm.mbid
    assert meta[default_trackid_iitm]["type"] == track_iitm.type
    assert meta[default_trackid_iitm]["tradition"] == track_iitm.tradition
