import os

import numpy as np

from mirdata.datasets import good_sounds
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "1"
    data_home = os.path.normpath("tests/resources/mir_datasets/good_sounds")
    dataset = good_sounds.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/good_sounds/"),
            "good-sounds/sound_files/flute_almudena_reference/akg/0000.wav",
        ),
        "track_id": "1",
    }

    expected_property_types = {
        "audio": tuple,
        "pack_info": dict,
        "ratings_info": list,
        "sound_info": dict,
        "take_info": dict,
        "microphone": str,
        "instrument": str,
        "klass": str,
        "semitone": int,
        "pitch_reference": int,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (44100,), "audio shape {} was not (44100,)".format(audio.shape)


def test_track_properties_and_attributes():
    default_trackid = "1"
    data_home = os.path.normpath("tests/resources/mir_datasets/good_sounds")
    dataset = good_sounds.Dataset(data_home)
    track = dataset.track(default_trackid)
    ground_truth_sound = {
        "id": 1,
        "instrument": "flute",
        "note": "C",
        "octave": 4,
        "dynamics": "mf",
        "recorded_at": "2013-10-28 12:00:00.000000",
        "location": "upf studio",
        "player": "almudena",
        "bow_velocity": None,
        "bridge_position": None,
        "string": None,
        "csv_file": 1,
        "csv_id": 1,
        "pack_filename": "0000.wav",
        "pack_id": 1,
        "attack": 105810,
        "decay": 110629,
        "sustain": None,
        "release": 332406,
        "offset": 343765,
        "reference": 1,
        "klass": "good-sound",
        "comments": None,
        "semitone": 48,
        "pitch_reference": 442,
    }
    ground_truth_take = {
        "id": 1,
        "microphone": "akg",
        "filename": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/good_sounds/"),
            "good-sounds/sound_files/flute_almudena_reference/akg/0000.wav",
        ),
        "original_filename": "AKG-costado-Left-01 render 001",
        "freesound_id": None,
        "sound_id": 1,
        "goodsound_id": None,
    }
    ground_truth_ratings = []
    ground_truth_pack = {
        "id": 1,
        "name": "flute_almudena_reference",
        "description": "Play reference notes",
    }
    assert track.sound_info == ground_truth_sound
    assert track.take_info == ground_truth_take
    assert track.pack_info == ground_truth_pack
    assert track.ratings_info == ground_truth_ratings
    assert track.microphone == "akg"
    assert track.instrument == "flute"
    assert track.klass == "good-sound"
    assert track.semitone == 48
    assert track.pitch_reference == 442


def test_to_jams():
    default_trackid = "1"
    data_home = os.path.normpath("tests/resources/mir_datasets/good_sounds")
    dataset = good_sounds.Dataset(data_home)
    track = dataset.track(default_trackid)

    jam = track.to_jams()
    ground_truth_sound = {
        "id": 1,
        "instrument": "flute",
        "note": "C",
        "octave": 4,
        "dynamics": "mf",
        "recorded_at": "2013-10-28 12:00:00.000000",
        "location": "upf studio",
        "player": "almudena",
        "bow_velocity": None,
        "bridge_position": None,
        "string": None,
        "csv_file": 1,
        "csv_id": 1,
        "pack_filename": "0000.wav",
        "pack_id": 1,
        "attack": 105810,
        "decay": 110629,
        "sustain": None,
        "release": 332406,
        "offset": 343765,
        "reference": 1,
        "klass": "good-sound",
        "comments": None,
        "semitone": 48,
        "pitch_reference": 442,
    }
    ground_truth_take = {
        "id": 1,
        "microphone": "akg",
        "filename": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/good_sounds/"),
            "good-sounds/sound_files/flute_almudena_reference/akg/0000.wav",
        ),
        "original_filename": "AKG-costado-Left-01 render 001",
        "freesound_id": None,
        "sound_id": 1,
        "goodsound_id": None,
    }
    ground_truth_ratings = []
    ground_truth_pack = {
        "id": 1,
        "name": "flute_almudena_reference",
        "description": "Play reference notes",
    }
    assert jam["sandbox"].sound == ground_truth_sound
    assert jam["sandbox"].take == ground_truth_take
    assert jam["sandbox"].pack == ground_truth_pack
    assert jam["sandbox"].ratings == ground_truth_ratings
    assert jam["sandbox"].microphone == "akg"
    assert jam["sandbox"].instrument == "flute"
    assert jam["sandbox"].klass == "good-sound"
    assert jam["sandbox"].semitone == 48
    assert jam["sandbox"].pitch_reference == 442
