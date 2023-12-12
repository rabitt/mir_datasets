import os

import numpy as np

from mirdata.datasets import freesound_one_shot_percussive_sounds
from tests.test_utils import run_track_tests

TEST_DATA_HOME = os.path.normpath(
    "tests/resources/mir_datasets/freesound_one_shot_percussive_sounds"
)


def test_track():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/freesound_one_shot_percussive_sounds/"),
            "one_shot_percussive_sounds/1/183.wav",
        ),
        "file_metadata_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/freesound_one_shot_percussive_sounds/"),
            "analysis/1/183_analysis.json",
        ),
        "track_id": "183",
    }

    expected_property_types = {
        "audio": tuple,
        "tags": list,
        "freesound_preview_urls": dict,
        "freesound_analysis": dict,
        "audiocommons_analysis": dict,
        "file_metadata": dict,
        "filename": str,
        "username": str,
        "license": str,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate cante100 jam schema
    assert jam.validate()

    assert jam.file_metadata["duration"] == 0.34575000405311584
    assert jam.annotations == []

    # Check jam sandbox
    jam_metadata = jam.sandbox
    assert jam_metadata["name"] == "1.wav"
    assert jam_metadata["username"] == "plagasul"
    assert jam_metadata["license"] == "http://creativecommons.org/licenses/by/3.0/"
    assert jam_metadata["tags"] == [
        "Zil",
        "Cymbals",
        "Finger",
        "Percussion",
        "Hand",
        "Zyl",
        "Bell",
        "Chimes",
    ]
    assert jam_metadata["previews"] == {
        "preview_lq_ogg": "https://freesound.org/data/previews/414/183_394391-lq.ogg",
        "preview_lq_mp3": "https://freesound.org/data/previews/414/183_394391-lq.mp3",
        "preview_hq_mp3": "https://freesound.org/data/previews/414/183_394391-hq.mp3",
        "preview_hq_ogg": "https://freesound.org/data/previews/414/183_394391-hq.ogg",
    }
    assert jam_metadata["analysis"] == {
        "lowlevel": {
            "average_loudness": 0.0028868783722041064,
            "silence_rate_30dB": {
                "min": 1.0,
                "max": 1.0,
                "dvar2": 0.0,
                "dmean2": 0.0,
                "dmean": 0.0,
                "var": 0.0,
                "dvar": 0.0,
                "mean": 1.0,
            },
            "stopFrame": 5.999972947644955,
            "startFrame": 0.0,
        },
        "sfx": {
            "duration": 0.31421999239346726,
            "logattacktime": {
                "max": -1.7424356459399086,
                "mean": -1.7424356459399086,
                "min": -1.7424356459399086,
            },
            "effective_duration": {
                "max": 0.3105442902494931,
                "mean": 0.3105442902494931,
                "min": 0.3105442902494931,
            },
            "temporal_centroid": {
                "max": 0.5048781145009638,
                "mean": 0.5048781145009638,
                "min": 0.5048781145009638,
            },
            "temporal_decrease": {
                "max": 0.010950043054148862,
                "mean": 0.010950043054148862,
                "min": 0.010950043054148862,
            },
        },
    }
    assert jam_metadata["ac_analysis"] == {
        "ac_tempo_confidence": 0.0,
        "ac_note_confidence": 0.4735875129699707,
        "ac_depth": 5.746736899993124,
        "ac_note_midi": 100,
        "ac_temporal_centroid": 0.17407511174678802,
        "ac_warmth": 16.86232842111746,
        "ac_loop": False,
        "ac_hardness": 78.2319728568732,
        "ac_loudness": -14.819193840026855,
        "ac_reverb": False,
        "ac_roughness": 73.50090654168224,
        "ac_log_attack_time": -1.741891860961914,
        "ac_boominess": 0.0,
        "ac_note_frequency": 2611.37890625,
        "ac_tempo": 0,
        "ac_brightness": 86.5129643484791,
        "ac_sharpness": 86.05006458619545,
        "ac_tonality_confidence": 0.7178749442100525,
        "ac_dynamic_range": 0.0,
        "ac_note_name": "E7",
        "ac_tonality": "F minor",
        "ac_single_event": True,
    }


def test_load_analysis():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    file_metadata_path = track.file_metadata_path
    file_metadata = freesound_one_shot_percussive_sounds.load_file_metadata(file_metadata_path)

    # check file metadata elements
    assert type(file_metadata) is dict
    assert file_metadata.get("duration") == 0.34575000405311584
    assert file_metadata.get("lossless") == 1.0
    assert file_metadata.get("codec") == "pcm_s16le"
    assert file_metadata.get("bitrate") == 256000.0
    assert file_metadata.get("samplerate") == 16000.0
    assert file_metadata.get("channels") == 1.0
    assert file_metadata.get("audio_md5") == "dd11a896b4d08c2a93d6480bb3e40016"
    assert file_metadata.get("filesize") == 11108


def test_load_audio():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    audio_path = track.audio_path
    audio, sr = freesound_one_shot_percussive_sounds.load_audio(audio_path)
    assert sr == 16000
    assert type(audio) is np.ndarray


def test_metadata():
    default_trackid = "183"
    dataset = freesound_one_shot_percussive_sounds.Dataset(TEST_DATA_HOME)
    metadata = dataset._metadata

    assert type(metadata) is dict
    assert metadata[default_trackid].get("name") == "1.wav"
    assert metadata[default_trackid].get("username") == "plagasul"
    assert metadata[default_trackid].get("license") == "http://creativecommons.org/licenses/by/3.0/"
    assert metadata[default_trackid].get("duration") == 0.34575000405311584

    # Check tags
    assert type(metadata[default_trackid].get("tags")) is list
    assert metadata[default_trackid].get("tags") == [
        "Zil",
        "Cymbals",
        "Finger",
        "Percussion",
        "Hand",
        "Zyl",
        "Bell",
        "Chimes",
    ]

    # Check previews
    assert type(metadata[default_trackid].get("previews")) is dict
    assert metadata[default_trackid].get("previews") == {
        "preview_lq_ogg": "https://freesound.org/data/previews/414/183_394391-lq.ogg",
        "preview_lq_mp3": "https://freesound.org/data/previews/414/183_394391-lq.mp3",
        "preview_hq_mp3": "https://freesound.org/data/previews/414/183_394391-hq.mp3",
        "preview_hq_ogg": "https://freesound.org/data/previews/414/183_394391-hq.ogg",
    }

    # Check freesound analysis
    assert type(metadata[default_trackid].get("analysis")) is dict
    assert metadata[default_trackid].get("analysis") == {
        "lowlevel": {
            "average_loudness": 0.0028868783722041064,
            "silence_rate_30dB": {
                "min": 1.0,
                "max": 1.0,
                "dvar2": 0.0,
                "dmean2": 0.0,
                "dmean": 0.0,
                "var": 0.0,
                "dvar": 0.0,
                "mean": 1.0,
            },
            "stopFrame": 5.999972947644955,
            "startFrame": 0.0,
        },
        "sfx": {
            "duration": 0.31421999239346726,
            "logattacktime": {
                "max": -1.7424356459399086,
                "mean": -1.7424356459399086,
                "min": -1.7424356459399086,
            },
            "effective_duration": {
                "max": 0.3105442902494931,
                "mean": 0.3105442902494931,
                "min": 0.3105442902494931,
            },
            "temporal_centroid": {
                "max": 0.5048781145009638,
                "mean": 0.5048781145009638,
                "min": 0.5048781145009638,
            },
            "temporal_decrease": {
                "max": 0.010950043054148862,
                "mean": 0.010950043054148862,
                "min": 0.010950043054148862,
            },
        },
    }

    # Check audiocommons analysis
    assert type(metadata[default_trackid].get("ac_analysis")) is dict
    assert metadata[default_trackid].get("ac_analysis") == {
        "ac_tempo_confidence": 0.0,
        "ac_note_confidence": 0.4735875129699707,
        "ac_depth": 5.746736899993124,
        "ac_note_midi": 100,
        "ac_temporal_centroid": 0.17407511174678802,
        "ac_warmth": 16.86232842111746,
        "ac_loop": False,
        "ac_hardness": 78.2319728568732,
        "ac_loudness": -14.819193840026855,
        "ac_reverb": False,
        "ac_roughness": 73.50090654168224,
        "ac_log_attack_time": -1.741891860961914,
        "ac_boominess": 0.0,
        "ac_note_frequency": 2611.37890625,
        "ac_tempo": 0,
        "ac_brightness": 86.5129643484791,
        "ac_sharpness": 86.05006458619545,
        "ac_tonality_confidence": 0.7178749442100525,
        "ac_dynamic_range": 0.0,
        "ac_note_name": "E7",
        "ac_tonality": "F minor",
        "ac_single_event": True,
    }
