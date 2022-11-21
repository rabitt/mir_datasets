import numpy as np
from mirdata import annotations
from mirdata.datasets import compmusic_hindustani_rhythm
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "20001"
    data_home = "tests/resources/mir_datasets/compmusic_hindustani_rhythm"
    dataset = compmusic_hindustani_rhythm.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "20001",
        "audio_path": "tests/resources/mir_datasets/compmusic_hindustani_rhythm/HMR_1.0/"
        + "audio/01_20001_02_Raag_Multani.wav",
        "beats_path": "tests/resources/mir_datasets/compmusic_hindustani_rhythm/HMR_1.0/"
        + "annotations/beats/01_20001_02_Raag_Multani.beats",
        "meter_path": "tests/resources/mir_datasets/compmusic_hindustani_rhythm/HMR_1.0/"
        + "annotations/meter/01_20001_02_Raag_Multani.meter",
    }

    expected_property_types = {
        "meter": str,
        "beats": annotations.BeatData,
        "audio": tuple,
        "mbid": str,
        "name": str,
        "artist": str,
        "release": str,
        "lead_instrument_code": str,
        "taala": str,
        "raaga": str,
        "laya": str,
        "num_of_beats": int,
        "num_of_samas": int,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    _, sr = track.audio
    assert sr == 44100


def test_to_jams():
    data_home = "tests/resources/mir_datasets/compmusic_hindustani_rhythm"
    dataset = compmusic_hindustani_rhythm.Dataset(data_home)
    track = dataset.track("20001")
    jam = track.to_jams()

    # Tonic
    assert jam["sandbox"].meter == "16/8"

    # Sama
    beats = jam.search(namespace="beat")[0]["data"]
    assert len(beats) == 3
    assert [beat.time for beat in beats] == [0.694, 3.233, 6.020]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [13, 14, 15]
    assert [beat.confidence for beat in beats] == [None, None, None]

    # Metadata
    assert jam["sandbox"]["mbid"] == "0bdad2a8-94d8-40c2-91ec-e77100fcaa02"
    assert jam["sandbox"]["raaga"] == "Multani"
    assert jam["sandbox"]["taala"] == "teentaal"
    assert jam["sandbox"]["name"] == "02_Raag_Multani"
    assert jam["sandbox"]["lead_instrument_code"] == "V"
    assert jam["sandbox"]["laya"] == "Vilambit"
    assert jam["sandbox"]["num_of_beats"] == 44
    assert jam["sandbox"]["num_of_samas"] == 3


def test_load_meter():
    data_home = "tests/resources/mir_datasets/compmusic_hindustani_rhythm"
    dataset = compmusic_hindustani_rhythm.Dataset(data_home)
    track = dataset.track("20001")
    meter_path = track.meter_path
    parsed_meter = compmusic_hindustani_rhythm.load_meter(meter_path)
    assert parsed_meter == "16/8"
    assert compmusic_hindustani_rhythm.load_meter(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/compmusic_hindustani_rhythm"
    dataset = compmusic_hindustani_rhythm.Dataset(data_home)
    track = dataset.track("20001")
    beats_path = track.beats_path
    parsed_beats = compmusic_hindustani_rhythm.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray
    assert type(parsed_beats.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([0.694, 3.233, 6.020]))
    assert np.array_equal(parsed_beats.positions, np.array([13, 14, 15]))
    assert compmusic_hindustani_rhythm.load_beats(None) is None


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/compmusic_hindustani_rhythm"
    dataset = compmusic_hindustani_rhythm.Dataset(data_home)
    meta = dataset._metadata  # get dataset metadata
    parsed_metadata = meta["20001"]  # get track metadata

    assert parsed_metadata["mbid"] == "0bdad2a8-94d8-40c2-91ec-e77100fcaa02"
    assert parsed_metadata["raaga"] == "Multani"
    assert parsed_metadata["taala"] == "teentaal"
    assert parsed_metadata["name"] == "02_Raag_Multani"
    assert parsed_metadata["lead_instrument_code"] == "V"
    assert parsed_metadata["laya"] == "Vilambit"
    assert parsed_metadata["num_of_beats"] == 44
    assert parsed_metadata["num_of_samas"] == 3


def test_load_audio():
    data_home = "tests/resources/mir_datasets/compmusic_hindustani_rhythm"
    dataset = compmusic_hindustani_rhythm.Dataset(data_home)
    track = dataset.track("20001")
    audio_path = track.audio_path
    audio, sr = compmusic_hindustani_rhythm.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert compmusic_hindustani_rhythm.load_audio(None) is None
