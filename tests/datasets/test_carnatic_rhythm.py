import os
import numpy as np
import pytest
from mirdata import annotations

try:
    from mirdata.datasets import compmusic_carnatic_rhythm
except ImportError:
    raise ImportError(
        "An error occured when importing this dataset. Most likely this is due to a dependency not being installed, in this case openpyxl."
    )

from tests.test_utils import run_track_tests


def test_importError():
    with pytest.raises(ImportError):
        raise (ImportError)


def test_track():
    default_trackid = "10003"
    data_home = os.path.normpath(
        "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    )
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "10003",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_carnatic_rhythm/"),
            "CMR_subset_1.0/audio/01_10003_1-04_Shri_Visvanatham.wav",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_carnatic_rhythm/"),
            "CMR_subset_1.0/annotations/beats/01_10003_1-04_Shri_Visvanatham.beats",
        ),
        "meter_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_carnatic_rhythm/"),
            "CMR_subset_1.0/annotations/meter/01_10003_1-04_Shri_Visvanatham.meter",
        ),
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
        "num_of_beats": int,
        "num_of_samas": int,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    _, sr = track.audio
    assert sr == 44100


def test_to_jams():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("10003")
    jam = track.to_jams()

    print(jam)

    # Tonic
    assert jam["sandbox"].meter == "8/4"

    # Sama
    beats = jam.search(namespace="beat")[0]["data"]
    assert len(beats) == 3
    assert [beat.time for beat in beats] == [1.124172, 1.788957, 2.502540]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [1, 2, 3]
    assert [beat.confidence for beat in beats] == [None, None, None]

    # Metadata
    assert jam["sandbox"]["mbid"] == "5769ea2f-aed4-4169-9a20-bae4cb733b8e"
    assert jam["sandbox"]["raaga"] == "chaturdasha ragamalika"
    assert jam["sandbox"]["taala"] == "adi"
    assert jam["sandbox"]["name"] == "1-04_Shri_Visvanatham"
    assert jam["sandbox"]["lead_instrument_code"] == "V"
    assert jam["sandbox"]["num_of_beats"] == 162
    assert jam["sandbox"]["num_of_samas"] == 21


def test_load_meter():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("10003")
    meter_path = track.meter_path
    parsed_meter = compmusic_carnatic_rhythm.load_meter(meter_path)
    assert parsed_meter == "8/4"
    assert compmusic_carnatic_rhythm.load_meter(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("10003")
    beats_path = track.beats_path
    parsed_beats = compmusic_carnatic_rhythm.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray
    assert type(parsed_beats.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([1.124172, 1.788957, 2.502540]))
    assert np.array_equal(parsed_beats.positions, np.array([1, 2, 3]))
    assert compmusic_carnatic_rhythm.load_beats(None) is None

    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="full_dataset")
    track = dataset.track("10001")
    beats_path = track.beats_path
    parsed_beats = compmusic_carnatic_rhythm.load_beats(beats_path)
    assert parsed_beats is None


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    meta = dataset._metadata  # get dataset metadata
    parsed_metadata = meta["10003"]  # get track metadata

    assert parsed_metadata["mbid"] == "5769ea2f-aed4-4169-9a20-bae4cb733b8e"
    assert parsed_metadata["raaga"] == "chaturdasha ragamalika"
    assert parsed_metadata["taala"] == "adi"
    assert parsed_metadata["name"] == "1-04_Shri_Visvanatham"
    assert parsed_metadata["lead_instrument_code"] == "V"
    assert parsed_metadata["num_of_beats"] == 162
    assert parsed_metadata["num_of_samas"] == 21

    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="full_dataset")
    meta = dataset._metadata  # get dataset metadata
    parsed_metadata = meta["10001"]  # get track metadata

    assert parsed_metadata["mbid"] == "6fb02d72-120f-415a-bf46-cd455a61165c"
    assert parsed_metadata["raaga"] == "salaga bhairavi"
    assert parsed_metadata["taala"] == "adi"
    assert parsed_metadata["name"] == "05_Thunga_Theera_Virajam"
    assert parsed_metadata["artist"] == "Abhishek Raghuram"
    assert parsed_metadata["lead_instrument_code"] == "V"
    assert parsed_metadata["num_of_beats"] == 193
    assert parsed_metadata["num_of_samas"] == 25


def test_load_audio():
    data_home = "tests/resources/mir_datasets/compmusic_carnatic_rhythm"
    dataset = compmusic_carnatic_rhythm.Dataset(data_home, version="test")
    track = dataset.track("10003")
    audio_path = track.audio_path
    audio, sr = compmusic_carnatic_rhythm.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert compmusic_carnatic_rhythm.load_audio(None) is None
