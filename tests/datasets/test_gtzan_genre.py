import os
import numpy as np
from tests.test_utils import run_track_tests
from mirdata import annotations
from mirdata.datasets import gtzan_genre

TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets/gtzan_genre")


def test_track():
    default_trackid = "country.00000"
    dataset = gtzan_genre.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    expected_attributes = {
        "genre": "country",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/gtzan_genre/"),
            "gtzan_mini-main/genres/country/country.00000.wav",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/gtzan_genre/"),
            "gtzan_tempo_beat-main/beats/gtzan_country_00000.beats",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/gtzan_genre/"),
            "gtzan_tempo_beat-main/tempo/gtzan_country_00000.bpm",
        ),
        "track_id": "country.00000",
    }
    expected_properties = {
        "audio": tuple,
        "beats": annotations.BeatData,
        "tempo": float,
    }
    run_track_tests(track, expected_attributes, expected_properties)

    audio, sr = track.audio
    assert sr == 22050
    assert audio.shape == (663300,)


def test_load_beats():
    beats_path = (
        "tests/resources/mir_datasets/gtzan_genre/"
        + "gtzan_tempo_beat-main/beats/gtzan_country_00000.beats"
    )
    beat_data = gtzan_genre.load_beats(beats_path)

    assert (
        type(beat_data) == annotations.BeatData
    ), "beat_data is not type annotations.BeatData"
    assert type(beat_data.times) == np.ndarray, "beat_data.times is not an np.ndarray"
    assert (
        type(beat_data.positions) == np.ndarray
    ), "beat_data.positions is not an np.ndarray"

    assert np.array_equal(
        beat_data.times, np.array([0.113, 0.829, 1.537, 2.28, 2.992])
    ), "beat_data.times different than expected"
    assert np.array_equal(
        beat_data.positions, np.array([1, 2, 3, 4, 1])
    ), "beat_data.positions different from expected"

    assert gtzan_genre.load_beats(None) is None, "load_beats(None) should return None"

    # check empty positions
    beats_path = (
        "tests/resources/mir_datasets/gtzan_genre/"
        + "gtzan_tempo_beat-main/beats/gtzan_country_00000_noposition.beats"
    )
    beat_data = gtzan_genre.load_beats(beats_path)

    assert np.array_equal(
        beat_data.times, np.array([0.113, 0.829, 1.537, 2.28, 2.992])
    ), "beat_data.times different than expected"
    assert np.array_equal(
        beat_data.positions, None
    ), "beat_data.positions different from expected"


def test_load_tempo():
    tempo_path = (
        "tests/resources/mir_datasets/gtzan_genre/"
        + "gtzan_tempo_beat-main/tempo/gtzan_country_00000.bpm"
    )
    tempo_data = gtzan_genre.load_tempo(tempo_path)

    assert type(tempo_data) == float, "tempo_data is not type float"

    assert np.array_equal(
        tempo_data, 8.553000000000000114e01
    ), "tempo_data different than expected"


def test_hiphop():
    dataset = gtzan_genre.Dataset(TEST_DATA_HOME)
    track = dataset.track("hiphop.00000")
    assert track.genre == "hip-hop"


def test_to_jams():
    default_trackid = "country.00000"
    dataset = gtzan_genre.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate GTZAN schema
    assert jam.validate()

    # Test the that the genre parser of mirdata is correct
    assert jam.annotations["tag_gtzan"][0].data[0].value == "country"

    beats = jam.search(namespace="beat")[0]["data"]
    assert [beat.time for beat in beats] == [
        0.113,
        0.829,
        1.537,
        2.28,
        2.992,
    ], "beat times do not match expected"
    assert [beat.duration for beat in beats] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], "beat durations do not match expected"
    assert [beat.value for beat in beats] == [
        1,
        2,
        3,
        4,
        1,
    ], "beat values do not match expected"
    assert [beat.confidence for beat in beats] == [
        None,
        None,
        None,
        None,
        None,
    ], "beat confidence does not match expected"
