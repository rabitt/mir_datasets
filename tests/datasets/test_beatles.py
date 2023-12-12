import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import beatles
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0111"
    data_home = os.path.normpath("tests/resources/mir_datasets/beatles")
    dataset = beatles.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatles/"),
            "audio/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.wav",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatles/"),
            "annotations/beat/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.txt",
        ),
        "chords_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatles/"),
            "annotations/chordlab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab",
        ),
        "keys_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatles/"),
            "annotations/keylab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/beatles/"),
            "annotations/seglab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab",
        ),
        "title": "11_-_Do_You_Want_To_Know_A_Secret",
        "track_id": "0111",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "chords": annotations.ChordData,
        "key": annotations.KeyData,
        "sections": annotations.SectionData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (44100 * 2,), "audio shape {} was not (88200,)".format(audio.shape)

    track = dataset.track("10212")
    assert track.beats is None, "expected track.beats to be None, got {}".format(track.beats)
    assert track.key is None, "expected track.key to be None, got {}".format(track.key)


def test_to_jams():
    data_home = os.path.normpath("tests/resources/mir_datasets/beatles")
    dataset = beatles.Dataset(data_home)
    track = dataset.track("0111")
    jam = track.to_jams()

    beats = jam.search(namespace="beat")[0]["data"]
    assert [beat.time for beat in beats] == [
        13.249,
        13.959,
        14.416,
        14.965,
        15.453,
        15.929,
        16.428,
    ], "beat times do not match expected"
    assert [beat.duration for beat in beats] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], "beat durations do not match expected"
    assert [beat.value for beat in beats] == [
        2,
        3,
        4,
        1,
        2,
        3,
        4,
    ], "beat values do not match expected"
    assert [beat.confidence for beat in beats] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ], "beat confidence does not match expected"

    segments = jam.search(namespace="segment")[0]["data"]
    assert [segment.time for segment in segments] == [
        0.0,
        0.465,
    ], "segment time does not match expected"
    assert [segment.duration for segment in segments] == [
        0.465,
        14.466,
    ], "segment duration does not match expected"
    assert [segment.value for segment in segments] == [
        "silence",
        "intro",
    ], "segment value does not match expected"
    assert [segment.confidence for segment in segments] == [
        None,
        None,
    ], "segment confidence does not match expected"

    chords = jam.search(namespace="chord")[0]["data"]
    assert [chord.time for chord in chords] == [
        0.0,
        4.586464,
        6.98973,
    ], "chord time does not match expected"
    assert [chord.duration for chord in chords] == [
        0.497838,
        2.4032659999999995,
        2.995374,
    ], "chord duration does not match expected"
    assert [chord.value for chord in chords] == [
        "N",
        "E:min",
        "G",
    ], "chord value does not match expected"
    assert [chord.confidence for chord in chords] == [
        None,
        None,
        None,
    ], "chord confidence does not match expected"

    keys = jam.search(namespace="key")[0]["data"]
    assert [key.time for key in keys] == [0.0], "key time does not match expected"
    assert [key.duration for key in keys] == [119.333], "key duration does not match expected"
    assert [key.value for key in keys] == ["E"], "key value does not match expected"
    assert [key.confidence for key in keys] == [None], "key confidence does not match expected"

    assert (
        jam["file_metadata"]["title"] == "11_-_Do_You_Want_To_Know_A_Secret"
    ), "title does not match expected"
    assert jam["file_metadata"]["artist"] == "The Beatles", "artist does not match expected"


def test_load_beats():
    beats_path = os.path.normpath(
        "tests/resources/mir_datasets/beatles/annotations/beat/"
        + "The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.txt"
    )
    beat_data = beatles.load_beats(beats_path)

    assert type(beat_data) == annotations.BeatData, "beat_data is not type annotations.BeatData"
    assert type(beat_data.times) == np.ndarray, "beat_data.times is not an np.ndarray"
    assert type(beat_data.positions) == np.ndarray, "beat_data.positions is not an np.ndarray"

    assert np.array_equal(
        beat_data.times,
        np.array([13.249, 13.959, 14.416, 14.965, 15.453, 15.929, 16.428]),
    ), "beat_data.times different than expected"
    assert np.array_equal(
        beat_data.positions, np.array([2, 3, 4, 1, 2, 3, 4])
    ), "beat_data.positions different from expected"

    assert beatles.load_beats(None) is None, "load_beats(None) should return None"


def test_load_chords():
    chords_path = os.path.normpath(
        "tests/resources/mir_datasets/beatles/annotations/chordlab/"
        + "The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab"
    )
    chord_data = beatles.load_chords(chords_path)

    assert type(chord_data) == annotations.ChordData
    assert type(chord_data.intervals) == np.ndarray
    assert type(chord_data.labels) == list

    assert np.array_equal(chord_data.intervals[:, 0], np.array([0.000000, 4.586464, 6.989730]))
    assert np.array_equal(chord_data.intervals[:, 1], np.array([0.497838, 6.989730, 9.985104]))
    assert np.array_equal(chord_data.labels, np.array(["N", "E:min", "G"]))

    assert beatles.load_chords(None) is None


def test_load_key():
    key_path = os.path.normpath(
        "tests/resources/mir_datasets/beatles/annotations/keylab/"
        + "The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab"
    )
    key_data = beatles.load_key(key_path)

    assert type(key_data) == annotations.KeyData
    assert type(key_data.intervals) == np.ndarray

    assert np.array_equal(key_data.intervals[:, 0], np.array([0.000]))
    assert np.array_equal(key_data.intervals[:, 1], np.array([119.333]))
    assert np.array_equal(key_data.keys, ["E"])

    assert beatles.load_key(None) is None


def test_load_sections():
    sections_path = os.path.normpath(
        "tests/resources/mir_datasets/beatles/annotations/seglab/"
        + "The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab"
    )
    section_data = beatles.load_sections(sections_path)

    assert type(section_data) == annotations.SectionData
    assert type(section_data.intervals) == np.ndarray
    assert type(section_data.labels) == list

    assert np.array_equal(section_data.intervals[:, 0], np.array([0.000000, 0.465]))
    assert np.array_equal(section_data.intervals[:, 1], np.array([0.465, 14.931]))
    assert np.array_equal(section_data.labels, np.array(["silence", "intro"]))

    assert beatles.load_sections(None) is None


def test_fix_newpoint():
    positions1 = np.array(["4", "1", "2", "New Point", "4"])
    new_positions1 = beatles._fix_newpoint(positions1)
    assert np.array_equal(new_positions1, np.array(["4", "1", "2", "3", "4"]))

    positions2 = np.array(["1", "2", "New Point"])
    new_positions2 = beatles._fix_newpoint(positions2)
    assert np.array_equal(new_positions2, np.array(["1", "2", "3"]))

    positions3 = np.array(["New Point", "2", "3"])
    new_positions3 = beatles._fix_newpoint(positions3)
    assert np.array_equal(new_positions3, np.array(["1", "2", "3"]))
