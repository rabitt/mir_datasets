import math
import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import ikala
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "10161_chorus"
    data_home = os.path.normpath("tests/resources/mir_datasets/ikala")
    dataset = ikala.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "10161_chorus",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/ikala/"),
            "Wavfile/10161_chorus.wav",
        ),
        "song_id": "10161",
        "section": "chorus",
        "singer_id": "1",
        "f0_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/ikala/"),
            "PitchLabel/10161_chorus.pv",
        ),
        "lyrics_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/ikala/"),
            "Lyrics/10161_chorus.lab",
        ),
        "notes_pyin_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/ikala/"),
            "ikala-pyin-notes/10161_chorus_vamp_pyin_pyin_notes.csv",
        ),
    }

    expected_property_types = {
        "f0": annotations.F0Data,
        "notes_pyin": annotations.NoteData,
        "lyrics": annotations.LyricData,
        "pronunciations": annotations.LyricData,
        "vocal_audio": tuple,
        "instrumental_audio": tuple,
        "mix_audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["Wavfile/10161_chorus.wav", "278ae003cb0d323e99b9a643c0f2eeda"],
        "pitch": ["PitchLabel/10161_chorus.pv", "0d93a011a9e668fd80673049089bbb14"],
        "notes_pyin": [
            "ikala-pyin-notes/10161_chorus_vamp_pyin_pyin_notes.csv",
            "015856455537ab232140746f2df7b857",
        ],
        "lyrics": ["Lyrics/10161_chorus.lab", "79bbeb72b422056fd43be4e8d63319ce"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    vocal, sr_vocal = track.vocal_audio
    assert sr_vocal == 44100
    assert vocal.shape == (44100 * 2,)

    instrumental, sr_instrumental = track.instrumental_audio
    assert sr_instrumental == 44100
    assert instrumental.shape == (44100 * 2,)

    # make sure we loaded the correct channels to vocal/instrumental
    # (in this example, the first quarter second has only instrumentals)
    assert np.mean(np.abs(vocal[:8820])) < np.mean(np.abs(instrumental[:8820]))

    mix, sr_mix = track.mix_audio
    assert sr_mix == 44100
    assert mix.shape == (44100 * 2,)
    assert np.array_equal(mix, instrumental + vocal)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/ikala"
    default_trackid = "10161_chorus"
    dataset = ikala.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    lyrics = jam.search(namespace="lyric")[0]["data"]
    assert [lyric.time for lyric in lyrics] == [0.027, 0.232]
    assert [lyric.duration for lyric in lyrics] == [0.20500000000000002, 0.736]
    assert [lyric.value for lyric in lyrics] == ["JUST", "WANNA"]
    assert [lyric.confidence for lyric in lyrics] == [None, None]

    f0s = jam.search(namespace="pitch_contour")[0]["data"]
    assert [f0.time for f0 in f0s] == [0.016, 0.048]
    assert [f0.duration for f0 in f0s] == [0.0, 0.0]
    expected_f0s = [
        {"frequency": 0.0, "index": 0, "voiced": 0.0},
        {"frequency": 260.94640451888694, "index": 0, "voiced": 1.0},
    ]
    for i, f0 in enumerate(f0s):
        assert "frequency" in f0.value
        assert "index" in f0.value
        assert "voiced" in f0.value
        assert math.isclose(f0.value["frequency"], expected_f0s[i]["frequency"], rel_tol=1e-12)
        assert f0.value["index"] == expected_f0s[i]["index"]
        assert f0.value["voiced"] == expected_f0s[i]["voiced"]
    assert [f0.confidence for f0 in f0s] == [None, None]


def test_load_f0():
    # load a file which exists
    f0_path = "tests/resources/mir_datasets/ikala/PitchLabel/10161_chorus.pv"
    f0_data = ikala.load_f0(f0_path)

    # check types
    assert type(f0_data) == annotations.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.voicing) is np.ndarray

    # check values
    assert np.array_equal(f0_data.times, np.array([0.016, 0.048]))
    assert type(f0_data.frequencies) == np.ndarray
    assert f0_data.frequencies[0] == 0.0
    assert math.isclose(f0_data.frequencies[1], 260.94640451888694, rel_tol=1e-12)
    assert np.array_equal(f0_data.voicing, np.array([0.0, 1.0]))


def test_load_notes():
    notes_path = (
        "tests/resources/mir_datasets/ikala/ikala-pyin-notes/10161_chorus_vamp_pyin_pyin_notes.csv"
    )
    note_data = ikala.load_notes(notes_path)

    # check types
    assert type(note_data) == annotations.NoteData

    # check values
    assert np.allclose(
        note_data.intervals,
        np.array([[1.35256236, 1.7414966], [1.85759637, 1.9969161]]),
    )
    assert np.allclose(note_data.pitches, np.array([234.328, 261.992]))
    assert note_data.confidence is None


def test_load_lyrics():
    # load a file without pronunciations
    lyrics_path_simple = "tests/resources/mir_datasets/ikala/Lyrics/10161_chorus.lab"
    lyrics_data_simple = ikala.load_lyrics(lyrics_path_simple)
    pronunciation_data_simple = ikala.load_pronunciations(lyrics_path_simple)

    # check types
    assert type(lyrics_data_simple) is annotations.LyricData
    assert type(lyrics_data_simple.intervals) is np.ndarray
    assert type(lyrics_data_simple.lyrics) is list

    # check values
    assert np.array_equal(lyrics_data_simple.intervals[:, 0], np.array([0.027, 0.232]))
    assert np.array_equal(lyrics_data_simple.intervals[:, 1], np.array([0.232, 0.968]))
    assert np.array_equal(lyrics_data_simple.lyrics, ["JUST", "WANNA"])
    assert np.array_equal(pronunciation_data_simple.intervals[:, 0], np.array([0.027, 0.232]))
    assert np.array_equal(pronunciation_data_simple.intervals[:, 1], np.array([0.232, 0.968]))
    assert np.array_equal(pronunciation_data_simple.lyrics, ["", ""])

    # load a file with pronunciations
    lyrics_path_pronun = "tests/resources/mir_datasets/ikala/Lyrics/10164_chorus.lab"
    lyrics_data_pronun = ikala.load_lyrics(lyrics_path_pronun)
    pronun_data = ikala.load_pronunciations(lyrics_path_pronun)

    # check types
    assert type(lyrics_data_pronun) is annotations.LyricData
    assert type(lyrics_data_pronun.intervals) is np.ndarray
    assert type(lyrics_data_pronun.lyrics) is list

    # check values
    assert np.array_equal(lyrics_data_pronun.intervals[:, 0], np.array([0.021, 0.571]))
    assert np.array_equal(lyrics_data_pronun.intervals[:, 1], np.array([0.189, 1.415]))
    assert np.array_equal(lyrics_data_pronun.lyrics, ["ASDF", "EVERYBODY"])
    assert np.array_equal(pronun_data.intervals[:, 0], np.array([0.021, 0.571]))
    assert np.array_equal(pronun_data.intervals[:, 1], np.array([0.189, 1.415]))
    assert np.array_equal(pronun_data.lyrics, ["t i au", ""])


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/ikala"
    dataset = ikala.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["10161"] == "1"
    assert metadata["21025"] == "1"
