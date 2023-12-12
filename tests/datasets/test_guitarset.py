import os

import jams
import numpy as np

from mirdata import annotations
from mirdata.datasets import guitarset
from tests.test_utils import run_track_tests

TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets/guitarset")


def test_track():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "03_BN3-119-G_solo",
        "audio_hex_cln_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/guitarset/"),
            "audio_hex-pickup_debleeded/03_BN3-119-G_solo_hex_cln.wav",
        ),
        "audio_hex_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/guitarset/"),
            "audio_hex-pickup_original/03_BN3-119-G_solo_hex.wav",
        ),
        "audio_mic_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/guitarset/"),
            "audio_mono-mic/03_BN3-119-G_solo_mic.wav",
        ),
        "audio_mix_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/guitarset/"),
            "audio_mono-pickup_mix/03_BN3-119-G_solo_mix.wav",
        ),
        "jams_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/guitarset/"),
            "annotation/03_BN3-119-G_solo.jams",
        ),
        "player_id": "03",
        "tempo": 119,
        "mode": "solo",
        "style": "Bossa Nova",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "leadsheet_chords": annotations.ChordData,
        "inferred_chords": annotations.ChordData,
        "key_mode": annotations.KeyData,
        "pitch_contours": dict,
        "multif0": annotations.MultiF0Data,
        "notes": dict,
        "notes_all": annotations.NoteData,
        "audio_mic": tuple,
        "audio_mix": tuple,
        "audio_hex": tuple,
        "audio_hex_cln": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    assert track.pitch_contours["E"] is None
    assert track.notes["E"] is None

    assert isinstance(track.pitch_contours["e"], annotations.F0Data)
    assert isinstance(track.notes["e"], annotations.NoteData)


def test_notes_and_all_notes():
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track("03_BN3-119-G_solo")
    notes_all = track.notes_all
    for note in track.notes.values():
        if note is None:
            continue
        for interval, pitch in zip(note.intervals, note.pitches):
            assert interval in notes_all.intervals
            assert int(pitch) in notes_all.pitches.astype(int)
        assert note.interval_unit == notes_all.interval_unit
        assert note.pitch_unit == notes_all.pitch_unit
        assert note.confidence_unit == notes_all.confidence_unit


def test_contours_and_multif0():
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track("03_BN3-119-G_solo")
    multif0 = track.multif0
    for contour in track.pitch_contours.values():
        if contour is None:
            continue
        assert np.allclose(contour.times[:100], multif0.times[:100])
        for i, f in enumerate(contour.frequencies):
            if f > 0:
                assert f in multif0.frequency_list[i]
            else:
                assert f not in multif0.frequency_list[i]

        assert contour.time_unit == multif0.time_unit
        assert contour.frequency_unit == multif0.frequency_unit


def test_fill_pitch_contour():
    times = np.array([1, 3, 4, 6])
    freqs = np.array([40, 50, 60, 70])
    voicings = np.array([1, 1, 1, 1])
    t, f, v = guitarset._fill_pitch_contour(times, freqs, voicings, 7, 1)
    te = np.array([0, 1, 2, 3, 4, 5, 6])
    fe = np.array([0, 40, 0, 50, 60, 0, 70])
    ve = np.array([0, 1, 0, 1, 1, 0, 1])
    assert np.array_equal(t, te)
    assert np.array_equal(f, fe)
    assert np.array_equal(v, ve)

    t, f, v = guitarset._fill_pitch_contour(times, freqs, voicings, 8, 1, duration=7)
    assert np.array_equal(t, te)
    assert np.array_equal(f, fe)
    assert np.array_equal(v, ve)


def test_load_beats():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(track.beats.times, [0.50420168, 1.00840336, 1.51260504])
    assert np.allclose(track.beats.positions, np.array([2, 3, 4]))


def test_load_chords():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(track.leadsheet_chords.intervals[:, 0], [0])
    assert np.allclose(track.leadsheet_chords.intervals[:, 1], [2])
    assert track.leadsheet_chords.labels == ["G:maj"]

    assert np.allclose(track.inferred_chords.intervals[:, 0], [0])
    assert np.allclose(track.inferred_chords.intervals[:, 1], [2])
    assert track.inferred_chords.labels == ["G:maj7/1"]


def test_load_keys():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(track.key_mode.intervals[:, 0], [0])
    assert np.allclose(track.key_mode.intervals[:, 1], [2])
    assert track.key_mode.keys == ["G:major"]


def test_load_contours():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(
        track.pitch_contours["e"].times[:10],
        [
            0.0,
            0.00580499,
            0.01160998,
            0.01741497,
            0.02321995,
            0.02902494,
            0.03482993,
            0.04063492,
            0.04643991,
            0.0522449,
        ],
    )
    assert np.allclose(
        track.pitch_contours["e"].frequencies[:10],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert track.pitch_contours["e"]._confidence is None


def test_load_notes():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(
        track.notes["e"].intervals[:, 0],
        [0.7612308390022235, 1.5072852607709137, 1.7806185941042258],
    )
    assert np.allclose(
        track.notes["e"].intervals[:, 1], [1.2604598639455844, 1.7336798185940552, 2.0]
    )
    assert np.allclose(
        track.notes["e"].pitches,
        [67.0576287044242, 71.03221526299762, 71.03297250121584],
    )
    assert track.notes["e"].confidence is None


def test_audio_mono():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    # test audio loading functions
    y, sr = track.audio_mic
    assert sr == 44100
    assert y.shape == (44100 * 2,)
    y, sr = track.audio_mix
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_audio_hex():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    y, sr = track.audio_hex
    assert sr == 44100
    assert y.shape == (6, int(44100 * 0.5))


def test_audio_hex_cln():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    y, sr = track.audio_hex_cln
    assert sr == 44100
    assert y.shape == (6, int(44100 * 0.5))


def test_to_jams():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset("tests/resources/mir_datasets/guitarset")
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert type(jam) == jams.JAMS
