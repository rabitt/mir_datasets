import os

import librosa
import numpy as np

from mirdata.datasets import mir_1k
from mirdata import annotations
from tests.test_utils import run_track_tests


DEFAULT_TRACK_ID = "amy_3_01"
DATA_HOME = os.path.normpath("tests/resources/mir_datasets/mir_1k")


def test_track():
    dataset = mir_1k.Dataset(DATA_HOME, version="test")
    track = dataset.track(DEFAULT_TRACK_ID)

    expected_attributes = {
        "track_id": DEFAULT_TRACK_ID,
        "audio_path": os.path.join(
            DATA_HOME,
            f"Wavfile/{DEFAULT_TRACK_ID}.wav",
        ),
        "f0_path": os.path.join(
            DATA_HOME,
            f"PitchLabel/{DEFAULT_TRACK_ID}.pv",
        ),
        "lyrics_path": os.path.join(
            DATA_HOME,
            f"Lyrics/{DEFAULT_TRACK_ID}.txt",
        ),
        "vocal_activity_path": os.path.join(
            DATA_HOME,
            f"vocal-nonvocalLabel/{DEFAULT_TRACK_ID}.vocal",
        ),
        "unvoiced_labels_path": os.path.join(
            DATA_HOME,
            f"UnvoicedFrameLabel/{DEFAULT_TRACK_ID}.unv",
        ),
    }

    expected_property_types = {
        "f0": annotations.F0Data,
        "lyrics": annotations.LyricData,
        "vocal_activity": annotations.EventData,
        "unvoiced_labels": annotations.EventData,
        "instrumental_audio": tuple,
        "vocal_audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["Wavfile/amy_3_01.wav", "60304d5698e632f9ee5df117da6e36d5"],
        "lyrics": ["Lyrics/amy_3_01.txt", "9eee41a192fe478cf913b4355ab6f255"],
        "f0": ["PitchLabel/amy_3_01.pv", "d1edb3d6bd5cd642bb35c0778291b850"],
        "unvoiced-category": [
            "UnvoicedFrameLabel/amy_3_01.unv",
            "9cb8e441a5f36cfcd4a7adee9b6a3717",
        ],
        "vocal-flag": [
            "vocal-nonvocalLabel/amy_3_01.vocal",
            "8d39472bd80d13eea08fe83978c86a03",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.instrumental_audio
    assert sr == 16000
    assert audio.shape == (31990,)

    audio, sr = track.vocal_audio
    assert sr == 16000
    assert audio.shape == (31990,)


def test_to_jams():
    dataset = mir_1k.Dataset(DATA_HOME, version="test")
    track = dataset.track(DEFAULT_TRACK_ID)
    jam = track.to_jams()

    jam_f0s = jam.search(namespace="pitch_contour")[0]["data"]

    annot_f0s = np.genfromtxt(track.f0_path)
    annot_f0s[annot_f0s > 0] = librosa.midi_to_hz(annot_f0s[annot_f0s > 0])

    assert np.all([f0.time for f0 in jam_f0s] == mir_1k.frame_timestamps(len(jam_f0s)))
    assert [f0.value for f0 in jam_f0s] == [
        {"frequency": annot_f0, "index": 0, "voiced": 0.0 if annot_f0 == 0.0 else 1.0}
        for annot_f0 in annot_f0s
    ]

    assert all(f0.duration == 0.0 for f0 in jam_f0s)
    assert all(f0.confidence is None for f0 in jam_f0s)


def test_load_f0():
    f0_path = os.path.join(DATA_HOME, "PitchLabel", f"{DEFAULT_TRACK_ID}.pv")
    f0_data = mir_1k.load_f0(f0_path)

    assert type(f0_data) == annotations.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.voicing) is np.ndarray

    assert f0_data.frequency_unit == "midi"
    assert f0_data.voicing_unit == "binary"
    assert f0_data.time_unit == "s"

    file_f0s = np.genfromtxt(f0_path)

    assert len(f0_data.times) == len(file_f0s)
    assert np.all(f0_data.times == mir_1k.frame_timestamps(len(file_f0s)))

    assert len(f0_data.frequencies) == len(file_f0s)
    assert np.all(f0_data.frequencies == file_f0s)

    assert len(f0_data.voicing) == len(file_f0s)
    assert np.all(f0_data.voicing == (file_f0s > 0).astype(np.float64))


def test_load_lyrics():
    lyrics_path = os.path.join(DATA_HOME, "Lyrics", f"{DEFAULT_TRACK_ID}.txt")
    lyrics = mir_1k.load_lyrics(lyrics_path)
    assert len(lyrics.lyrics) == 1
    assert lyrics.lyrics[0] == "今天晚上的星星很少"
    assert lyrics.lyric_unit == "words"
    assert np.all(lyrics.intervals == np.array([[0.0, 0.0]]))
    assert lyrics.interval_unit == "s"


def test_load_event_labels():
    unvoiced_labels_path = os.path.join(
        DATA_HOME, "UnvoicedFrameLabel", f"{DEFAULT_TRACK_ID}.unv"
    )
    unvoiced_labels = mir_1k.load_event_labels(unvoiced_labels_path)

    assert type(unvoiced_labels) == annotations.EventData
    assert unvoiced_labels.interval_unit == "s"

    labels = np.genfromtxt(unvoiced_labels_path, dtype=str)
    times = mir_1k.frame_timestamps(len(labels))

    for i, interval in enumerate(unvoiced_labels.intervals):
        from_idx = np.argmin(np.abs(times - interval[0]))
        to_idx = np.argmin(np.abs(times - interval[1]))

        assert np.all(labels[from_idx:to_idx] == unvoiced_labels.events[i])
