import os

import pretty_midi

from mirdata import annotations
from mirdata.datasets import slakh
from tests.test_utils import run_multitrack_tests, run_track_tests


def test_track():
    default_trackid = "Track00001-S00"
    data_home = os.path.normpath("tests/resources/mir_datasets/slakh")
    dataset = slakh.Dataset(data_home, version="test")

    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "Track00001-S00",
        "mtrack_id": "Track00001",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/slakh/"),
            "babyslakh_16k/Track00001/stems/S00.wav",
        ),
        "midi_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/slakh/"),
            "babyslakh_16k/Track00001/MIDI/S00.mid",
        ),
        "metadata_path": (
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/slakh/"),
                "babyslakh_16k/Track00001/metadata.yaml",
            )
        ),
        "instrument": "Guitar",
        "integrated_loudness": -12.82074180245363,
        "is_drum": False,
        "midi_program_name": "Distortion Guitar",
        "plugin_name": "elektrik_guitar.nkm",
        "program_number": 30,
        "mixing_group": "guitar",
        "data_split": None,
        "split": None,
    }

    expected_property_types = {
        "midi": pretty_midi.PrettyMIDI,
        "notes": annotations.NoteData,
        "multif0": annotations.MultiF0Data,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": [
            "babyslakh_16k/Track00001/stems/S00.wav",
            "ea0e7b3d996bb3fedfbf9ee43b5c414f",
        ],
        "midi": [
            "babyslakh_16k/Track00001/MIDI/S00.mid",
            "68f9d227a4fd70acdcd80a5bd3b69e22",
        ],
        "metadata": [
            "babyslakh_16k/Track00001/metadata.yaml",
            "ffde21b0625fd72ba04103ca55f6765d",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 16000
    assert audio.shape == (16000 * 2,)

    # test a track which has no notes
    track_id = "Track00007-S00"
    track = dataset.track(track_id)
    assert track.notes is None


def test_track_full():
    default_trackid = "Track00001-S00"
    data_home = os.path.normpath("tests/resources/mir_datasets/slakh")
    dataset_full = slakh.Dataset(data_home, version="2100-redux")
    track_full = dataset_full.track(default_trackid)

    expected_attributes = {
        "track_id": "Track00001-S00",
        "mtrack_id": "Track00001",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/slakh/"),
            "slakh2100_flac_redux/train/Track00001/stems/S00.flac",
        ),
        "midi_path": (
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/slakh/"),
                "slakh2100_flac_redux/train/Track00001/MIDI/S00.mid",
            )
        ),
        "metadata_path": (
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/slakh/"),
                "slakh2100_flac_redux/train/Track00001/metadata.yaml",
            )
        ),
        "instrument": "Guitar",
        "integrated_loudness": -12.82074180245363,
        "is_drum": False,
        "midi_program_name": "Distortion Guitar",
        "plugin_name": "elektrik_guitar.nkm",
        "program_number": 30,
        "mixing_group": "guitar",
        "data_split": "train",
        "split": "train",
    }

    expected_property_types = {
        "midi": pretty_midi.PrettyMIDI,
        "notes": annotations.NoteData,
        "multif0": annotations.MultiF0Data,
        "audio": tuple,
    }

    assert track_full._track_paths == {
        "audio": [
            "slakh2100_flac_redux/train/Track00001/stems/S00.flac",
            "bb4a50848831853a086e0f6e5b595804",
        ],
        "midi": [
            "slakh2100_flac_redux/train/Track00001/MIDI/S00.mid",
            "68f9d227a4fd70acdcd80a5bd3b69e22",
        ],
        "metadata": [
            "slakh2100_flac_redux/train/Track00001/metadata.yaml",
            "5258ffe8376e16e5e34b71e7323c0477",
        ],
    }

    run_track_tests(track_full, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track_full.audio
    assert sr == 44100
    assert audio.shape == (44100 * 2,)

    # this catches a bug, where we got the datasplit
    # logic wrong for the full version
    mtrack = dataset_full.multitrack("Track00001")
    assert mtrack.data_split == "train"


def test_load_tracks():
    # this test catches a bug we had when initializing tracks with no audio!
    dataset = slakh.Dataset(version="test")
    [track.data_split for track in dataset.load_tracks().values()]

    dataset = slakh.Dataset(version="2100-redux")
    [track.data_split for track in dataset.load_tracks().values()]


def test_to_jams():
    default_trackid = "Track00001-S00"
    data_home = "tests/resources/mir_datasets/slakh"
    dataset = slakh.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    notes = jam.annotations[0]["data"][:2]
    assert [annotation.time for annotation in notes] == [
        0.7811520833333333,
        1.2420318125,
    ]
    assert [annotation.duration for annotation in notes] == [
        0.4765027708333335,
        0.25778018749999987,
    ]


def test_multitrack():
    default_trackid = "Track00001"
    data_home = os.path.normpath("tests/resources/mir_datasets/slakh")
    dataset = slakh.Dataset(data_home, version="test")
    mtrack = dataset.multitrack(default_trackid)

    expected_attributes = {
        "mtrack_id": "Track00001",
        "midi_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/slakh/"),
            "babyslakh_16k/Track00001/all_src.mid",
        ),
        "mix_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/slakh/"),
            "babyslakh_16k/Track00001/mix.wav",
        ),
        "metadata_path": (
            os.path.join(
                os.path.normpath("tests/resources/mir_datasets/slakh/"),
                "babyslakh_16k/Track00001/metadata.yaml",
            )
        ),
        "data_split": None,
        "track_ids": [
            "Track00001-S00",
            "Track00001-S01",
            "Track00001-S02",
            "Track00001-S03",
            "Track00001-S04",
            "Track00001-S05",
            "Track00001-S07",
            "Track00001-S08",
            "Track00001-S09",
            "Track00001-S10",
        ],
        "lakh_midi_dir": (
            "lmd_matched/O/O/H/TROOHTB128F931F9DF/1a81ae092884234f3264e2f45927f00a.mid"
        ),
        "normalized": True,
        "overall_gain": 0.18270259567062658,
        "uuid": "1a81ae092884234f3264e2f45927f00a",
        "split": None,
    }

    expected_property_types = {
        "tracks": dict,
        "track_audio_property": str,
        "midi": pretty_midi.PrettyMIDI,
        "notes": annotations.NoteData,
        "multif0": annotations.MultiF0Data,
        "audio": tuple,
    }

    run_track_tests(mtrack, expected_attributes, expected_property_types)
    run_multitrack_tests(mtrack)

    # test submixing
    submixes, groups = mtrack.get_submix_by_group(["guitar", "drums"])
    assert list(submixes.keys()) == ["guitar", "drums", "other"]
    assert submixes["drums"].shape == (1, 2 * 16000)
    assert submixes["guitar"].shape == (1, 2 * 16000)
    assert submixes["other"].shape == (1, 2 * 16000)
    assert list(groups.keys()) == ["guitar", "drums", "other"]
    assert groups["guitar"] == ["Track00001-S00", "Track00001-S07", "Track00001-S08"]
    assert groups["drums"] == ["Track00001-S01"]
    assert groups["other"] == [
        "Track00001-S02",
        "Track00001-S03",
        "Track00001-S04",
        "Track00001-S05",
        "Track00001-S09",
        "Track00001-S10",
    ]

    submixes, groups = mtrack.get_submix_by_group(["asdf"])
    assert list(submixes.keys()) == ["asdf", "other"]
    assert submixes["asdf"] is None
    assert submixes["other"].shape == (1, 2 * 16000)
    assert groups["asdf"] == []
    assert groups["other"] == [
        "Track00001-S00",
        "Track00001-S01",
        "Track00001-S02",
        "Track00001-S03",
        "Track00001-S04",
        "Track00001-S05",
        "Track00001-S07",
        "Track00001-S08",
        "Track00001-S09",
        "Track00001-S10",
    ]


def test_multitrack_to_jams():
    default_mtrackid = "Track00001"
    data_home = "tests/resources/mir_datasets/slakh"
    dataset = slakh.Dataset(data_home, version="test")
    track = dataset.multitrack(default_mtrackid)
    jam = track.to_jams()

    notes = jam.annotations[0]["data"][:2]
    assert [annotation.time for annotation in notes] == [
        0.7811520833333333,
        1.2420318125,
    ]
    assert [annotation.duration for annotation in notes] == [
        0.4765027708333335,
        0.25778018749999987,
    ]
