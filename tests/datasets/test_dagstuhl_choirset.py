import os

import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import dagstuhl_choirset
from tests.test_utils import run_multitrack_tests, run_track_tests


def test_track():
    default_trackid = "DCS_LI_QuartetB_Take04_B2"
    data_home = os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset")
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "DCS_LI_QuartetB_Take04_B2",
        "audio_dyn_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "audio_wav_22050_mono/DCS_LI_QuartetB_Take04_B2_DYN.wav",
        ),
        "audio_hsm_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "audio_wav_22050_mono/DCS_LI_QuartetB_Take04_B2_HSM.wav",
        ),
        "audio_lrx_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "audio_wav_22050_mono/DCS_LI_QuartetB_Take04_B2_LRX.wav",
        ),
        "f0_crepe_dyn_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_F0_CREPE/DCS_LI_QuartetB_Take04_B2_DYN.csv",
        ),
        "f0_crepe_hsm_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_F0_CREPE/DCS_LI_QuartetB_Take04_B2_HSM.csv",
        ),
        "f0_crepe_lrx_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_F0_CREPE/DCS_LI_QuartetB_Take04_B2_LRX.csv",
        ),
        "f0_pyin_dyn_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_F0_PYIN/DCS_LI_QuartetB_Take04_B2_DYN.csv",
        ),
        "f0_pyin_hsm_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_F0_PYIN/DCS_LI_QuartetB_Take04_B2_HSM.csv",
        ),
        "f0_pyin_lrx_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_F0_PYIN/DCS_LI_QuartetB_Take04_B2_LRX.csv",
        ),
        "f0_manual_lrx_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_F0_manual/DCS_LI_QuartetB_Take04_B2_LRX.csv",
        ),
        "score_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_scorerepresentation/DCS_LI_QuartetB_Take04_Stereo_STM_B.csv",
        ),
    }

    expected_property_types = {
        "f0_crepe_dyn": annotations.F0Data,
        "f0_crepe_hsm": annotations.F0Data,
        "f0_crepe_lrx": annotations.F0Data,
        "f0_pyin_dyn": annotations.F0Data,
        "f0_pyin_hsm": annotations.F0Data,
        "f0_pyin_lrx": annotations.F0Data,
        "f0_manual_lrx": annotations.F0Data,
        "score": annotations.NoteData,
        "audio_dyn": tuple,
        "audio_hsm": tuple,
        "audio_lrx": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_audio_track():
    default_trackid = "DCS_LI_QuartetB_Take04_B2"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)

    y, sr = track.audio_dyn
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = track.audio_lrx
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = track.audio_hsm
    assert sr == 22050
    assert y.shape == (22050,)

    default_trackid = "DCS_SE_QuartetB_Tuning02_S1"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)
    assert track.audio_hsm is None


def test_mtrack_no_notes():
    default_mtrackid = "DCS_SE_Basses_Tuning01"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_mtrackid)
    assert mtrack.notes is None


def test_load_f0():
    f0_path = "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_CREPE/DCS_LI_QuartetB_Take04_B2_DYN.csv"
    f0 = dagstuhl_choirset.load_f0(f0_path)
    assert isinstance(f0, annotations.F0Data)

    assert np.array_equal(f0.times, np.array([0.0, 0.01, 0.02, 0.03, 0.04]))
    assert np.array_equal(
        f0.frequencies,
        np.array(
            [
                204.329447420415,
                205.1518935649711,
                205.52465246964104,
                205.45427855215388,
                205.51663864023027,
            ]
        ),
    )
    assert np.array_equal(
        f0._confidence,
        np.array(
            [
                0.050135254859924316,
                0.02886691689491272,
                0.0374043881893158,
                0.04053276777267456,
                0.053232729434967034,
            ]
        ),
    )

    f0_path = "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_manual/DCS_LI_QuartetB_Take04_B2_LRX.csv"
    f0 = dagstuhl_choirset.load_f0(f0_path)
    assert isinstance(f0, annotations.F0Data)

    assert np.array_equal(
        f0.times,
        np.array([0.400544218, 0.406349206, 0.412154195, 0.417959184, 0.423764172]),
    )
    assert np.array_equal(f0.frequencies, np.array([129.387, 126.634, 125.182, 124.943, 124.491]))
    assert np.array_equal(f0.voicing, np.array([1, 1, 1, 1, 1]))


def test_load_score():
    score_path = "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_scorerepresentation/DCS_LI_QuartetB_Take04_Stereo_STM_B.csv"
    score = dagstuhl_choirset.load_score(score_path)
    assert isinstance(score, annotations.NoteData)

    assert np.array_equal(
        score.intervals,
        np.array(
            [
                [0.1600, 2.5165],
                [2.5165, 3.3487],
                [3.3487, 4.8130],
                [4.8130, 5.7252],
                [5.7252, 6.3774],
            ]
        ),
    )

    assert np.allclose(
        score.pitches,
        np.array([130.81278265, 130.81278265, 130.81278265, 130.81278265, 130.81278265]),
    )
    assert score.confidence is None


def test_load_beat():
    beat_path = "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_beat/DCS_LI_QuartetB_Take04_Stereo_STM.csv"
    beat = dagstuhl_choirset.load_beat(beat_path)
    assert isinstance(beat, annotations.BeatData)

    assert np.array_equal(
        beat.times,
        np.array([0.174149660, 0.949115646, 1.724081633, 2.525170068, 3.308843537]),
    )

    assert np.array_equal(beat.positions, np.array([1, 2, 3, 4, 1]))


def test_score_track():
    default_trackid = "DCS_TP_FullChoir_Outtake_A1"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)

    assert track.score is None


def test_to_jams_track():
    default_trackid = "DCS_LI_QuartetB_Take04_B2"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam.validate()

    notes = jam.annotations[0]["data"]
    assert [note.time for note in notes] == [0.1600, 2.5165, 3.3487, 4.8130, 5.7252]
    assert [note.duration for note in notes] == (
        np.array([2.5165, 3.3487, 4.8130, 5.7252, 6.3774])
        - np.array([0.1600, 2.5165, 3.3487, 4.8130, 5.7252])
    ).tolist()
    assert [note.value for note in notes] == (
        2 ** ((np.array([48, 48, 48, 48, 48]) - 69) / 12) * 440
    ).tolist()

    f0s = jam.annotations[1]["data"]
    assert [f0.time for f0 in f0s] == [0.0, 0.01, 0.02, 0.03, 0.04]
    assert [f0.value["frequency"] for f0 in f0s] == [
        204.329447420415,
        205.1518935649711,
        205.52465246964104,
        205.45427855215388,
        205.51663864023027,
    ]
    assert [f0.confidence for f0 in f0s] == [
        0.050135254859924316,
        0.02886691689491272,
        0.0374043881893158,
        0.04053276777267456,
        0.053232729434967034,
    ]


def test_multitrack():
    default_trackid = "DCS_LI_QuartetB_Take04"
    data_home = os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset")
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    expected_attributes = {
        "mtrack_id": "DCS_LI_QuartetB_Take04",
        "audio_stm_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "audio_wav_22050_mono/DCS_LI_QuartetB_Take04_Stereo_STM.wav",
        ),
        "audio_str_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "audio_wav_22050_mono/DCS_LI_QuartetB_Take04_Stereo_STR.wav",
        ),
        "audio_stl_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "audio_wav_22050_mono/DCS_LI_QuartetB_Take04_Stereo_STL.wav",
        ),
        "audio_rev_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "audio_wav_22050_mono/DCS_LI_QuartetB_Take04_StereoReverb_STM.wav",
        ),
        "audio_spl_path": None,
        "audio_spr_path": None,
        "beat_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dagstuhl_choirset/"),
            "annotations_csv_beat/DCS_LI_QuartetB_Take04_Stereo_STM.csv",
        ),
        "track_ids": [
            "DCS_LI_QuartetB_Take04_A2",
            "DCS_LI_QuartetB_Take04_B2",
            "DCS_LI_QuartetB_Take04_S1",
            "DCS_LI_QuartetB_Take04_T2",
        ],
    }

    expected_property_types = {
        "tracks": dict,
        "track_audio_property": str,
        "beat": annotations.BeatData,
        "notes": annotations.NoteData,
        "multif0": annotations.MultiF0Data,
        "audio_stm": tuple,
        "audio_str": tuple,
        "audio_stl": tuple,
        "audio_rev": tuple,
        "audio_spl": type(None),
        "audio_spr": type(None),
    }

    run_track_tests(mtrack, expected_attributes, expected_property_types)
    run_multitrack_tests(mtrack)


def test_beat_multitrack():
    default_trackid = "DCS_LI_QuartetB_Solo03"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    assert mtrack.beat is None


def test_audio_multitrack():
    default_trackid = "DCS_LI_QuartetB_Take04"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    y, sr = mtrack.audio_stm
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = mtrack.audio_rev
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = mtrack.audio_stl
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = mtrack.audio_str
    assert sr == 22050
    assert y.shape == (22050,)

    assert mtrack.audio_spl is None


def test_to_jams_multitrack():
    default_trackid = "DCS_LI_QuartetB_Take04"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    jam = mtrack.to_jams()
    assert jam.validate()

    beats = jam.annotations[0]["data"]
    assert [beat.time for beat in beats] == [
        0.174149660,
        0.949115646,
        1.724081633,
        2.525170068,
        3.308843537,
    ]
    assert [beat.value for beat in beats] == [1, 2, 3, 4, 1]
