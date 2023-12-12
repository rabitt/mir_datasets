import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import rwc_classical
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "RM-C003"
    data_home = os.path.normpath("tests/resources/mir_datasets/rwc_classical")
    dataset = rwc_classical.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "RM-C003",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_classical/"),
            "audio/rwc-c-m01/3.wav",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_classical/"),
            "annotations/AIST.RWC-MDB-C-2001.CHORUS/RM-C003.CHORUS.TXT",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_classical/"),
            "annotations/AIST.RWC-MDB-C-2001.BEAT/RM-C003.BEAT.TXT",
        ),
        "piece_number": "No. 3",
        "suffix": "M01",
        "track_number": "Tr. 03",
        "title": "Symphony no.5 in C minor, op.67. 1st mvmt.",
        "composer": "Beethoven, Ludwig van",
        "artist": "Tokyo City Philharmonic Orchestra",
        "duration": 435,
        "category": "Symphony",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "sections": annotations.SectionData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/rwc_classical"
    dataset = rwc_classical.Dataset(data_home)
    track = dataset.track("RM-C003")
    jam = track.to_jams()

    beats = jam.search(namespace="beat")[0]["data"]
    assert [beat.time for beat in beats] == [
        1.65,
        2.58,
        2.95,
        3.33,
        3.71,
        4.09,
        5.18,
        6.28,
    ]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [2, 1, 2, 1, 2, 1, 2, 1]
    assert [beat.confidence for beat in beats] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    segments = jam.search(namespace="segment")[0]["data"]
    assert [segment.time for segment in segments] == [0.29, 419.96]
    assert [segment.duration for segment in segments] == [45.85, 13.75]
    assert [segment.value for segment in segments] == ["chorus A", "ending"]
    assert [segment.confidence for segment in segments] == [None, None]

    assert jam["file_metadata"]["title"] == "Symphony no.5 in C minor, op.67. 1st mvmt."
    assert jam["file_metadata"]["artist"] == "Tokyo City Philharmonic Orchestra"


def test_load_sections():
    # load a file which exists
    section_path = (
        "tests/resources/mir_datasets/rwc_classical/"
        + "annotations/AIST.RWC-MDB-C-2001.CHORUS/RM-C003.CHORUS.TXT"
    )
    section_data = rwc_classical.load_sections(section_path)

    # check types
    assert type(section_data) == annotations.SectionData
    assert type(section_data.intervals) is np.ndarray
    assert type(section_data.labels) is list

    # check values
    assert np.array_equal(section_data.intervals[:, 0], np.array([0.29, 419.96]))
    assert np.array_equal(section_data.intervals[:, 1], np.array([46.14, 433.71]))
    assert np.array_equal(section_data.labels, np.array(["chorus A", "ending"]))

    # empty file
    section_path = (
        "tests/resources/mir_datasets/rwc_classical/"
        + "annotations/AIST.RWC-MDB-C-2001.CHORUS/RM-C025_A.CHORUS.TXT"
    )

    section_data = rwc_classical.load_sections(section_path)

    assert section_data == None


def test_position_in_bar():
    positions1 = np.array([48, 384, 48, 384, 48, 384, 48, 384])
    times1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fixed_positions1 = np.array([2, 1, 2, 1, 2, 1, 2, 1])
    fixed_times1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    actual_positions1, actual_times1 = rwc_classical._position_in_bar(positions1, times1)
    assert np.array_equal(actual_positions1, fixed_positions1)
    assert np.array_equal(actual_times1, fixed_times1)

    positions2 = np.array([-1, 48, 384, 48, 384, 48, 384, 48])
    times2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fixed_positions2 = np.array([2, 1, 2, 1, 2, 1, 2])
    fixed_times2 = np.array([2, 3, 4, 5, 6, 7, 8])
    actual_positions2, actual_times2 = rwc_classical._position_in_bar(positions2, times2)
    assert np.array_equal(actual_positions2, fixed_positions2)
    assert np.array_equal(actual_times2, fixed_times2)

    positions3 = np.array([384, 48, 384, 48, 384, 48, 384, 48, 384])
    times3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    fixed_positions3 = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1])
    fixed_times3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    actual_positions3, actual_times3 = rwc_classical._position_in_bar(positions3, times3)
    assert np.array_equal(actual_positions3, fixed_positions3)
    assert np.array_equal(actual_times3, fixed_times3)

    positions4 = np.array([384, 24, 48, 72, 96, 120, 384, 24])
    times4 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fixed_positions4 = np.array([1, 2, 3, 4, 5, 6, 1, 2])
    fixed_times4 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    actual_positions4, actual_times4 = rwc_classical._position_in_bar(positions4, times4)
    assert np.array_equal(actual_positions4, fixed_positions4)
    assert np.array_equal(actual_times4, fixed_times4)

    positions5 = np.array([96, 384, 48, 96, 384, 48, 96])
    times5 = np.array([1, 2, 3, 4, 5, 6, 7])
    fixed_positions5 = np.array([3, 1, 2, 3, 1, 2, 3])
    fixed_times5 = np.array([1, 2, 3, 4, 5, 6, 7])
    actual_positions5, actual_times5 = rwc_classical._position_in_bar(positions5, times5)
    assert np.array_equal(actual_positions5, fixed_positions5)
    assert np.array_equal(actual_times5, fixed_times5)


def test_load_beats():
    beats_path = (
        "tests/resources/mir_datasets/rwc_classical/"
        + "annotations/AIST.RWC-MDB-C-2001.BEAT/RM-C003.BEAT.TXT"
    )
    beat_data = rwc_classical.load_beats(beats_path)

    # check types
    assert type(beat_data) is annotations.BeatData
    assert type(beat_data.times) is np.ndarray
    assert type(beat_data.positions) is np.ndarray

    # check values
    assert np.array_equal(
        beat_data.times, np.array([1.65, 2.58, 2.95, 3.33, 3.71, 4.09, 5.18, 6.28])
    )
    assert np.array_equal(beat_data.positions, np.array([2, 1, 2, 1, 2, 1, 2, 1]))


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/rwc_classical"
    dataset = rwc_classical.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["RM-C003"] == {
        "piece_number": "No. 3",
        "suffix": "M01",
        "track_number": "Tr. 03",
        "title": "Symphony no.5 in C minor, op.67. 1st mvmt.",
        "composer": "Beethoven, Ludwig van",
        "artist": "Tokyo City Philharmonic Orchestra",
        "duration": 435,
        "category": "Symphony",
    }
