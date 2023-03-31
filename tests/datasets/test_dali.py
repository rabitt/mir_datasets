import logging
import os

import pytest

try:
    import DALI
except ImportError:
    raise ImportError(
        "An error occured when importing this dataset. Most likely this is due to a dependency not being installed, in this case DALI."
    )

from mirdata.datasets import dali
from mirdata import annotations
from tests.test_utils import run_track_tests
import numpy as np


def test_track():
    default_trackid = "4b196e6c99574dd49ad00d56e132712b"
    data_home = os.path.normpath("tests/resources/mir_datasets/dali")
    dataset = dali.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "album": "Mezmerize",
        "annotation_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dali/"),
            "annotations/4b196e6c99574dd49ad00d56e132712b.gz",
        ),
        "artist": "System Of A Down",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/dali/"),
            "audio/4b196e6c99574dd49ad00d56e132712b.mp3",
        ),
        "audio_url": "zUzd9KyIDrM",
        "dataset_version": 1,
        "genres": ["Pop", "Rock", "Hard Rock", "Metal"],
        "ground_truth": False,
        "language": "english",
        "release_date": "2005",
        "scores_manual": 0,
        "scores_ncc": 0.9644769596900552,
        "title": "B.Y.O.B.",
        "track_id": "4b196e6c99574dd49ad00d56e132712b",
        "url_working": True,
    }

    expected_property_types = {
        "notes": annotations.NoteData,
        "words": annotations.LyricData,
        "lines": annotations.LyricData,
        "paragraphs": annotations.LyricData,
        "annotation_object": DALI.Annotations,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    path_save = "/home/mfuentes/astre/code/repositories/mirdata/tests/resources/mir_datasets/dali/annotations"
    name = "test"
    track.annotation_object.write_json(path_save, name)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (94208,)


def test_load_notes():
    notes_path = (
        "tests/resources/mir_datasets/dali/annotations/"
        + "4b196e6c99574dd49ad00d56e132712b.gz"
    )
    note_data = dali.load_annotations_granularity(notes_path, "notes")

    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) == np.ndarray
    assert type(note_data.pitches) == np.ndarray

    assert np.array_equal(note_data.intervals[:, 0], np.array([24.125, 24.273, 24.420]))
    assert np.array_equal(note_data.intervals[:, 1], np.array([24.273, 24.420, 24.568]))
    assert np.array_equal(note_data.pitches, np.array([1108.731, 1108.731, 1108.731]))


def test_load_words():
    data_path = (
        "tests/resources/mir_datasets/dali/annotations/"
        + "4b196e6c99574dd49ad00d56e132712b.gz"
    )
    word_data = dali.load_annotations_granularity(data_path, "words")

    assert type(word_data) == annotations.LyricData
    assert type(word_data.intervals) == np.ndarray
    assert type(word_data.lyrics) == list

    assert np.array_equal(word_data.intervals[:, 0], np.array([24.125, 24.273, 24.42]))
    assert np.array_equal(word_data.intervals[:, 1], np.array([24.273, 24.42, 24.568]))
    assert np.array_equal(word_data.lyrics, ["why", "do", "they"])


def test_load_lines():
    data_path = (
        "tests/resources/mir_datasets/dali/annotations/"
        + "4b196e6c99574dd49ad00d56e132712b.gz"
    )
    line_data = dali.load_annotations_granularity(data_path, "lines")

    assert type(line_data) == annotations.LyricData
    assert type(line_data.intervals) == np.ndarray
    assert type(line_data.lyrics) == list

    assert np.array_equal(line_data.intervals[:, 0], np.array([24.125, 24.42]))
    assert np.array_equal(line_data.intervals[:, 1], np.array([24.42, 24.568]))
    assert np.array_equal(line_data.lyrics, ["why do", "they"])


def test_load_paragraphs():
    data_path = (
        "tests/resources/mir_datasets/dali/annotations/"
        + "4b196e6c99574dd49ad00d56e132712b.gz"
    )
    par_data = dali.load_annotations_granularity(data_path, "paragraphs")

    assert type(par_data) == annotations.LyricData
    assert type(par_data.intervals) == np.ndarray
    assert type(par_data.lyrics) == list

    assert np.array_equal(par_data.intervals[:, 0], np.array([24.125, 24.420]))
    assert np.array_equal(par_data.intervals[:, 0], np.array([24.125, 24.420]))
    assert np.array_equal(par_data.intervals[:, 1], np.array([24.420, 24.568]))
    assert np.array_equal(par_data.lyrics, ["why do", "they"])


def test_load_dali_object():
    data_path = (
        "tests/resources/mir_datasets/dali/annotations/"
        + "4b196e6c99574dd49ad00d56e132712b.gz"
    )
    dali_data = dali.load_annotations_class(data_path)

    assert type(dali_data) == DALI.Annotations
    assert dali_data.annotations["annot"]["notes"] == [
        {
            "text": "why",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.12471002069169, 24.272507833284063],
            "index": 0,
        },
        {
            "text": "do",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.272507833284063, 24.42030564587644],
            "index": 1,
        },
        {
            "text": "they",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.42030564587644, 24.568103458468812],
            "index": 2,
        },
    ]
    assert dali_data.annotations["annot"]["words"] == [
        {
            "text": "why",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.12471002069169, 24.272507833284063],
            "index": 0,
        },
        {
            "text": "do",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.272507833284063, 24.42030564587644],
            "index": 0,
        },
        {
            "text": "they",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.42030564587644, 24.568103458468812],
            "index": 1,
        },
    ]
    assert dali_data.annotations["annot"]["lines"] == [
        {
            "text": "why do",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.12471002069169, 24.42030564587644],
            "index": 0,
        },
        {
            "text": "they",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.42030564587644, 24.568103458468812],
            "index": 1,
        },
    ]
    assert dali_data.annotations["annot"]["paragraphs"] == [
        {
            "text": "why do",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.12471002069169, 24.42030564587644],
        },
        {
            "text": "they",
            "freq": [1108.7305239074883, 1108.7305239074883],
            "time": [24.42030564587644, 24.568103458468812],
        },
    ]
