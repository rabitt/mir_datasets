import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import billboard


def test_track():
    default_trackid = "3"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    # test attributes are loaded as expected
    assert track.track_id == default_trackid
    assert track._data_home == data_home

    assert track._track_paths == {
        "audio": [
            "audio/1960s/James Brown/I Don't Mind/audio.flac",
            "bb9f022b25c43983cf19aef562b00eac",
        ],
        "salami": [
            "McGill-Billboard/0003/salami_chords.txt",
            "8deb413e4cecadcffa5a7180a5f4c597",
        ],
        "bothchroma": [
            "McGill-Billboard/0003/bothchroma.csv",
            "c92ee46045f5bacd681543e8b9aa55b8",
        ],
        "tuning": [
            "McGill-Billboard/0003/tuning.csv",
            "31c744b447b739bc8c4ed29891dc1fb1",
        ],
        "lab_full": [
            "McGill-Billboard/0003/full.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
        "lab_majmin7": [
            "McGill-Billboard/0003/majmin7.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
        "lab_majmin7inv": [
            "McGill-Billboard/0003/majmin7inv.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
        "lab_majmin": [
            "McGill-Billboard/0003/majmin.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
        "lab_majmininv": [
            "McGill-Billboard/0003/majmininv.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
    }

    assert track.title == "I Don't Mind"
    assert track.artist == "James Brown"

    # test that cached properties don't fail and have the expected type
    assert type(track.chords_full) is annotations.ChordData
    assert type(track.chords_majmin7) is annotations.ChordData
    assert type(track.chords_majmin7inv) is annotations.ChordData
    assert type(track.chords_majmin) is annotations.ChordData
    assert type(track.chords_majmininv) is annotations.ChordData
    assert type(track.chroma) is np.ndarray
    assert type(track.tuning) is list
    assert type(track.sections) is annotations.SectionData
    assert type(track.named_sections) is annotations.SectionData
    assert type(track.salami_metadata) is dict


def test_to_jams():
    default_trackid = "3"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    jam = track.to_jams()

    segments = jam.search(namespace="segment")[0]["data"]
    assert [segment.time for segment in segments] == [
        0.073469387,
        22.346394557,
        49.23802721,
        76.123990929,
        102.924353741,
        130.206598639,
    ]

    assert [segment.duration for segment in segments] == [
        22.27292517,
        26.891632653,
        26.885963719000003,
        26.800362812000003,
        27.282244897999988,
        20.70278911600002,
    ]

    assert [segment.value for segment in segments] == ["A", "B", "B", "A", "B", "A"]

    assert [segment.confidence for segment in segments] == [
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    named_segments = jam.search(namespace="segment")[1]["data"]
    assert [segment.value for segment in named_segments] == [
        "intro",
        "verse",
        "verse",
        "interlude",
        "verse",
        "interlude",
    ]

    assert jam["file_metadata"]["title"] == "I Don't Mind"
    assert jam["file_metadata"]["artist"] == "James Brown"

    chords = jam.search(namespace="chord")[0]["data"]
    assert [chord.value for chord in chords][:10] == [
        "N",
        "N",
        "N",
        "A:min",
        "A:min",
        "C:maj",
        "C:maj",
        "A:min",
        "A:min",
        "C:maj",
    ]

    chords = jam.search(namespace="chord")
    assert len(chords) == 5
    assert chords[0]["sandbox"]["name"] == "Full chords"


def test_load_chords():
    default_trackid = "35"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    full_chords = track.chords_full

    assert type(full_chords) == annotations.ChordData
    assert type(full_chords.intervals) is np.ndarray
    assert type(full_chords.labels) is list

    assert full_chords.labels[:36] == [
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "Eb:5",
        "C:5",
        "Eb:5",
        "C:5",
        "Eb:5",
        "C:5",
        "Eb:5",
        "C:5",
        "Eb:5",
        "C:5",
        "Eb:5",
        "C:5",
        "Eb:5",
        "C:5",
        "Eb:5",
        "C:5",
        "C:5",
    ]


def test_load_sections():
    default_trackid = "35"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    sections = track.sections

    assert type(sections) == annotations.SectionData
    assert type(sections.intervals) is np.ndarray
    assert type(sections.labels) is list

    assert np.array_equal(
        sections.labels,
        np.array(["A'", "A", "B", "C", "A", "B", "D", "E", "F", "A'", "B", "G", "Z"]),
    )

    named_sections = track.named_sections
    assert np.array_equal(
        named_sections.labels,
        np.array(
            [
                "intro",
                "verse",
                "chorus",
                "solo",
                "verse",
                "chorus",
                "trans",
                "bridge",
                "solo",
                "verse",
                "chorus",
                "outro",
                "fadeout",
            ]
        ),
    )

    with pytest.raises(ValueError):
        sections = billboard._load_sections(
            "tests/resources/mir_datasets/billboard/McGill-Billboard/0035/salami_chords.txt",
            "no_section_type",
        )


def test_load_chroma():
    default_trackid = "35"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    chroma = track.chroma
    assert chroma.shape[0] == 5666
    assert chroma.shape[1] == 25

    default_trackid = "3"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    chroma = track.chroma
    assert chroma.shape[0] == 3250
    assert chroma.shape[1] == 25


def test_load_tuning():
    default_trackid = "35"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    tuning = track.tuning

    assert type(tuning) == list
    assert len(tuning) == 4


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)

    metadata = dataset._metadata

    assert metadata["3"] == {
        "title": "I Don't Mind",
        "artist": "James Brown",
        "actual_rank": 57,
        "peak_rank": 47,
        "target_rank": 56,
        "weeks_on_chart": 8,
        "chart_date": "1961-07-03",
    }
