import numpy as np
from mirdata.datasets import billboard
from mirdata import annotations


def test_track():

    default_trackid = "3"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    dataset.download()

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
            "annotation/0003/salami_chords.txt",
            "8deb413e4cecadcffa5a7180a5f4c597",
        ],
        "bothchroma": [
            "annotation/0003/bothchroma.csv",
            "c92ee46045f5bacd681543e8b9aa55b8",
        ],
        "tuning": ["annotation/0003/tuning.csv", "31c744b447b739bc8c4ed29891dc1fb1"],
        "lab_full": ["annotation/0003/full.lab", "59c73209de645ef7e4e4293f4d6882b3"],
        "lab_majmin7": [
            "annotation/0003/majmin7.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
        "lab_majmin7inv": [
            "annotation/0003/majmin7inv.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
        "lab_majmin": [
            "annotation/0003/majmin.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
        "lab_majmininv": [
            "annotation/0003/majmininv.lab",
            "59c73209de645ef7e4e4293f4d6882b3",
        ],
    }

    assert track.title == "I Don't Mind"
    assert track.artist == "James Brown"

    # test that cached properties don't fail and have the expected type
    assert type(track.sections) is annotations.SectionData


# def test_to_jams():
#     default_trackid = "3"
#     data_home = "tests/resources/mir_datasets/billboard"
#     dataset = billboard.Dataset(data_home)
#     track = dataset.track(default_trackid)


#     jam = track.to_jams()

#     segments = jam.search(namespace="segment")[0]["data"]
#     print([segment.time for segment in segments])
#     assert [segment.time for segment in segments] == [
#         0.073469387,
#         22.346394557,
#         49.23802721,
#         76.123990929,
#         102.924353741,
#         130.206598639,
#     ]

#     assert [segment.duration for segment in segments] == [
#         22.27292517,
#         26.891632653,
#         26.885963719000003,
#         26.800362812000003,
#         27.282244897999988,
#         20.70278911600002,
#     ]

#     assert [segment.value for segment in segments] == [
#         ("A", "intro"),
#         ("B", "verse"),
#         ("B", "verse"),
#         ("A", "interlude"),
#         ("B", "verse"),
#         ("A", "interlude"),
#     ]

#     assert [segment.confidence for segment in segments] == [
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#     ]

#     assert jam["file_metadata"]["title"] == "I Don't Mind"
#     assert jam["file_metadata"]["artist"] == "James Brown"


def test_load_chords():
    default_trackid = "35"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    full_chords = track.chords_full

    # check types
    assert type(full_chords) == annotations.ChordData
    assert type(full_chords.intervals) is np.ndarray
    assert type(full_chords.labels) is list

    # check values
    assert full_chords.labels == [
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
        "Eb:5",
        "F:5",
        "G:5",
        "Bb:5",
        "C:5",
        "F#:5",
        "F#:5",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "F:7",
        "Bb:7",
        "Bb:7",
        "Eb:5",
        "D:5",
        "C:5",
        "Bb:5",
        "Ab:5",
        "G:5",
        "F:5",
        "Eb:5",
        "Bb:7",
        "Bb:7",
        "Eb:5",
        "D:5",
        "C:5",
        "Bb:5",
        "Ab:5",
        "G:5",
        "F:5",
        "Eb:5",
        "F:5",
        "G:5",
        "Ab:5",
        "C:5",
        "Db:5",
        "Bb:5",
        "Ab:7",
        "A:5",
        "Bb:5",
        "B:5",
        "C:5",
        "C:5",
        "C:7(#9)",
        "C:7(#9)",
        "C:5",
        "C:5",
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
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:7(#9)",
        "C:1/1",
        "C:1/1",
        "C:5",
        "F:5",
        "Bb:5",
        "Eb:5",
        "Ab:5",
        "Db:5",
        "G:5",
        "C:5",
        "Eb:5",
        "F#:5",
        "A:5",
        "C:7",
        "C:7",
        "N",
        "N",
        "N",
        "N",
    ]


def test_load_sections():
    default_trackid = "35"
    data_home = "tests/resources/mir_datasets/billboard"
    dataset = billboard.Dataset(data_home)
    track = dataset.track(default_trackid)

    sections = track.sections

    # check types
    assert type(sections) == annotations.SectionData
    assert type(sections.intervals) is np.ndarray
    assert type(sections.labels) is list

    # check values
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
