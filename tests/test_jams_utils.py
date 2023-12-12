import jams
import numpy as np
import pytest

from mirdata import annotations, jams_utils


def get_jam_data(jam, namespace, annot_numb):
    time = []
    duration = []
    value = []
    confidence = []
    for obs in jam.search(namespace=namespace)[annot_numb]["data"]:
        time.append(obs.time)
        duration.append(round(obs.duration, 3))
        value.append(obs.value)
        confidence.append(obs.confidence)
    return time, duration, value, confidence


def test_beats():
    beat_data_1 = [
        (
            annotations.BeatData(np.array([0.2, 0.3]), "s", np.array([1, 2]), "bar_index"),
            None,
        )
    ]
    beat_data_2 = [
        (
            annotations.BeatData(np.array([0.5, 0.7]), "s", np.array([2, 3]), "bar_index"),
            "beats_2",
        )
    ]
    beat_data_3 = [
        (
            annotations.BeatData(np.array([0.0, 0.3]), "s", np.array([1, 2]), "bar_index"),
            "beats_1",
        ),
        (
            annotations.BeatData(np.array([0.5, 0.8]), "s", np.array([4, 3]), "bar_index"),
            "beats_2",
        ),
    ]
    beat_data_4 = (
        annotations.BeatData(np.array([0.0, 0.3]), "s", np.array([1, 2]), "bar_index"),
        "beats_1",
    )
    beat_data_5 = [
        (
            annotations.BeatData(np.array([0.0, 0.3]), "s", np.array([1, 2]), "bar_index"),
            "beats_1",
        ),
        [
            annotations.BeatData(np.array([0.5, 0.8]), "s", np.array([4, 3]), "bar_index"),
            "beats_2",
        ],
    ]
    beat_data_6 = [(None, None)]
    beat_data_7 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                "s",
                ["event A", "event B"],
                "open",
            ),
            None,
        )
    ]
    beat_data_8 = [(annotations.BeatData(np.array([0.2, 0.3]), "s", None, "bar_index"), None)]

    jam_1 = jams_utils.jams_converter(beat_data=beat_data_1)
    jam_2 = jams_utils.jams_converter(beat_data=beat_data_2)
    jam_3 = jams_utils.jams_converter(beat_data=beat_data_3)
    jam_6 = jams_utils.jams_converter(beat_data=beat_data_6)
    jam_8 = jams_utils.jams_converter(beat_data=beat_data_8)

    time, duration, value, confidence = get_jam_data(jam_1, "beat", 0)
    assert time == [0.2, 0.3]
    assert duration == [0.0, 0.0]
    assert value == [1, 2]
    assert confidence == [None, None]

    assert jam_2.annotations[0]["sandbox"]["name"] == "beats_2"

    time, duration, value, confidence = get_jam_data(jam_3, "beat", 0)
    assert time == [0.0, 0.3]
    assert duration == [0.0, 0.0]
    assert value == [1, 2]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_3, "beat", 1)
    assert time == [0.5, 0.8]
    assert duration == [0.0, 0.0]
    assert value == [4, 3]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_6, "beat", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    time, duration, value, confidence = get_jam_data(jam_8, "beat", 0)
    assert time == [0.2, 0.3]
    assert duration == [0.0, 0.0]
    assert value == [None, None]
    assert confidence == [None, None]

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=beat_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=beat_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=beat_data_7)


def test_chords():
    chord_data_1 = [
        (
            annotations.ChordData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                ["A", "A", "E"],
                "harte",
            ),
            None,
        )
    ]
    chord_data_2 = [
        (
            annotations.ChordData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                ["A", "B", "C"],
                "harte",
            ),
            "chords_2",
        )
    ]
    chord_data_3 = [
        (
            annotations.ChordData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                ["A", "A", "E"],
                "harte",
            ),
            "chords_1",
        ),
        (
            annotations.ChordData(
                np.array([[0.0, 0.7, 1.0], [0.7, 1.0, 1.5]]).T,
                "s",
                ["A", "B", "C"],
                "harte",
            ),
            "chords_2",
        ),
    ]
    chord_data_4 = (
        annotations.ChordData(
            np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
            "s",
            ["A", "A", "E"],
            "harte",
        ),
        None,
    )
    chord_data_5 = [
        [
            annotations.ChordData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                ["A", "A", "E"],
                "harte",
            ),
            None,
        ],
        (
            annotations.ChordData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                ["A", "B", "C"],
                "harte",
            ),
            "chords_2",
        ),
    ]
    chord_data_6 = [(None, None)]
    chord_data_7 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                "s",
                ["event A", "event B"],
                "open",
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(chord_data=chord_data_1)
    jam_2 = jams_utils.jams_converter(chord_data=chord_data_2)
    jam_3 = jams_utils.jams_converter(chord_data=chord_data_3)
    jam_6 = jams_utils.jams_converter(chord_data=chord_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, "chord", 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == ["A", "A", "E"]
    assert confidence == [None, None, None]

    assert jam_2.annotations[0]["sandbox"]["name"] == "chords_2"

    time, duration, value, confidence = get_jam_data(jam_3, "chord", 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == ["A", "A", "E"]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, "chord", 1)
    assert time == [0.0, 0.7, 1.0]
    assert duration == [0.7, 0.3, 0.5]
    assert value == ["A", "B", "C"]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_6, "chord", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(chord_data=chord_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(chord_data=chord_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(chord_data=chord_data_7)


def test_notes():
    note_data_1 = [
        (
            annotations.NoteData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                np.array([1108.731, 1108.731, 1108.731]),
                "hz",
                np.array([1.0, 1.0, 1.0]),
                "binary",
            ),
            None,
        )
    ]
    note_data_2 = [
        (
            annotations.NoteData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                np.array([1108.731, 1108.731, 1108.731]),
                "hz",
                np.array([1.0, 1.0, 1.0]),
                "binary",
            ),
            "notes_2",
        )
    ]
    note_data_3 = [
        (
            annotations.NoteData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                np.array([1108.731, 1108.731, 1108.731]),
                "hz",
                np.array([1.0, 1.0, 1.0]),
                "binary",
            ),
            "notes_1",
        ),
        (
            annotations.NoteData(
                np.array([[0.0, 0.7, 1.0], [0.7, 1.0, 1.5]]).T,
                "s",
                np.array([1108.731, 1108.731, 1108.731]),
                "hz",
                np.array([1.0, 1.0, 1.0]),
                "binary",
            ),
            "notes_2",
        ),
    ]
    note_data_4 = (
        annotations.NoteData(
            np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
            "s",
            np.array([1108.731, 1108.731, 1108.731]),
            "hz",
            np.array([1.0, 1.0, 1.0]),
            "binary",
        ),
        None,
    )
    note_data_5 = [
        [
            annotations.NoteData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                np.array([1108.731, 1108.731, 1108.731]),
                "hz",
                np.array([1.0, 1.0, 1.0]),
                "binary",
            ),
            None,
        ],
        (
            annotations.NoteData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                "s",
                np.array([1108.731, 1108.731, 1108.731]),
                "hz",
                np.array([1.0, 1.0, 1.0]),
                "binary",
            ),
            "notes_2",
        ),
    ]
    note_data_6 = [(None, None)]
    note_data_7 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                "s",
                ["event A", "event B"],
                "open",
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(note_data=note_data_1)
    jam_2 = jams_utils.jams_converter(note_data=note_data_2)
    jam_3 = jams_utils.jams_converter(note_data=note_data_3)
    jam_6 = jams_utils.jams_converter(note_data=note_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, "note_hz", 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == [1108.731, 1108.731, 1108.731]
    assert confidence == [None, None, None]

    assert jam_2.annotations[0]["sandbox"]["name"] == "notes_2"

    time, duration, value, confidence = get_jam_data(jam_3, "note_hz", 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == [1108.731, 1108.731, 1108.731]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, "note_hz", 1)
    assert time == [0.0, 0.7, 1.0]
    assert duration == [0.7, 0.3, 0.5]
    assert value == [1108.731, 1108.731, 1108.731]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_6, "note_hz", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(note_data=note_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(note_data=note_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(note_data=note_data_7)


def test_sections():
    section_data_1 = [
        (
            annotations.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                "s",
                ["verse A", "verse B", "verse A"],
                "open",
            ),
            None,
        )
    ]
    section_data_2 = [
        (
            annotations.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                "s",
                ["verse A", "verse B", "verse A"],
                "open",
            ),
            "sections_2",
        )
    ]
    section_data_3 = [
        (
            annotations.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                "s",
                ["verse A", "verse B", "verse A"],
                "open",
            ),
            "sections_1",
        ),
        (
            annotations.SectionData(
                np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 30.0]]).T,
                "s",
                ["verse A", "verse B", "verse C"],
                "open",
            ),
            "sections_2",
        ),
    ]
    section_data_4 = (
        annotations.SectionData(
            np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
            "s",
            ["verse A", "verse B", "verse A"],
            "open",
        ),
        None,
    )
    section_data_5 = [
        [
            annotations.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                "s",
                ["verse A", "verse B", "verse A"],
                "open",
            ),
            None,
        ],
        (
            annotations.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                "s",
                ["verse A", "verse B", "verse A"],
                "open",
            ),
            "sections_2",
        ),
    ]
    section_data_6 = [(None, None)]
    section_data_7 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                "s",
                ["event A", "event B"],
                "open",
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(section_data=section_data_1)
    jam_2 = jams_utils.jams_converter(section_data=section_data_2)
    jam_3 = jams_utils.jams_converter(section_data=section_data_3)
    jam_6 = jams_utils.jams_converter(section_data=section_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, "segment", 0)
    assert time == [0.0, 10.0, 20.0]
    assert duration == [10.0, 10.0, 5.0]
    assert value == ["verse A", "verse B", "verse A"]
    assert confidence == [None, None, None]

    assert jam_2.annotations[0]["sandbox"]["name"] == "sections_2"

    time, duration, value, confidence = get_jam_data(jam_3, "segment", 0)
    assert time == [0.0, 10.0, 20.0]
    assert duration == [10.0, 10.0, 5.0]
    assert value == ["verse A", "verse B", "verse A"]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, "segment", 1)
    assert time == [0.0, 15.0, 20.0]
    assert duration == [15.0, 5.0, 10.0]
    assert value == ["verse A", "verse B", "verse C"]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_6, "segment", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(section_data=section_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(section_data=section_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(section_data=section_data_7)


def test_multi_sections():
    multi_section_data_1 = [
        (
            [
                (
                    annotations.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse A", "verse B", "verse A"],
                        "open",
                    ),
                    None,
                ),
                (
                    annotations.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse a", "verse b", "verse a"],
                        "open",
                    ),
                    None,
                ),
            ],
            None,
        )
    ]

    multi_section_data_2 = [
        (
            [
                (
                    annotations.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse A", "verse B", "verse A"],
                        "open",
                    ),
                    0,
                ),
                (
                    annotations.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse a", "verse b", "verse a"],
                        "open",
                    ),
                    1,
                ),
            ],
            "annotator_1",
        )
    ]
    multi_section_data_3 = [
        (
            [
                (
                    annotations.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse A", "verse B", "verse A"],
                        "open",
                    ),
                    0,
                ),
                (
                    annotations.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse a", "verse b", "verse a"],
                        "open",
                    ),
                    1,
                ),
            ],
            "annotator_1",
        ),
        (
            [
                (
                    annotations.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse A", "verse B", "verse A"],
                        "open",
                    ),
                    0,
                ),
                (
                    annotations.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse a", "verse b", "verse a"],
                        "open",
                    ),
                    1,
                ),
            ],
            "annotator_2",
        ),
    ]
    multi_section_data_4 = (
        [
            (
                annotations.SectionData(
                    np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                    "s",
                    ["verse A", "verse B", "verse A"],
                    "open",
                ),
                None,
            ),
            (
                annotations.SectionData(
                    np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                    "s",
                    ["verse a", "verse b", "verse a"],
                    "open",
                ),
                None,
            ),
        ],
        None,
    )
    multi_section_data_5 = [
        [
            [
                (
                    annotations.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse A", "verse B", "verse A"],
                        "open",
                    ),
                    None,
                ),
                (
                    annotations.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse a", "verse b", "verse a"],
                        "open",
                    ),
                    None,
                ),
            ],
            None,
        ]
    ]
    multi_section_data_6 = [
        (
            (
                (
                    annotations.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse A", "verse B", "verse A"],
                        "open",
                    ),
                    None,
                ),
                (
                    annotations.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        "s",
                        ["verse a", "verse b", "verse a"],
                        "open",
                    ),
                    None,
                ),
            ),
            None,
        )
    ]
    multi_section_data_7 = [([(None, None), (None, None)], None)]
    multi_section_data_8 = [
        (
            [
                (
                    annotations.EventData(
                        np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                        "s",
                        ["event A", "event B"],
                        "open",
                    ),
                    None,
                ),
                (
                    annotations.EventData(
                        np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                        "s",
                        ["event A", "event B"],
                        "open",
                    ),
                    None,
                ),
            ],
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(multi_section_data=multi_section_data_1)
    jam_2 = jams_utils.jams_converter(multi_section_data=multi_section_data_2)
    jam_3 = jams_utils.jams_converter(multi_section_data=multi_section_data_3)
    jam_7 = jams_utils.jams_converter(multi_section_data=multi_section_data_7)

    time, duration, value, confidence = get_jam_data(jam_1, "multi_segment", 0)
    assert time == [0.0, 0.0, 10.0, 15.0, 20.0, 20.0]
    assert duration == [10.0, 15.0, 10.0, 5.0, 5.0, 5.0]
    assert value == [
        {"label": "verse A", "level": None},
        {"label": "verse a", "level": None},
        {"label": "verse B", "level": None},
        {"label": "verse b", "level": None},
        {"label": "verse A", "level": None},
        {"label": "verse a", "level": None},
    ]
    assert confidence == [None, None, None, None, None, None]

    assert jam_2.annotations[0]["annotation_metadata"]["annotator"]["name"] == "annotator_1"

    time, duration, value, confidence = get_jam_data(jam_3, "multi_segment", 0)
    assert time == [0.0, 0.0, 10.0, 15.0, 20.0, 20.0]
    assert duration == [10.0, 15.0, 10.0, 5.0, 5.0, 5.0]
    assert value == [
        {"label": "verse A", "level": 0},
        {"label": "verse a", "level": 1},
        {"label": "verse B", "level": 0},
        {"label": "verse b", "level": 1},
        {"label": "verse A", "level": 0},
        {"label": "verse a", "level": 1},
    ]
    assert confidence == [None, None, None, None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, "multi_segment", 1)
    assert time == [0.0, 0.0, 10.0, 15.0, 20.0, 20.0]
    assert duration == [10.0, 15.0, 10.0, 5.0, 5.0, 5.0]
    assert value == [
        {"label": "verse A", "level": 0},
        {"label": "verse a", "level": 1},
        {"label": "verse B", "level": 0},
        {"label": "verse b", "level": 1},
        {"label": "verse A", "level": 0},
        {"label": "verse a", "level": 1},
    ]
    assert confidence == [None, None, None, None, None, None]

    time, duration, value, confidence = get_jam_data(jam_7, "multi_segment", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_6)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_8)


def test_keys():
    key_data_1 = [
        (
            annotations.KeyData(np.array([[0.0], [100.0]]).T, "s", ["A"], "key_mode"),
            None,
        )
    ]
    key_data_2 = [
        (
            annotations.KeyData(np.array([[0.0], [100.0]]).T, "s", ["A"], "key_mode"),
            "keys_1",
        )
    ]
    key_data_3 = [
        (
            annotations.KeyData(np.array([[0.0], [100.0]]).T, "s", ["A"], "key_mode"),
            "keys_1",
        ),
        (
            annotations.KeyData(np.array([[0.0], [50.0]]).T, "s", ["B"], "key_mode"),
            "keys_2",
        ),
    ]
    key_data_4 = (
        annotations.KeyData(np.array([[0.0], [100.0]]).T, "s", ["A"], "key_mode"),
        "keys_1",
    )
    key_data_5 = [
        [
            annotations.KeyData(np.array([[0.0], [100.0]]).T, "s", ["A"], "key_mode"),
            "keys_1",
        ],
        (
            annotations.KeyData(np.array([[0.0], [50.0]]).T, "s", ["B"], "key_mode"),
            "keys_2",
        ),
    ]
    key_data_6 = [(None, None)]
    key_data_7 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                "s",
                ["event A", "event B"],
                "open",
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(key_data=key_data_1)
    jam_2 = jams_utils.jams_converter(key_data=key_data_2)
    jam_3 = jams_utils.jams_converter(key_data=key_data_3)
    jam_6 = jams_utils.jams_converter(key_data=key_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, "key", 0)
    assert time == [0.0]
    assert duration == [100.0]
    assert value == ["A"]
    assert confidence == [None]

    assert jam_2.annotations[0]["sandbox"]["name"] == "keys_1"

    time, duration, value, confidence = get_jam_data(jam_3, "key", 0)
    assert time == [0.0]
    assert duration == [100.0]
    assert value == ["A"]
    assert confidence == [None]

    time, duration, value, confidence = get_jam_data(jam_3, "key", 1)
    assert time == [0.0]
    assert duration == [50.0]
    assert value == ["B"]
    assert confidence == [None]

    time, duration, value, confidence = get_jam_data(jam_6, "key", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(key_data=key_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(key_data=key_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(key_data=key_data_7)


def test_f0s():
    f0_data_1 = [
        (
            annotations.F0Data(
                np.array([0.016, 0.048]),
                "s",
                np.array([0.0, 260.9]),
                "hz",
                np.array([0.0, 1.0]),
                "binary",
            ),
            None,
        )
    ]
    f0_data_2 = [
        (
            annotations.F0Data(
                np.array([0.016, 0.048]),
                "s",
                np.array([0.0, 260.9]),
                "hz",
                np.array([0.0, 1.0]),
                "binary",
            ),
            "f0s_1",
        )
    ]
    f0_data_3 = [
        (
            annotations.F0Data(
                np.array([0.016, 0.048]),
                "s",
                np.array([0.0, 260.9]),
                "hz",
                np.array([0.0, 1.0]),
                "binary",
            ),
            "f0s_1",
        ),
        (
            annotations.F0Data(
                np.array([0.003, 0.012]),
                "s",
                np.array([0.0, 230.5]),
                "hz",
                np.array([0.0, 1.0]),
                "binary",
            ),
            "f0s_2",
        ),
    ]
    f0_data_4 = (
        annotations.F0Data(
            np.array([0.016, 0.048]),
            "s",
            np.array([0.0, 260.9]),
            "hz",
            np.array([0.0, 1.0]),
            "binary",
        ),
        "f0s_1",
    )
    f0_data_5 = [
        [
            annotations.F0Data(
                np.array([0.016, 0.048]),
                "s",
                np.array([0.0, 260.9]),
                "hz",
                np.array([0.0, 1.0]),
                "binary",
            ),
            "f0s_1",
        ],
        (
            annotations.F0Data(
                np.array([0.003, 0.012]),
                "s",
                np.array([0.0, 230.5]),
                "hz",
                np.array([0.0, 1.0]),
                "binary",
            ),
            "f0s_2",
        ),
    ]
    f0_data_6 = [(None, None)]
    f0_data_7 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                "s",
                ["event A", "event B"],
                "open",
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(f0_data=f0_data_1)
    jam_2 = jams_utils.jams_converter(f0_data=f0_data_2)
    jam_3 = jams_utils.jams_converter(f0_data=f0_data_3)
    jam_6 = jams_utils.jams_converter(f0_data=f0_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, "pitch_contour", 0)
    assert time == [0.016, 0.048]
    assert duration == [0.0, 0.0]
    assert value == [
        {"frequency": 0.0, "index": 0, "voiced": False},
        {"frequency": 260.9, "index": 0, "voiced": True},
    ]
    assert confidence == [None, None]

    assert jam_2.annotations[0]["sandbox"]["name"] == "f0s_1"

    time, duration, value, confidence = get_jam_data(jam_3, "pitch_contour", 0)
    assert time == [0.016, 0.048]
    assert duration == [0.0, 0.0]
    assert value == [
        {"frequency": 0.0, "index": 0, "voiced": False},
        {"frequency": 260.9, "index": 0, "voiced": True},
    ]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_3, "pitch_contour", 1)
    assert time == [0.003, 0.012]
    assert duration == [0.0, 0.0]
    assert value == [
        {"frequency": 0.0, "index": 0, "voiced": False},
        {"frequency": 230.5, "index": 0, "voiced": True},
    ]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_6, "pitch_contour", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(f0_data=f0_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(f0_data=f0_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(f0_data=f0_data_7)


def test_lyrics():
    lyrics_data_1 = [
        (
            annotations.LyricData(
                np.array([[0.027, 0.232], [0.227, 0.742]]).T,
                "s",
                ["The", "Test"],
                "words",
            ),
            None,
        )
    ]
    lyrics_data_2 = [
        (
            annotations.LyricData(
                np.array([[0.027, 0.232], [0.227, 0.742]]).T,
                "s",
                ["The", "Test"],
                "words",
            ),
            "lyrics_1",
        )
    ]
    lyrics_data_3 = [
        (
            annotations.LyricData(
                np.array([[0.027, 0.232], [0.227, 0.742]]).T,
                "s",
                ["The", "Test"],
                "words",
            ),
            "lyrics_1",
        ),
        (
            annotations.LyricData(
                np.array([[0.0, 0.232], [0.227, 0.742]]).T, "s", ["is", "cool"], "words"
            ),
            "lyrics_2",
        ),
    ]
    lyrics_data_4 = (
        annotations.LyricData(
            np.array([[0.027, 0.232], [0.227, 0.742]]).T, "s", ["The", "Test"], "words"
        ),
        "lyrics_1",
    )
    lyrics_data_5 = [
        (
            annotations.LyricData(
                np.array([[0.027, 0.232], [0.227, 0.742]]).T,
                "s",
                ["The", "Test"],
                "words",
            ),
            "lyrics_1",
        ),
        [
            annotations.LyricData(
                np.array([[0.0, 0.232], [0.227, 0.742]]).T, "s", ["is", "cool"], "words"
            ),
            "lyrics_2",
        ],
    ]
    lyrics_data_6 = [(None, None)]
    lyrics_data_7 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]).T,
                "s",
                ["event A", "event B"],
                "open",
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(lyrics_data=lyrics_data_1)
    jam_2 = jams_utils.jams_converter(lyrics_data=lyrics_data_2)
    jam_3 = jams_utils.jams_converter(lyrics_data=lyrics_data_3)
    jam_6 = jams_utils.jams_converter(lyrics_data=lyrics_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, "lyrics", 0)
    assert time == [0.027, 0.232]
    assert duration == [0.2, 0.51]
    assert value == ["The", "Test"]
    assert confidence == [None, None]

    assert jam_2.annotations[0]["sandbox"]["name"] == "lyrics_1"

    time, duration, value, confidence = get_jam_data(jam_3, "lyrics", 0)
    assert time == [0.027, 0.232]
    assert duration == [0.2, 0.51]
    assert value == ["The", "Test"]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_3, "lyrics", 1)
    assert time == [0.0, 0.232]
    assert duration == [0.227, 0.51]
    assert value == ["is", "cool"]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_6, "lyrics", 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(lyrics_data=lyrics_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(lyrics_data=lyrics_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(lyrics_data=lyrics_data_7)


def test_tags():
    tag_data1 = [("blues", "I am a description")]
    tag_data2 = [("disco", "tag 1"), ("rock", "tag 2")]
    tag_data3 = [("invalid", "asdf")]
    tag_data4 = ("jazz", "wrong format")
    tag_data5 = ["wrong format too"]
    tag_data6 = [(123, "asdf")]
    jam1 = jams_utils.jams_converter(tags_gtzan_data=tag_data1, metadata={"duration": 10.0})
    assert jam1.validate()
    jam2 = jams_utils.jams_converter(tags_gtzan_data=tag_data2, metadata={"duration": 10.0})
    assert jam2.validate()
    jam3 = jams_utils.jams_converter(tags_gtzan_data=tag_data3, metadata={"duration": 10.0})
    with pytest.raises(jams.SchemaError):
        assert jam3.validate()
    with pytest.raises(TypeError):
        jams_utils.jams_converter(tags_gtzan_data=tag_data4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(tags_gtzan_data=tag_data5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(tags_gtzan_data=tag_data6)


def test_tempos():
    tempo_data1 = [(120, "I am a description")]
    tempo_data2 = [(120.0, "tempo 1"), (240, "tempo 2")]
    tempo_data3 = [(-1, "asdf")]
    tempo_data4 = (120.5, "wrong format")
    tempo_data5 = ["wrong format too"]
    tempo_data6 = [("string!", "string!")]
    jam1 = jams_utils.jams_converter(tempo_data=tempo_data1, metadata={"duration": 10.0})
    assert jam1.validate()
    jam2 = jams_utils.jams_converter(tempo_data=tempo_data2, metadata={"duration": 10.0})
    assert jam2.validate()
    jam3 = jams_utils.jams_converter(tempo_data=tempo_data3, metadata={"duration": 10.0})
    with pytest.raises(jams.SchemaError):
        assert jam3.validate()
    with pytest.raises(TypeError):
        jams_utils.jams_converter(tempo_data=tempo_data4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(tempo_data=tempo_data5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(tempo_data=tempo_data6)


def test_events():
    event_data1 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]), "s", ["event A", "event B"], "open"
            ),
            "I am a description",
        )
    ]
    event_data2 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]), "s", ["2", "event B"], "open"
            ),
            "events 1",
        ),
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 0.4]]), "s", ["", "a great label"], "open"
            ),
            "events 2",
        ),
    ]
    event_data3 = [
        (
            annotations.EventData(
                np.array([[0.2, 0.3], [0.3, 20.0]]),
                "s",  # invalid because > duration
                ["", "a great label"],
                "open",
            ),
            "asdf",
        )
    ]
    event_data4 = ("jazz", "wrong format")
    event_data5 = ["wrong format too"]
    event_data6 = [("wrong", "description")]
    jam1 = jams_utils.jams_converter(event_data=event_data1, metadata={"duration": 10.0})
    assert jam1.validate()
    jam2 = jams_utils.jams_converter(event_data=event_data2, metadata={"duration": 10.0})

    jam3 = jams_utils.jams_converter(event_data=event_data3, metadata={"duration": 10.0})

    with pytest.raises(TypeError):
        jams_utils.jams_converter(event_data=event_data4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(event_data=event_data5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(event_data=event_data6)


def test_metadata():
    metadata_1 = {
        "duration": 1.5,
        "artist": "Meatloaf",
        "title": "Le ciel est blue",
        "favourite_color": "rainbow",
    }

    jam_1 = jams_utils.jams_converter(lyrics_data=[(None, None)], metadata=metadata_1)

    assert jam_1["file_metadata"]["title"] == "Le ciel est blue"
    assert jam_1["file_metadata"]["artist"] == "Meatloaf"
    assert jam_1["file_metadata"]["duration"] == 1.5
    assert jam_1["sandbox"]["favourite_color"] == "rainbow"

    # test meatadata value None
    metadata_2 = {
        "duration": 1.5,
        "artist": "breakmaster cylinder",
        "title": None,
        "extra": None,
    }
    jam2 = jams_utils.jams_converter(metadata=metadata_2)
    assert jam2.validate()
    assert jam2["file_metadata"]["duration"] == 1.5
    assert jam2["file_metadata"]["artist"] == "breakmaster cylinder"
    assert jam2["file_metadata"]["title"] == ""
    assert "extra" not in jam2["sandbox"]


def test_duration():
    # duration from audio file
    jam = jams_utils.jams_converter(
        audio_path="tests/resources/mir_datasets/ikala/Wavfile/10161_chorus.wav"
    )
    assert jam.file_metadata.duration == 2.0
    assert jam.validate()

    # test invalid file path
    with pytest.raises(OSError):
        jams_utils.jams_converter(audio_path="i/dont/exist")

    jam1 = jams_utils.jams_converter(metadata={"duration": 4})
    assert jam1.file_metadata.duration == 4.0
    assert jam1.validate()

    # test incomplete metadata
    jam2 = jams_utils.jams_converter(metadata={"artist": "b"})
    with pytest.raises(jams_utils.jams.SchemaError):
        jam2.validate()

    # test metadata duration and audio file equal
    jam3 = jams_utils.jams_converter(
        audio_path="tests/resources/mir_datasets/ikala/Wavfile/10161_chorus.wav",
        metadata={"duration": 2},
    )
    assert jam3.file_metadata.duration == 2
    assert jam3.validate()

    # test metadata and duration not equal
    jam4 = jams_utils.jams_converter(
        audio_path="tests/resources/mir_datasets/ikala/Wavfile/10161_chorus.wav",
        metadata={"duration": 1000},
    )
    assert jam4.file_metadata.duration == 1000
    assert jam4.validate()


def test_mp3_remote_raises_error():
    with pytest.raises(NotImplementedError):
        jam = jams_utils.jams_converter(audio_path="gs://asdf/asdf.mp3")
