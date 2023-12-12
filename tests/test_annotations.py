import mir_eval
import numpy as np
import pytest

import mirdata
from mirdata import annotations


def test_repr():
    class TestAnnotation(annotations.Annotation):
        def __init__(self):
            self.a = ["a", "b", "c"]
            self.b = np.array([[1, 2], [1, 4]])
            self._c = "hidden"

    test_track = TestAnnotation()
    assert test_track.__repr__() == """TestAnnotation(a, b)"""

    beat_data = annotations.BeatData(np.array([1.0, 2.0]), "s", np.array([1, 2]), "bar_index")
    assert (
        beat_data.__repr__()
        == "BeatData(confidence, confidence_unit, " + "position_unit, positions, time_unit, times)"
    )


def test_beat_data():
    times = np.array([1.0, 2.0])
    positions = np.array([3, 4])
    beat_data = annotations.BeatData(times, "s", positions, "bar_index")
    assert np.allclose(beat_data.times, times)
    assert np.allclose(beat_data.positions, positions)

    with pytest.raises(ValueError):
        annotations.BeatData(None, None, None, None)

    with pytest.raises(TypeError):
        annotations.BeatData([1.0, 2.0], "s", [1, 2], "bar_index")

    with pytest.raises(TypeError):
        annotations.BeatData(times, "s", [3, 4], "bar_index")

    with pytest.raises(TypeError):
        annotations.BeatData(times.astype(int), "s", [3, 4], "bar_index")

    with pytest.raises(ValueError):
        annotations.BeatData(times, "s", np.array([1]), "bar_index")


def test_section_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    labels = ["a", "b", "c"]
    section_data = annotations.SectionData(intervals, "s")
    assert np.allclose(section_data.intervals, intervals)
    assert section_data.labels is None
    section_data2 = annotations.SectionData(intervals, "s", labels, "open")
    assert np.allclose(section_data2.intervals, intervals)
    assert section_data2.labels == labels

    with pytest.raises(ValueError):
        annotations.SectionData(None, None)

    with pytest.raises(TypeError):
        annotations.SectionData([1.0, 2.0], "s")

    with pytest.raises(TypeError):
        annotations.SectionData(intervals, "s", np.array(labels), "open")

    with pytest.raises(TypeError):
        annotations.SectionData(intervals.astype(int), "s")

    with pytest.raises(ValueError):
        annotations.SectionData(intervals, "s", ["a"], "open")

    with pytest.raises(ValueError):
        annotations.SectionData(np.array([1.0, 2.0, 3.0]), "s")

    with pytest.raises(ValueError):
        annotations.SectionData(np.array([[1.0, 2.0], [2.0, 1.0]]), "s")


def test_note_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    notes = np.array([100.0, 150.0, 120.0])
    confidence = np.array([0.1, 0.4, 0.2])
    note_data = annotations.NoteData(intervals, "s", notes, "hz")
    assert np.allclose(note_data.intervals, intervals)
    assert np.allclose(note_data.pitches, notes)
    assert note_data.confidence is None
    note_data2 = annotations.NoteData(intervals, "s", notes, "hz", confidence, "likelihood")
    assert np.allclose(note_data2.intervals, intervals)
    assert np.allclose(note_data2.pitches, notes)
    assert np.allclose(note_data2.confidence, confidence)

    intervals_dup = np.array([[1.0, 2.0], [1.5, 3.0], [1.0, 2.0], [2.0, 3.0]])
    notes_dup = np.array([100.0, 150.0, 100.0, 120.0])
    confidence_dup = np.array([0.1, 0.4, 0.5, 0.2])
    note_data_dup = annotations.NoteData(intervals_dup, "s", notes_dup, "hz")
    assert np.allclose(note_data_dup.intervals, intervals)
    assert np.allclose(note_data_dup.pitches, notes)
    assert note_data_dup.confidence is None
    note_data2_dup = annotations.NoteData(
        intervals_dup, "s", notes_dup, "hz", confidence_dup, "likelihood"
    )
    assert np.allclose(note_data2_dup.intervals, intervals)
    assert np.allclose(note_data2_dup.pitches, notes)
    assert np.allclose(note_data2_dup.confidence, confidence)

    with pytest.raises(ValueError):
        annotations.NoteData(None, "s", notes, "hz")

    with pytest.raises(ValueError):
        annotations.NoteData(intervals, "s", None, "hz")

    with pytest.raises(TypeError):
        annotations.NoteData([1.0, 2.0], "s", notes, "hz")

    with pytest.raises(TypeError):
        annotations.NoteData(intervals, "s", [3.0, 4.0], "hz")

    with pytest.raises(TypeError):
        annotations.NoteData(intervals.astype(int), "s", notes, "hz")

    with pytest.raises(ValueError):
        annotations.NoteData(intervals, "s", np.array([1.0]), "hz")

    # test add
    note_data_add = note_data + note_data
    assert np.allclose(note_data_add.intervals, intervals)
    assert np.allclose(note_data_add.pitches, notes)
    assert note_data_add.confidence is None

    note_data_add = note_data + None
    assert np.allclose(note_data_add.intervals, intervals)
    assert np.allclose(note_data_add.pitches, notes)
    assert note_data_add.confidence is None

    note_data2_add = note_data2 + note_data2
    assert np.allclose(note_data2_add.intervals, intervals)
    assert np.allclose(note_data2_add.pitches, notes)
    assert np.allclose(note_data2_add.confidence, confidence)

    note_data_add = note_data + note_data2
    assert np.allclose(note_data_add.intervals, intervals)
    assert np.allclose(note_data_add.pitches, notes)
    assert note_data_add.confidence is None

    note_data_add = annotations.NoteData(
        np.array([[1.0, 2.0]]),
        "s",
        np.array([32.7]),
        "hz",
        np.array([0.1]),
        "likelihood",
    ) + annotations.NoteData(
        np.array([[100, 1000.0]]),
        "ms",
        np.array([20.0]),
        "midi",
        np.array([127.0]),
        "velocity",
    )
    assert np.allclose(note_data_add.intervals, np.array([[0.1, 1.0], [1.0, 2.0]]))
    assert np.allclose(note_data_add.pitches, np.array([25.9565436, 32.7]))
    assert np.allclose(note_data_add.confidence, np.array([1.0, 0.1]))
    assert note_data_add.interval_unit == "s"
    assert note_data_add.pitch_unit == "hz"
    assert note_data_add.confidence_unit == "likelihood"

    with pytest.raises(TypeError):
        note_data + 1

    # test to_sparse index
    time_scale = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    frequency_scale = np.array([50.0, 90.0, 130.0])
    sparse_index, conf = note_data.to_sparse_index(time_scale, "s", frequency_scale, "hz")
    expected_index = [[2, 1], [3, 1], [4, 1], [4, 2], [5, 2]]
    expected_conf = np.array([1, 1, 1, 1, 1])
    assert np.allclose(sparse_index, expected_index)
    assert np.allclose(conf, expected_conf)

    sparse_index, conf = note_data.to_sparse_index(
        np.array([1.0, 1.5, 2.0, 2.5, 3.0]), "s", frequency_scale, "hz"
    )
    expected_index = [[0, 1], [1, 1], [2, 1], [2, 2], [3, 2], [4, 2]]
    expected_conf = np.array([1, 1, 1, 1, 1, 1])
    assert np.allclose(sparse_index, expected_index)
    assert np.allclose(conf, expected_conf)

    sparse_index, conf = note_data2.to_sparse_index(
        time_scale, "s", frequency_scale, "hz", "likelihood"
    )
    expected_index = [[2, 1], [3, 1], [4, 1], [4, 2], [5, 2]]
    expected_conf = np.array([0.1, 0.1, 0.1, 0.2, 0.2])
    assert np.allclose(sparse_index, expected_index)
    assert np.allclose(conf, expected_conf)

    sparse_index, conf = note_data.to_sparse_index(
        time_scale, "s", frequency_scale, "hz", "binary", True
    )
    expected_index = [[2, 1], [4, 2]]
    expected_conf = np.array([1, 1])
    assert np.allclose(sparse_index, expected_index)
    assert np.allclose(conf, expected_conf)

    sparse_index, conf = note_data2.to_sparse_index(
        time_scale, "s", frequency_scale, "hz", "likelihood", True
    )
    expected_index = [[2, 1], [4, 2]]
    expected_conf = np.array([0.1, 0.2])
    assert np.allclose(sparse_index, expected_index)
    assert np.allclose(conf, expected_conf)

    # test to matrix
    matrix = note_data.to_matrix(time_scale, "s", frequency_scale, "hz")
    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
    assert np.allclose(matrix, expected)

    matrix = note_data2.to_matrix(time_scale, "s", frequency_scale, "hz", "likelihood")
    expected = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 0.1, 0], [0, 0.1, 0], [0, 0.1, 0.2], [0, 0, 0.2]]
    )
    assert np.allclose(matrix, expected)

    matrix = note_data.to_matrix(time_scale, "s", frequency_scale, "hz", "binary", True)
    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert np.allclose(matrix, expected)

    matrix = note_data2.to_matrix(time_scale, "s", frequency_scale, "hz", "likelihood", True)
    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0.1, 0], [0, 0, 0], [0, 0, 0.2], [0, 0, 0]])
    assert np.allclose(matrix, expected)

    # test to_multif0
    mf0_data = note_data2.to_multif0(0.5, "s")
    assert mf0_data.time_unit == "s"
    assert mf0_data.frequency_unit == "hz"
    assert mf0_data.confidence_unit == "likelihood"
    assert np.allclose(mf0_data.times, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    assert mf0_data.frequency_list == [
        [],
        [],
        [100.0],
        [100.0, 150.0],
        [100.0, 150.0, 120.0],
        [150.0, 120.0],
        [150.0, 120.0],
    ]
    assert mf0_data.confidence_list == [
        [],
        [],
        [0.1],
        [0.1, 0.4],
        [0.1, 0.4, 0.2],
        [0.4, 0.2],
        [0.4, 0.2],
    ]

    mf0_data = note_data.to_multif0(500, "ms", max_time=3500.0)
    assert mf0_data.time_unit == "ms"
    assert mf0_data.frequency_unit == "hz"
    assert mf0_data.confidence_unit is None
    assert np.allclose(
        mf0_data.times,
        np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0]),
    )
    assert mf0_data.frequency_list == [
        [],
        [],
        [100.0],
        [100.0, 150.0],
        [100.0, 150.0, 120.0],
        [150.0, 120.0],
        [150.0, 120.0],
        [],
    ]
    assert mf0_data.confidence_list is None

    with pytest.raises(ValueError):
        mf0_data = note_data.to_multif0(0.5, "s", max_time=2.5)

    # test to mireval
    note_data = annotations.NoteData(
        intervals, "s", notes, "hz", np.array([0.0, 1.0, 1.0]), "binary"
    )
    intervals_me, pitches_me, velocity_me = note_data.to_mir_eval()
    assert np.allclose(intervals_me, intervals)
    assert np.allclose(pitches_me, notes)
    assert np.allclose(velocity_me, np.array([0, 127.0, 127.0]))
    scores = mir_eval.transcription.evaluate(intervals_me, pitches_me, intervals_me, pitches_me)
    scores = mir_eval.transcription_velocity.evaluate(
        intervals_me, pitches_me, velocity_me, intervals_me, pitches_me, velocity_me
    )

    note_data = annotations.NoteData(intervals, "ms", np.array([60.0, 70.0, 100.0]), "midi")
    intervals_me, pitches_me, velocity_me = note_data.to_mir_eval()
    assert np.allclose(intervals_me, np.array([[0.001, 0.002], [0.0015, 0.003], [0.002, 0.003]]))
    assert np.allclose(pitches_me, np.array([261.6255653, 466.16376152, 2637.0204553]))
    assert velocity_me is None
    mir_eval.transcription.evaluate(intervals_me, pitches_me, intervals_me, pitches_me)


def test_chord_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    labels = ["E:min", "A", "G:7"]
    confidence = np.array([0.1, 0.4, 0.2])
    chord_data = annotations.ChordData(intervals, "s", labels, "harte", confidence, "likelihood")
    assert np.allclose(chord_data.intervals, intervals)
    assert chord_data.labels == labels
    assert np.allclose(chord_data.confidence, confidence)


def test_f0_data():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    frequencies = np.array([100.0, 150.0, 0.0, 120.0])
    voicing = np.array([0.1, 0.4, 0.0, 0.2])
    confidence = np.array([0.0, 0.0, 1.0, 0.0])
    f0_data = annotations.F0Data(times, "s", frequencies, "hz", voicing, "likelihood")
    assert np.allclose(f0_data.times, times)
    assert np.allclose(f0_data.frequencies, frequencies)
    assert np.allclose(f0_data.voicing, voicing)

    voicing2 = np.array([1.0, 1.0, 0.0, 1.0])
    f0_data2 = annotations.F0Data(
        times, "s", frequencies, "hz", voicing2, "binary", confidence, "binary"
    )
    assert np.allclose(f0_data2.times, times)
    assert np.allclose(f0_data2.frequencies, frequencies)
    assert np.allclose(f0_data2.voicing, voicing2)
    assert np.allclose(f0_data2._confidence, confidence)

    f0_data3 = annotations.F0Data(
        times, "s", frequencies, "hz", voicing2, "binary", confidence, "likelihood"
    )

    with pytest.raises(ValueError):
        frequencies_bad = np.array([100.0, 0, 120.0])
        voicing_bad = np.array([1.0, 1.0, 1.0])
        annotations.F0Data(times, "s", frequencies_bad, "hz", voicing_bad, "binary")

    with pytest.raises(ValueError):
        times_bad = np.array([0.0, 1.0, 3.0, 4.0])
        annotations.F0Data(times_bad, "s", frequencies, "hz", voicing, "likelihood")

    # test resample
    new_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    resampled_f0 = f0_data.resample(new_times, "s")
    assert resampled_f0.time_unit == "s"
    assert resampled_f0.frequency_unit == f0_data.frequency_unit
    assert resampled_f0.voicing_unit == f0_data.voicing_unit
    assert resampled_f0.confidence_unit == f0_data.confidence_unit
    assert np.allclose(resampled_f0.times, new_times)
    assert np.allclose(
        resampled_f0.frequencies,
        np.array([0.0, 0.0, 100.0, 125.0, 150.0, 150.0, 0.0, 0.0]),
    )
    assert np.allclose(resampled_f0.voicing, np.array([0, 0, 0.1, 0.25, 0.4, 0.2, 0.0, 0.0]))
    assert resampled_f0._confidence is None

    resampled_f0 = f0_data2.resample(new_times, "s")
    assert resampled_f0.time_unit == "s"
    assert resampled_f0.frequency_unit == f0_data2.frequency_unit
    assert resampled_f0.voicing_unit == f0_data2.voicing_unit
    assert resampled_f0.confidence_unit == f0_data2.confidence_unit
    assert np.allclose(resampled_f0.times, new_times)
    assert np.allclose(
        resampled_f0.frequencies,
        np.array([0.0, 0.0, 100.0, 125.0, 150.0, 150.0, 0.0, 0.0]),
    )
    assert np.allclose(resampled_f0.voicing, np.array([0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))
    assert np.allclose(resampled_f0._confidence, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))

    resampled_f0 = f0_data3.resample(new_times, "s")
    assert resampled_f0.time_unit == "s"
    assert resampled_f0.frequency_unit == f0_data3.frequency_unit
    assert resampled_f0.voicing_unit == f0_data3.voicing_unit
    assert resampled_f0.confidence_unit == f0_data3.confidence_unit
    assert np.allclose(resampled_f0.times, new_times)
    assert np.allclose(
        resampled_f0.frequencies,
        np.array([0.0, 0.0, 100.0, 125.0, 150.0, 150.0, 0.0, 0.0]),
    )
    assert np.allclose(resampled_f0.voicing, np.array([0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))
    assert np.allclose(resampled_f0._confidence, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5]))

    f0_note_class = annotations.F0Data(
        times, "s", np.array(["A", "B", "B", "F#"]), "note_name", voicing, "likelihood"
    )
    with pytest.raises(NotImplementedError):
        f0_note_class.resample(new_times, "s")

    # test to_sparse_index
    time_scale = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    frequency_scale = np.array([50.0, 90.0, 130.0])
    sparse_index, voc = f0_data.to_sparse_index(
        time_scale, "s", frequency_scale, "hz", "likelihood"
    )
    expected_index = np.array([[2, 1], [3, 2]])
    np.allclose(sparse_index, expected_index)
    expected_voc = np.array([0.1, 0.25])
    assert np.allclose(voc, expected_voc)

    # test to_matrix
    matrix = f0_data.to_matrix(time_scale, "s", frequency_scale, "hz", "likelihood")
    expected_matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(matrix, expected_matrix)

    # test to mir_eval
    times_me, frequency_me, voicing_me = f0_data.to_mir_eval()
    assert np.allclose(times_me, times)
    assert np.allclose(frequencies, frequency_me)
    assert np.allclose(voicing, voicing_me)
    scores = mir_eval.melody.evaluate(
        times_me, frequency_me, times_me, frequency_me, voicing_me, voicing_me
    )

    # test to multif0
    mf0_data = f0_data.to_multif0()
    assert np.allclose(mf0_data.times, f0_data.times)
    assert mf0_data.time_unit == f0_data.time_unit
    assert mf0_data.frequency_list == [[100.0], [150.0], [], [120.0]]
    assert mf0_data.frequency_unit == f0_data.frequency_unit
    assert mf0_data.confidence_list is None
    assert mf0_data.confidence_unit == f0_data.confidence_unit

    mf0_data = f0_data2.to_multif0()
    assert np.allclose(mf0_data.times, f0_data2.times)
    assert mf0_data.time_unit == f0_data2.time_unit
    assert mf0_data.frequency_list == [[100.0], [150.0], [], [120.0]]
    assert mf0_data.frequency_unit == f0_data2.frequency_unit
    assert mf0_data.confidence_list == [[0.0], [0.0], [], [0.0]]
    assert mf0_data.confidence_unit == f0_data2.confidence_unit


def test_multif0_data():
    times = np.array([1.0, 2.0, 3.0])
    frequencies = [[100.0], [150.0, 120.0], []]
    confidence = [[0.1], [0.4, 0.2], []]
    f0_data = annotations.MultiF0Data(times, "s", frequencies, "hz", confidence, "likelihood")
    assert np.allclose(f0_data.times, times)
    assert f0_data.time_unit == "s"
    assert f0_data.frequency_list == frequencies
    assert f0_data.frequency_unit == "hz"
    assert f0_data.confidence_list == confidence
    assert f0_data.confidence_unit == "likelihood"

    f0_data2 = annotations.MultiF0Data(times, "s", frequencies, "hz")
    assert np.allclose(f0_data2.times, times)
    assert f0_data2.time_unit == "s"
    assert f0_data2.frequency_list == frequencies
    assert f0_data2.frequency_unit == "hz"
    assert f0_data2.confidence_list is None
    assert f0_data2.confidence_unit is None

    # test duplicates
    times_dup = np.array([1.0, 2.0, 3.0])
    frequencies_dup = [[100.0], [150.0, 120.0, 150.0], []]
    confidence_dup = [[0.1], [0.4, 0.2, 0.3], []]
    f0_data_dup = annotations.MultiF0Data(
        times_dup, "s", frequencies_dup, "hz", confidence_dup, "likelihood"
    )
    assert np.allclose(f0_data_dup.times, times)
    assert f0_data_dup.time_unit == "s"
    assert f0_data_dup.frequency_list == frequencies
    assert f0_data_dup.frequency_unit == "hz"
    assert f0_data_dup.confidence_list == confidence
    assert f0_data_dup.confidence_unit == "likelihood"

    f0_data2_dup = annotations.MultiF0Data(times_dup, "s", frequencies_dup, "hz")
    assert np.allclose(f0_data2_dup.times, times)
    assert f0_data2_dup.time_unit == "s"
    assert f0_data2_dup.frequency_list == frequencies
    assert f0_data2_dup.frequency_unit == "hz"
    assert f0_data2_dup.confidence_list is None
    assert f0_data2_dup.confidence_unit is None

    # test resample
    time_scale = np.array([0.5, 1.0, 1.5])
    mf0_rsmp = f0_data.resample(time_scale, "s")
    assert mf0_rsmp.time_unit == "s"
    assert mf0_rsmp.frequency_unit == f0_data.frequency_unit
    assert mf0_rsmp.confidence_unit == f0_data.confidence_unit
    assert np.allclose(mf0_rsmp.times, time_scale)
    assert mf0_rsmp.frequency_list == [[], [100.0], [100.0]]
    assert mf0_rsmp.confidence_list == [[], [0.1], [0.1]]

    mf0_rsmp = f0_data2.resample(time_scale, "s")
    assert mf0_rsmp.time_unit == "s"
    assert mf0_rsmp.frequency_unit == f0_data2.frequency_unit
    assert mf0_rsmp.confidence_unit == f0_data2.confidence_unit
    assert np.allclose(mf0_rsmp.times, time_scale)
    assert mf0_rsmp.frequency_list == [[], [100.0], [100.0]]
    assert mf0_rsmp.confidence_list is None

    # test add
    mf0_add = f0_data + f0_data
    assert np.allclose(mf0_add.times, f0_data.times)
    assert mf0_add.time_unit == f0_data.time_unit
    assert mf0_add.frequency_list == f0_data.frequency_list
    assert mf0_add.frequency_unit == f0_data.frequency_unit
    assert mf0_add.confidence_list == f0_data.confidence_list
    assert mf0_add.confidence_unit == f0_data.confidence_unit

    mf0_add = f0_data2 + f0_data2
    assert np.allclose(mf0_add.times, f0_data2.times)
    assert mf0_add.time_unit == f0_data2.time_unit
    assert mf0_add.frequency_list == f0_data2.frequency_list
    assert mf0_add.frequency_unit == f0_data2.frequency_unit
    assert mf0_add.confidence_list == f0_data2.confidence_list
    assert mf0_add.confidence_unit == f0_data2.confidence_unit

    mf0_add = f0_data + None
    assert np.allclose(mf0_add.times, f0_data.times)
    assert mf0_add.time_unit == f0_data.time_unit
    assert mf0_add.frequency_list == f0_data.frequency_list
    assert mf0_add.frequency_unit == f0_data.frequency_unit
    assert mf0_add.confidence_list == f0_data.confidence_list
    assert mf0_add.confidence_unit == f0_data.confidence_unit

    mf0_add = f0_data + f0_data2
    assert np.allclose(mf0_add.times, f0_data.times)
    assert mf0_add.time_unit == f0_data.time_unit
    assert mf0_add.frequency_list == f0_data.frequency_list
    assert mf0_add.frequency_unit == f0_data.frequency_unit
    assert mf0_add.confidence_list is None
    assert mf0_add.confidence_unit is None

    mf0_add = f0_data + annotations.MultiF0Data(
        np.array([1000.0, 2000.0, 3000.0, 4000.0]),
        "ms",
        [[20.0], [30.0, 43.0], [], []],
        "midi",
    )
    assert np.allclose(mf0_add.times, np.array([1.0, 2.0, 3.0, 4.0]))
    assert mf0_add.time_unit == f0_data.time_unit
    assert mf0_add.frequency_list == [
        [100.0, 25.956543598746574],
        [150.0, 120.0, 46.2493028389543, 97.99885899543733],
        [],
        [],
    ]
    assert mf0_add.frequency_unit == f0_data.frequency_unit
    assert mf0_add.confidence_list is None
    assert mf0_add.confidence_unit is None

    # mf0 + F0Data
    mf0_add = f0_data + annotations.F0Data(
        np.array([1000.0, 2000.0, 3000.0, 4000.0]),
        "ms",
        np.array([20.0, 30.0, 0.0, 30.0]),
        "midi",
        np.array([1.0, 1.0, 0.0, 1.0]),
        "binary",
    )
    assert np.allclose(mf0_add.times, np.array([1.0, 2.0, 3.0, 4.0]))
    assert mf0_add.time_unit == f0_data.time_unit
    assert mf0_add.frequency_list == [
        [100.0, 25.956543598746574],
        [150.0, 120.0, 46.2493028389543],
        [],
        [46.2493028389543],
    ]
    assert mf0_add.frequency_unit == f0_data.frequency_unit
    assert mf0_add.confidence_list is None
    assert mf0_add.confidence_unit is None

    with pytest.raises(TypeError):
        f0_data + 1

    # test sparse index
    frequency_scale = np.array([50.0, 90.0, 130.0])
    sparse_idx, voc = f0_data.to_sparse_index(time_scale, "s", frequency_scale, "hz", "likelihood")
    sparse_idx_expected = np.array([[1, 1], [2, 1]])
    assert np.allclose(sparse_idx, sparse_idx_expected)
    voc_expected = np.array([0.1, 0.1])
    assert np.allclose(voc, voc_expected)

    sparse_idx, voc = f0_data2.to_sparse_index(time_scale, "s", frequency_scale, "hz", "binary")
    sparse_idx_expected = np.array([[1, 1], [2, 1]])
    assert np.allclose(sparse_idx, sparse_idx_expected)
    voc_expected = np.array([1.0, 1.0])
    assert np.allclose(voc, voc_expected)

    sparse_idx, voc = f0_data2.to_sparse_index(
        time_scale, "s", np.array([40.0, 50.0, 100.0]), "midi", "binary"
    )
    sparse_idx_expected = np.array([[1, 0], [2, 0]])
    assert np.allclose(sparse_idx, sparse_idx_expected)
    voc_expected = np.array([1.0, 1.0])
    assert np.allclose(voc, voc_expected)

    # test matrix
    matrix = f0_data.to_matrix(time_scale, "s", frequency_scale, "hz", "likelihood")
    matrix_expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.1, 0.0]])
    assert np.allclose(matrix, matrix_expected)

    matrix = f0_data2.to_matrix(time_scale, "s", frequency_scale, "hz", "binary")
    matrix_expected = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(matrix, matrix_expected)

    times_me, frequencies_me = f0_data.to_mir_eval()
    assert np.allclose(times_me, times)
    for flist, farr in zip(frequencies, frequencies_me):
        assert np.allclose(flist, farr)
    mir_eval.multipitch.evaluate(times_me, frequencies_me, times_me, frequencies_me)


def test_key_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    keys = ["E:minor", "A", "G"]
    key_data = annotations.KeyData(intervals, "s", keys, "key_mode")
    assert np.allclose(key_data.intervals, intervals)
    assert key_data.keys == keys


def test_lyric_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    lyrics = ["E:m", "A", "G:7"]
    lyric_data = annotations.LyricData(intervals, "s", lyrics, "words")
    assert np.allclose(lyric_data.intervals, intervals)
    assert lyric_data.lyrics == lyrics
    # deprecation test
    assert lyric_data.pronunciations == lyrics


def test_tempo_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    value = np.array([140.0, 110.0, 111.0])
    confidence = np.array([0.1, 0.4, 0.2])
    tempo_data = annotations.TempoData(intervals, "s", value, "bpm", confidence, "likelihood")
    assert np.allclose(tempo_data.intervals, intervals)
    assert np.allclose(tempo_data.tempos, value)
    assert np.allclose(tempo_data.confidence, confidence)

    with pytest.raises(ValueError):
        value = np.array([140.0, -150.0, 20.0])
        tempo_data = annotations.TempoData(intervals, "s", value, "bpm", confidence, "likelihood")


def test_event_data():
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    events = ["E:m", "A", "G:7"]
    event_data = annotations.EventData(intervals, "s", events, "open")
    assert np.allclose(event_data.intervals, intervals)
    assert event_data.events == events


def test_convert_time_units():
    times = np.array([100.0, 200.0])

    actual = annotations.convert_time_units(times, "s", "s")
    expected = times
    assert np.allclose(actual, expected)

    actual = annotations.convert_time_units(times, "ms", "ms")
    expected = times
    assert np.allclose(actual, expected)

    actual = annotations.convert_time_units(times, "ticks", "ticks")
    expected = times
    assert np.allclose(actual, expected)

    actual = annotations.convert_time_units(times, "s", "ms")
    expected = np.array([100000.0, 200000.0])
    assert np.allclose(actual, expected)

    actual = annotations.convert_time_units(times, "ms", "s")
    expected = np.array([0.1, 0.2])
    assert np.allclose(actual, expected)

    with pytest.raises(NotImplementedError):
        annotations.convert_time_units(times, "s", "ticks")

    with pytest.raises(NotImplementedError):
        annotations.convert_time_units(times, "ticks", "s")


def test_convert_pitch_units():
    pitches = np.array([100.0, 127.0])
    pitches_note = np.array(["A♯4", "A5"])
    pitches_class = np.array(["C", "Eb"])

    actual = annotations.convert_pitch_units(pitches, "hz", "hz")
    expected = pitches
    assert np.allclose(actual, expected)

    actual = annotations.convert_pitch_units(pitches, "midi", "midi")
    expected = pitches
    assert np.allclose(actual, expected)

    actual = annotations.convert_pitch_units(pitches_note, "note_name", "note_name")
    expected = pitches_note
    assert np.array_equal(actual, expected)

    actual = annotations.convert_pitch_units(pitches_class, "pc", "pc")
    expected = pitches_class
    assert np.array_equal(actual, expected)

    actual = annotations.convert_pitch_units(pitches, "hz", "midi")
    expected = np.array([43.34995772, 47.48789968])
    assert np.allclose(actual, expected)

    actual = annotations.convert_pitch_units(pitches, "midi", "hz")
    expected = np.array([2637.0204553, 12543.85395142])
    assert np.allclose(actual, expected)

    actual = annotations.convert_pitch_units(pitches, "hz", "note_name")
    expected = np.array(["G2", "B2"])
    assert np.array_equal(actual, expected)

    actual = annotations.convert_pitch_units(pitches_note, "note_name", "hz")
    expected = np.array([466.16376152, 880.0])
    assert np.allclose(actual, expected)

    actual = annotations.convert_pitch_units(pitches, "midi", "note_name")
    expected = np.array(["E7", "G9"])
    assert np.array_equal(actual, expected)

    actual = annotations.convert_pitch_units(pitches_note, "note_name", "midi")
    expected = np.array([70.0, 81.0])
    assert np.allclose(actual, expected)

    actual = annotations.convert_pitch_units([[100.0], [127.0], []], "hz", "midi")
    expected = [[43.34995771500077], [47.48789967897007], []]
    assert actual == expected

    with pytest.raises(NotImplementedError):
        annotations.convert_pitch_units(pitches_note, "note_name", "pc")

    with pytest.raises(NotImplementedError):
        annotations.convert_pitch_units(pitches_class, "pc", "hz")


def test_convert_amplitude_units():
    confidence = np.array([0.7, 1.0, 0.4, 0.0])
    conf_binary = np.array([0, 1])
    conf_velocity = np.array([50.0, 127, 0.0])

    actual = annotations.convert_amplitude_units(conf_binary, "binary", "binary")
    expected = conf_binary
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(confidence, "likelihood", "likelihood")
    expected = confidence
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(conf_velocity, "velocity", "velocity")
    expected = conf_velocity
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(conf_binary, "binary", "likelihood")
    expected = conf_binary
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(conf_binary, "binary", "velocity")
    expected = np.array([0, 127])
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(confidence, "likelihood", "binary")
    expected = np.array([1, 1, 1, 0.0])
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(confidence, "likelihood", "velocity")
    expected = np.array([88.9, 127.0, 50.8, 0])
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(
        [[0.7], [1.0, 0.4], [], [0.0]], "likelihood", "velocity"
    )
    expected = [[88.89999999999999], [127.0, 50.800000000000004], [], [0]]
    assert actual == expected

    actual = annotations.convert_amplitude_units(conf_velocity, "velocity", "binary")
    expected = np.array([1, 1, 0])
    assert np.allclose(actual, expected)

    actual = annotations.convert_amplitude_units(conf_velocity, "velocity", "likelihood")
    expected = np.array([0.39370079, 1, 0])
    assert np.allclose(actual, expected)

    with pytest.raises(NotImplementedError):
        annotations.convert_amplitude_units(conf_velocity, "velocity", "asdf")

    with pytest.raises(NotImplementedError):
        annotations.convert_amplitude_units(conf_velocity, "asdf", "velocity")


def test_closest_index():
    input_array = np.array([1, 5, 7, 2])[:, np.newaxis]
    target_array = np.array([2, 6])[:, np.newaxis]
    actual = annotations.closest_index(input_array, target_array)
    expected = np.array([-1, 1, -1, 0])
    assert np.array_equal(actual, expected)


def test_validate_array_like():
    with pytest.raises(ValueError):
        annotations.validate_array_like(None, list, str)

    annotations.validate_array_like(None, list, str, none_allowed=True)

    with pytest.raises(TypeError):
        annotations.validate_array_like([1, 2], np.ndarray, str)

    with pytest.raises(TypeError):
        annotations.validate_array_like([1, 2], list, str)

    with pytest.raises(TypeError):
        annotations.validate_array_like(np.array([1, 2]), np.ndarray, str)

    with pytest.raises(ValueError):
        annotations.validate_array_like([], list, int)


def test_validate_lengths_equal():
    annotations.validate_lengths_equal([np.array([0, 1])])
    annotations.validate_lengths_equal([np.array([]), None])
    annotations.validate_lengths_equal([np.array([0]), np.array([1]), np.array([2])])

    with pytest.raises(ValueError):
        annotations.validate_lengths_equal([np.array([0, 1]), np.array([0])])

    with pytest.raises(ValueError):
        annotations.validate_lengths_equal([np.array([0]), np.array([1]), np.array([2, 3])])


def test_validate_tempos():
    annotations.validate_tempos(np.array([120.0, 140.0]), "bpm")

    with pytest.raises(ValueError):
        annotations.validate_tempos(np.array([120.0, 140.0]), "asdf")

    with pytest.raises(ValueError):
        annotations.validate_tempos(np.array([120.0, -140.0]), "bpm")


def test_validate_beat_positions():
    annotations.validate_beat_positions(np.array([0, 1, 2, 3, 1, 2]), "bar_index")
    annotations.validate_beat_positions(np.array([0, 1, 2, 3, 4, 5]), "global_index")
    annotations.validate_beat_positions(np.array([0.0, 0.25, 0.5, 0.75, 0.0]), "bar_fraction")
    annotations.validate_beat_positions(np.array([1.0, 1.25, 1.5, 1.75, 2.0]), "global_fraction")

    with pytest.raises(ValueError):
        annotations.validate_beat_positions(np.array([0, 1, 2]), "asdf")

    with pytest.raises(ValueError):
        annotations.validate_beat_positions(np.array([[0, 1, 2]]), "bar_index")

    with pytest.raises(ValueError):
        annotations.validate_beat_positions(np.array([0, -1, 22]), "bar_index")

    with pytest.raises(ValueError):
        annotations.validate_beat_positions(np.array([0, 1.1, 2]), "bar_index")

    with pytest.raises(ValueError):
        annotations.validate_beat_positions(np.array([0, 1.1, 2]), "global_index")

    with pytest.raises(ValueError):
        annotations.validate_beat_positions(np.array([15, 16, 40]), "bar_index")

    with pytest.raises(ValueError):
        annotations.validate_beat_positions(np.array([0.5, 1.25, 1.5, 1.75, 2.0]), "bar_fraction")
    # check it doesn't break with empty positions
    assert annotations.validate_beat_positions(None, "bar_fraction") == None


def test_validate_confidence():
    annotations.validate_confidence(None, None)
    annotations.validate_confidence(np.array([0, 1, 1, 0]), "binary")
    annotations.validate_confidence(np.array([1, 1, 1, 1]), "binary")
    annotations.validate_confidence(np.array([0.1, 0, 0.5]), "likelihood")
    annotations.validate_confidence([[0.1, 0], [0.5]], "likelihood")
    annotations.validate_confidence(np.array([0.1, 0, 0.5]), "energy")
    annotations.validate_confidence(np.array([0, 57.3, 127]), "velocity")

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([0, 1, 1, 0.0]), "asdf")

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([[0, 1, 0, 2.0]]), "likelihood")

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([[0, 1, 0, -1.0]]), "likelihood")

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([[0, 1, 0, -1.0]]), "energy")

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([0, 2.0]), "binary")

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([-1, 60.0]), "velocity")

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([60, 128.0]), "velocity")


def test_validate_voicing():
    annotations.validate_voicing([0, 1, 0, 0], "binary")
    annotations.validate_voicing([0, 0, 0], "binary")
    annotations.validate_voicing([1, 1, 1], "binary")
    annotations.validate_voicing([1, 0.2, 0], "likelihood")

    with pytest.raises(ValueError):
        annotations.validate_voicing([1, 0], "asdf")

    with pytest.raises(ValueError):
        annotations.validate_voicing([[1, 0]], "binary")

    with pytest.raises(ValueError):
        annotations.validate_voicing([0.5, -0.5], "likelihood")

    with pytest.raises(ValueError):
        annotations.validate_voicing([0.5, 1.5], "likelihood")

    with pytest.raises(ValueError):
        annotations.validate_voicing([0, 2], "binary")


def test_validate_pitches():
    annotations.validate_pitches([330.2, 440.1], "hz")
    annotations.validate_pitches([[330.2], [440.1, 1203.0]], "hz")
    annotations.validate_pitches([33.5, 127], "midi")
    annotations.validate_pitches(["A", "B"], "pc")
    annotations.validate_pitches(["A4", "B5"], "note_name")

    with pytest.raises(ValueError):
        annotations.validate_pitches([330.2, 440.1], "asdf")

    with pytest.raises(ValueError):
        annotations.validate_pitches([330.2, -440.1], "hz")

    with pytest.raises(ValueError):
        annotations.validate_pitches([60, -60], "midi")

    with pytest.raises(ValueError):
        annotations.validate_pitches([60, 128], "midi")

    with pytest.raises(ValueError):
        annotations.validate_pitches(["X", "asdf"], "pc")

    with pytest.raises(ValueError):
        annotations.validate_pitches(["X", "asdf"], "note_name")


def test_validate_chord_labels():
    annotations.validate_chord_labels(["Ab:maj7", "D:min6/6"], "harte")
    annotations.validate_chord_labels(["Ab:maj7", "D:5"], "jams")
    annotations.validate_chord_labels(["asdf", "asdd"], "open")

    with pytest.raises(ValueError):
        annotations.validate_chord_labels(["A:maj7", "D:min6/6"], "asdf")

    with pytest.raises(ValueError):
        annotations.validate_chord_labels(["A:maj7", "D:5"], "harte")

    with pytest.raises(ValueError):
        annotations.validate_chord_labels(["A-:maj7", "D:min6/6"], "jams")


def test_validate_key_labels():
    annotations.validate_key_labels(["G#:minor", "Cb:major"], "key_mode")

    with pytest.raises(ValueError):
        annotations.validate_key_labels(["G#:minor", "Cb:major"], "asdf")

    with pytest.raises(ValueError):
        annotations.validate_key_labels(["G#:min", "Cb:major"], "key_mode")


def test_validate_times():
    annotations.validate_times(None, None)
    annotations.validate_times(np.array([1.0, 1.4, 1.6]), "s")
    annotations.validate_times(np.array([100, 140, 160]), "ms")
    annotations.validate_times(np.array([100, 140, 160]), "ticks")

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([1.0, 1.4, 1.6]), "asdf")

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([[0, 1], [0, 2]]), "s")

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([1.0, -1.4, 1.6]), "s")

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([2, 0]), "s")


def test_validate_intervals():
    annotations.validate_intervals(None, None)
    annotations.validate_intervals(np.array([[0.0, 0.2], [0.1, 0.3]]), "s")
    annotations.validate_intervals(np.array([[0.0, 0.2], [0.1, 0.3]]), "ms")
    annotations.validate_intervals(np.array([[0.0, 0.2], [0.1, 0.3]]), "ticks")

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([0, 2]), "asdf")

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([0, 2]), "s")

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([0, -2]), "s")

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([[0, 1], [1, 0.5]]), "s")


def test_validate_unit():
    annotations.validate_unit("a", {"a": "asdf", "b": "asdfd"})
    annotations.validate_unit(None, {"a": "asdf", "b": "asdfd"}, allow_none=True)

    with pytest.raises(ValueError):
        annotations.validate_unit("c", {"a": "asdf", "b": "asdfd"})

    with pytest.raises(ValueError):
        annotations.validate_unit(None, {"a": "asdf", "b": "asdfd"})
