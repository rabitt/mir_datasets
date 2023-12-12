import tempfile
from io import BufferedReader, BytesIO, StringIO, TextIOWrapper

import numpy as np
import pytest

from mirdata import io


def test_load_midi():
    midi_file = (
        "tests/resources/mir_datasets/maestro/2018/"
        + "MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi"
    )
    midi = io.load_midi(midi_file)
    assert len(midi.instruments) == 1
    assert len(midi.instruments[0].notes) == 4197


def test_load_notes_from_midi():
    midi_file = (
        "tests/resources/mir_datasets/maestro/2018/"
        + "MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi"
    )
    notes_from_file = io.load_notes_from_midi(midi_file)
    midi = io.load_midi(midi_file)
    notes_from_midi = io.load_notes_from_midi(midi=midi)
    for note_data in [notes_from_file, notes_from_midi]:
        expected_intervals = np.array([[0.98307292, 1.80989583], [1.78385417, 1.90625]])
        assert np.allclose(note_data.intervals[0:2], expected_intervals)
        assert note_data.interval_unit == "s"
        assert np.allclose(note_data.pitches[0:2], np.array([67.0, 72.0]))
        assert note_data.pitch_unit == "midi"
        assert np.allclose(note_data.confidence[0:2], np.array([52.0, 67.0]))
        assert note_data.confidence_unit == "velocity"

    with pytest.raises(ValueError):
        io.load_notes_from_midi(None, None)


def test_load_multif0_from_midi():
    midi_file = "tests/resources/mir_datasets/slakh/babyslakh_16k/Track00001/MIDI/S08.mid"
    multif0_from_file = io.load_multif0_from_midi(midi_file)
    midi = io.load_midi(midi_file)
    multif0_from_midi = io.load_multif0_from_midi(midi=midi)
    for mf0_data in [multif0_from_file, multif0_from_midi]:
        assert np.allclose(mf0_data.times[2885:2887], np.array([22.5362376, 22.54404912]))
        assert mf0_data.time_unit == "s"
        assert mf0_data.frequency_list[2885:2887] == [[], [77.0, 89.0]]
        assert mf0_data.frequency_unit == "midi"
        assert mf0_data.confidence_list[2885:2887] == [[], [89.0, 89.0]]
        assert mf0_data.confidence_unit == "velocity"

    with pytest.raises(ValueError):
        io.load_notes_from_midi(None, None)


def test_coerce_to_string_with_none():
    @io.coerce_to_string_io
    def func(fh):
        assert fh is None

    func(None)


def test_coerce_to_string_io_with_path():
    with tempfile.NamedTemporaryFile(delete=False) as f:

        @io.coerce_to_string_io
        def func(fh):
            assert isinstance(fh, TextIOWrapper)

        func(f.name)


def test_coerce_to_string_io_with_stringio():
    @io.coerce_to_string_io
    def func(fh):
        assert isinstance(fh, StringIO)

    with StringIO("abc") as f:
        func(f)


def test_invalid_coerce_to_string_io():
    @io.coerce_to_string_io
    def func(fh):
        raise RuntimeError("YOU SHOULDNT BE HERE")

    with pytest.raises(ValueError):
        func(123)


def test_coerce_to_bytes_with_none():
    @io.coerce_to_bytes_io
    def func(fh):
        assert fh is None

    func(None)


def test_coerce_to_bytes_io_with_path():
    with tempfile.NamedTemporaryFile(delete=False) as f:

        @io.coerce_to_bytes_io
        def func(fh):
            assert isinstance(fh, BufferedReader)

        func(f.name)


def test_coerce_to_bytes_io_with_bytesio():
    @io.coerce_to_bytes_io
    def func(fh):
        assert isinstance(fh, BytesIO)

    with BytesIO(b"abc") as f:
        func(f)


def test_invalid_coerce_to_bytes_io():
    @io.coerce_to_bytes_io
    def func(fh):
        raise RuntimeError("YOU SHOULDNT BE HERE")

    with pytest.raises(ValueError):
        func(123)
