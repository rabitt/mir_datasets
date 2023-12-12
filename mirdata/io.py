import functools
import io
from typing import BinaryIO, Callable, List, Optional, TextIO, TypeVar, Union

import numpy as np
import pretty_midi
from smart_open import open

from mirdata import annotations

T = TypeVar("T")  # Can be anything


def coerce_to_string_io(
    func: Callable[[TextIO], T]
) -> Callable[[Optional[Union[str, TextIO]]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(file_path_or_obj: Optional[Union[str, TextIO]]) -> Optional[T]:
        if not file_path_or_obj:
            return None
        if isinstance(file_path_or_obj, str):
            with open(file_path_or_obj, encoding="utf-8") as f:
                return func(f)
        elif isinstance(file_path_or_obj, io.StringIO):
            return func(file_path_or_obj)
        else:
            raise ValueError(
                "Invalid argument passed to {}, argument has the type {}",
                func.__name__,
                type(file_path_or_obj),
            )

    return wrapper


def coerce_to_bytes_io(
    func: Callable[[BinaryIO], T]
) -> Callable[[Optional[Union[str, BinaryIO]]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(file_path_or_obj: Optional[Union[str, BinaryIO]]) -> Optional[T]:
        if not file_path_or_obj:
            return None
        if isinstance(file_path_or_obj, str):
            with open(file_path_or_obj, "rb") as f:
                return func(f)
        elif isinstance(file_path_or_obj, io.BytesIO):
            return func(file_path_or_obj)
        else:
            raise ValueError(
                "Invalid argument passed to {}, argument has the type {}",
                func.__name__,
                type(file_path_or_obj),
            )

    return wrapper


@coerce_to_bytes_io
def load_midi(fhandle: BinaryIO) -> pretty_midi.PrettyMIDI:
    """Load a midi file.

    Args:
        fhandle (str or file-like): File-like object or path to midi file

    Returns:
        pretty_midi.PrettyMIDI: pretty_midi object
    """
    return pretty_midi.PrettyMIDI(fhandle)


def load_notes_from_midi(
    midi_path: Optional[Union[str, BinaryIO]] = None,
    midi: Optional[pretty_midi.PrettyMIDI] = None,
    skip_drums: bool = True,
) -> Optional[annotations.NoteData]:
    """Load note data from a midi file.

    Args:
        midi_path (str or None): path to midi file or None
        midi (pretty_midi.PrettyMIDI or None): pre-loaded midi object or None
            if None, the midi object is loaded using midi_path
        skip_drums (bool): if True, skips notes from intruments which are drums.

    Returns:
        NoteData: note annotations
    """
    if not midi and not midi_path:
        raise ValueError("At least one of midi_path or midi must be provided")
    elif not midi:
        midi = load_midi(midi_path)

    intervals = []
    pitches = []
    confidence = []
    for instrument in midi.instruments:  # type: ignore
        if instrument.is_drum and skip_drums:
            continue

        # remove notes which have start_time >= end_time
        instrument.remove_invalid_notes()
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)
            confidence.append(note.velocity)

    # if there are no notes, return None
    if len(intervals) == 0:
        return None

    return annotations.NoteData(
        np.array(intervals),
        "s",
        np.array(pitches, dtype=float),
        "midi",
        np.array(confidence, dtype=float),
        "velocity",
    )


def load_multif0_from_midi(
    midi_path: Optional[Union[str, BinaryIO]] = None,
    midi: Optional[pretty_midi.PrettyMIDI] = None,
    skip_drums: bool = True,
    pitch_bend: bool = False,
) -> Optional[annotations.MultiF0Data]:
    """Load multif0 data from a midi file, optionally considering pitch bend
    information.

    Args:
        midi_path (str or None): path to midi file or None
        midi (pretty_midi.PrettyMIDI or None): pre-loaded midi object or None
            if None, the midi object is loaded using midi_path
        skip_drums (bool): if True, skips notes from intruments which are drums.
        pitch_bend (bool): if True, adjusts pitch values containing pitch bend information.

    Returns:
        MultiF0Data: multif0 annotation
    """

    def _to_idx(time_in_sec, hop):
        return int(np.round(time_in_sec / hop))

    if not midi and not midi_path:
        raise ValueError("At least one of midi_path or midi must be provided")
    elif not midi:
        midi = load_midi(midi_path)

    times_raw = midi._PrettyMIDI__tick_to_time  # type: ignore
    time_hop = np.min(np.diff(times_raw))
    times = np.arange(0, np.max(times_raw) + time_hop, time_hop)
    freqs_list: List[list] = [[] for _ in times]
    confidence: List[list] = [[] for _ in times]
    has_data = False
    for instrument in midi.instruments:  # type: ignore
        if instrument.is_drum and skip_drums:
            continue

        # remove notes which have start_time >= end_time
        instrument.remove_invalid_notes()

        time_idx: List[int] = []
        pitch_val: List[float] = []
        conf_val: List[float] = []
        for note in instrument.notes:
            has_data = True
            # index into times
            this_idx = range(_to_idx(note.start, time_hop), _to_idx(note.end, time_hop) + 1)
            time_idx.extend(this_idx)
            pitch_val.extend([float(note.pitch) for _ in this_idx])
            conf_val.extend([float(note.velocity) for _ in this_idx])

        # look up any pitch bend information
        do_pb = pitch_bend and len(instrument.pitch_bends) > 0
        pb_idx = [_to_idx(p.time, time_hop) for p in instrument.pitch_bends] if do_pb else []
        pb_shifts = (
            [pretty_midi.utilities.pitch_bend_to_semitones(p.pitch) for p in instrument.pitch_bends]
            if do_pb
            else []
        )

        # add notes with optional pitch bend to the array
        for t_idx, pv, cv in zip(time_idx, pitch_val, conf_val):
            if t_idx in pb_idx:
                pv += pb_shifts[pb_idx.index(t_idx)]
            freqs_list[t_idx].append(pv)
            confidence[t_idx].append(cv)

    if not has_data:
        return None

    return annotations.MultiF0Data(times, "s", freqs_list, "midi", confidence, "velocity")
