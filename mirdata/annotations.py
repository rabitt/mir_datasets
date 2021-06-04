"""mirdata annotation data types
"""
import logging
import re
from typing import List, Optional, Tuple

from jams.schema import namespace
import librosa
import numpy as np
import scipy

#: Beat position units
BEAT_POSITION_UNITS = {
    "bar_index": "beat index within a bar, 1-indexed",
    "global_index": "beat index within full track, 1-indexed",
    "bar_fraction": "beat position as fractions of bars, e.g. 0.25",
    "global_fraction": "bar_frac, but where the integer part indicates the bar. e.g. 4.25",
}

#: Chord units
CHORD_UNITS = {
    "harte": "chords in harte format, e.g. Ab:maj7",
    "jams": "chords in jams 'chord' format",
    "open": "no strict schema or units",
}

#: Amplitude/voicing units
AMPLITUDE_UNITS = {
    "likelihood": "score between 0 and 1",
    "velocity": "MIDI velocity between 0 and 127",
    "binary": "0 or 1",
    "energy": "energy value, measured as the sum of a squared signal",
}

#: Event units
EVENT_UNITS = {"open": "no scrict schema or units"}

#: Key units
KEY_UNITS = {"key_mode": "key labels in key-mode format, e.g. G#:minor"}

#: Lyric units
LYRIC_UNITS = {
    "words": "lyrics as words or phrases",
    "syllable_open": "lyrics segmented by syllable, no strict schema",
    "pronunciations_open": "lyric pronunciations, no strict schema",
}

#: Pitch units
PITCH_UNITS = {
    "hz": "hertz",
    "midi": "MIDI note number",
    "pc": "pitch class, e.g. G#",
    "note_name": "pc with octave, e.g. Ab4",
}

#: Section units
SECTION_UNITS = {"open": "no scrict schema or units"}

#: Tempo units
TEMPO_UNITS = {"bpm": "beats per minute"}

#: Time units
TIME_UNITS = {
    "s": "seconds",
    "ms": "miliseconds",
    "ticks": "MIDI ticks",
}

#: Voicing units
VOICING_UNITS = {k: AMPLITUDE_UNITS[k] for k in ["binary", "likelihood"]}


class Annotation(object):
    """Annotation base class"""

    def __repr__(self):
        attributes = [v for v in dir(self) if not v.startswith("_")]
        repr_str = f"{self.__class__.__name__}({', '.join(attributes)})"
        return repr_str


class MultiAnnotator(object):
    """Multiple annotator class.
    This class should be used for datasets with multiple annotators (e.g. multiple annotators per track).

    Attributes:
        annotators (list): list with annotator ids
        annotations (list): list of annotations (e.g. [beat_data1, beat_data2] each with
                             type BeatData or  [chord_data1, chord_data2] each with type chord data

    """

    def __init__(self, annotators, annotations, dtype) -> None:
        validate_array_like(annotators, list, str, none_allowed=True)
        validate_array_like(annotations, list, dtype, none_allowed=True)
        validate_lengths_equal([annotators, annotations])

        self.annotators = annotators
        self.annotations = annotations


class BeatData(Annotation):
    """BeatData class

    Attributes:
        times (np.ndarray): array of time stamps with positive,
            strictly increasing values
        time_unit (str): time unit, one of TIME_UNITS
        positions (np.ndarray): array of beat positions in the format
            of position_unit. For all units, values of 0 indicate beats which
            fall outside of a measure.
        position_unit (str): beat position unit, one of BEAT_POSITION_UNITS
        confidence (np.ndarray): array of confidence values
        confidence_unit (str): confidence unit, one of AMPLITUDE_UNITS

    """

    def __init__(
        self,
        times,
        time_unit,
        positions,
        position_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(times, np.ndarray, float)
        validate_lengths_equal([times, positions])
        validate_times(times, time_unit)
        validate_beat_positions(positions, position_unit)
        validate_confidence(confidence, confidence_unit)

        self.times = times
        self.time_unit = time_unit
        self.positions = positions
        self.position_unit = position_unit
        self.confidence = confidence
        self.confidence_unit = confidence_unit


class SectionData(Annotation):
    """SectionData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        labels (list or None): list of section labels
        label_unit (str or None): label unit, one of SECTION_UNITS

    """

    def __init__(self, intervals, interval_unit, labels=None, label_unit=None):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str, none_allowed=True)
        validate_lengths_equal([intervals, labels])
        validate_intervals(intervals, interval_unit)
        validate_unit(label_unit, SECTION_UNITS, allow_none=True)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.labels = labels
        self.label_unit = label_unit


class ChordData(Annotation):
    """ChordData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        labels (list): list chord labels (as strings)
        label_unit (str): chord label schema
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of AMPLITUDE_UNITS
    """

    def __init__(
        self,
        intervals,
        interval_unit,
        labels,
        label_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, labels, confidence])
        validate_intervals(intervals, interval_unit)
        validate_unit(label_unit, CHORD_UNITS)
        validate_chord_labels(labels, label_unit)
        validate_confidence(confidence, confidence_unit)

        self.intervals = intervals
        self.labels = labels
        self.confidence = confidence


class F0Data(Annotation):
    """F0Data class

    Attributes:
        times (np.ndarray): array of time stamps (as floats)
            with positive, strictly increasing values
        time_unit (str): time unit, one of TIME_UNITS
        frequencies (np.ndarray): array of frequency values (as floats)
        frequency_unit (str): frequency unit, one of PITCH_UNITS
        voicing (np.ndarray): array of voicing values, indicating whether or
            not a time frame has an active pitch
        voicing_unit (str): voicing unit, one of VOICING_UNITS
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of AMPLITUDE_UNITS

    """

    def __init__(
        self,
        times,
        time_unit,
        frequencies,
        frequency_unit,
        voicing,
        voicing_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(times, np.ndarray, float)
        if frequency_unit in ["note_name", "pc"]:
            validate_array_like(frequencies, np.ndarray, None)
        else:
            validate_array_like(frequencies, np.ndarray, float)
        validate_array_like(voicing, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([times, frequencies, voicing, confidence])
        validate_times(times, time_unit)
        validate_uniform_times(times)
        validate_pitches(frequencies, frequency_unit)
        validate_voicing(voicing, voicing_unit)
        validate_confidence(confidence, confidence_unit)
        if any(voicing[frequencies == 0] != 0):
            raise ValueError("Found frequencies with value 0, but a nonzero voicing.")

        self.times = times
        self.time_unit = time_unit
        self.frequencies = frequencies
        self.frequency_unit = frequency_unit
        self.voicing = voicing
        self.voicing_unit = voicing_unit
        self._confidence = confidence
        self.confidence_unit = confidence_unit

    @property
    def confidence(self):
        logging.warning(
            "Warning: the API for annotations.F0Data.confidence has changed. "
            + "For most datasets, confidence will now be None, and "
            + "F0Data.voicing should be used instead."
        )
        return self._confidence

    def resample(self, times_new, times_new_unit):
        """Resample the annotation to a new time scale. This function is adapted from:
        https://github.com/craffel/mir_eval/blob/master/mir_eval/melody.py#L212

        Args:
            times_new (np.ndarray): new time base, in units of times_new_unit
            times_new_unit (str): time unit, one of TIME_UNITS

        Returns:
            F0Data: F0 data sampled at new time scale

        """
        times = convert_time_units(self.times, self.time_unit, times_new_unit)
        if self.frequency_unit not in ["hz", "midi"]:
            raise NotImplementedError(
                "resampling is not supported for {}".format(self.frequency_unit)
            )
        frequencies = self.frequencies
        voicing = self.voicing
        confidence = self._confidence

        # We need to fix zero transitions
        # Fill in zero values with the last reported frequency
        # to avoid erroneous values when resampling
        frequencies_held = np.array(frequencies)
        for n, frequency in enumerate(frequencies[1:]):
            if frequency == 0:
                frequencies_held[n + 1] = frequencies_held[n]
        # Linearly interpolate frequencies
        frequencies_resampled = scipy.interpolate.interp1d(
            times, frequencies_held, "linear", bounds_error=False, fill_value=0.0
        )(times_new)
        # Retain zeros
        frequency_mask = scipy.interpolate.interp1d(
            times, frequencies, "zero", bounds_error=False, fill_value=0
        )(times_new)
        frequencies_resampled *= frequency_mask != 0

        # Use nearest-neighbor for voicing if it was used for frequencies
        # if voicing is not binary, use linear interpolation
        if self.voicing_unit != "binary":
            voicing_resampled = scipy.interpolate.interp1d(
                times, voicing, "linear", bounds_error=False, fill_value=0
            )(times_new)
        else:
            voicing_resampled = scipy.interpolate.interp1d(
                times, voicing, "nearest", bounds_error=False, fill_value=0
            )(times_new)

        voicing_resampled[frequencies_resampled == 0] = 0

        if confidence is None:
            confidence_resampled = None
        # binary confidence
        elif self.confidence_unit == "binary":
            confidence_resampled = scipy.interpolate.interp1d(
                times, confidence, "nearest", bounds_error=False, fill_value=0
            )(times_new)
        # nonbinary confidence
        else:
            confidence_resampled = scipy.interpolate.interp1d(
                times, confidence, "linear", bounds_error=False, fill_value=0
            )(times_new)

        return F0Data(
            times_new,
            times_new_unit,
            frequencies_resampled,
            self.frequency_unit,
            voicing_resampled,
            self.voicing_unit,
            confidence_resampled,
            self.confidence_unit,
        )

    def to_sparse_index(
        self,
        time_scale,
        time_scale_unit,
        frequency_scale,
        frequency_scale_unit,
        amplitude_unit="binary",
    ):
        """
        Convert F0 annotation to sparse matrix indices for a time-frequency matrix.

        Args:
            time_scale (np.array): times in units time_unit
            time_scale_unit (str): time scale units, one of TIME_UNITS
            frequency_scale (np.array): frequencies in frequency_unit
            frequency_scale_unit (str): frequency scale units, one of PITCH_UNITS
            amplitude_unit (str): amplitude units, one of AMPLITUDE_UNITS
                Defaults to "binary".

        Returns:
            * sparse_index (np.ndarray): Array of sparce indices [(time_index, frequency_index)]
            * amplitude (np.ndarray): Array of amplitude values for each index

        """
        f0dat = self.resample(time_scale, time_scale_unit)
        frequencies = convert_pitch_units(
            f0dat.frequencies, self.frequency_unit, frequency_scale_unit
        )

        # get indexes in matrix
        nonzero_freqs = frequencies > 0  # find indexes for frequencies not equal to 0
        frequencies[frequencies == 0] = 1  # change zero frequency value to avoid NaN
        time_indexes = np.arange(len(time_scale))
        freq_indexes = closest_index(
            np.log(frequencies)[:, np.newaxis],
            np.log(frequency_scale)[:, np.newaxis],
        )

        # create sparse index
        index = [
            (t, f)
            for t, f in zip(time_indexes[nonzero_freqs], freq_indexes[nonzero_freqs])
            if t != -1 and f != -1
        ]
        voicing = np.array(
            [
                v
                for (v, t, f) in zip(
                    f0dat.voicing[nonzero_freqs],
                    time_indexes[nonzero_freqs],
                    freq_indexes[nonzero_freqs],
                )
                if t != -1 and f != -1
            ]
        )

        return (
            np.array(index),
            convert_amplitude_units(voicing, self.voicing_unit, amplitude_unit),
        )

    def to_matrix(
        self,
        time_scale,
        time_scale_unit,
        frequency_scale,
        frequency_scale_unit,
        amplitude_unit="binary",
    ):
        """Convert f0 data to a matrix (piano roll) defined by a time and frequency scale

        Args:
            time_scale (np.array): times in units time_unit
            time_scale_unit (str): time scale units, one of TIME_UNITS
            frequency_scale (np.array): frequencies in frequency_unit
            frequency_scale_unit (str): frequency scale units, one of PITCH_UNITS
            amplitude_unit (str): amplitude units, one of AMPLITUDE_UNITS
                Defaults to "binary".

        Returns:
            np.ndarray: 2D matrix of shape len(time_scale) x len(frequency_scale)
        """
        index, voicing = self.to_sparse_index(
            time_scale,
            time_scale_unit,
            frequency_scale,
            frequency_scale_unit,
            amplitude_unit,
        )
        matrix = np.zeros((len(time_scale), len(frequency_scale)))
        matrix[index[:, 0], index[:, 1]] = voicing
        return matrix

    def to_multif0(self):
        """Convert annotation to multif0 format

        Returns:
            MultiF0Data: data in multif0 format

        """
        frequency_list = [[f] if f > 0 else [] for f in self.frequencies]
        confidence_list = (
            None
            if self._confidence is None
            else [
                [c] if f > 0 else [] for c, f in zip(self._confidence, self.frequencies)
            ]
        )
        return MultiF0Data(
            self.times,
            self.time_unit,
            frequency_list,
            self.frequency_unit,
            confidence_list,
            self.confidence_unit,
        )

    def to_mir_eval(self):
        """Convert units and format to what is expected by mir_eval.melody.evaluate

        Returns:
            * times (np.ndarray) - uniformly spaced times in seconds
            * frequencies (np.ndarray) - frequency values in hz
            * voicing (np.ndarray) - voicings, as likelihood values
        """
        times = convert_time_units(self.times, self.time_unit, "s")
        frequencies = convert_pitch_units(self.frequencies, self.frequency_unit, "hz")
        voicing = convert_amplitude_units(self.voicing, self.voicing_unit, "likelihood")
        return times, frequencies, voicing


class MultiF0Data(Annotation):
    """MultiF0Data class

    Attributes:
        times (np.ndarray): array of time stamps (as floats)
            with positive, strictly increasing values
        time_unit (str): time unit, one of TIME_UNITS
        frequency_list (list): list of lists of frequency values (as floats)
        frequency_unit (str): frequency unit, one of PITCH_UNITS
        confidence_list (np.ndarray or None): list of lists of confidence values
        confidence_unit (str or None): confidence unit, one of AMPLITUDE_UNITS

    """

    def __init__(
        self,
        times,
        time_unit,
        frequency_list,
        frequency_unit,
        confidence_list=None,
        confidence_unit=None,
    ):
        validate_array_like(times, np.ndarray, float)
        validate_array_like(frequency_list, list, list)
        validate_array_like(confidence_list, list, list, none_allowed=True)
        validate_lengths_equal([times, frequency_list, confidence_list])
        validate_times(times, time_unit)
        validate_uniform_times(times)
        validate_pitches(frequency_list, frequency_unit)
        validate_confidence(confidence_list, confidence_unit)

        self.times = times
        self.time_unit = time_unit
        self.frequency_list = frequency_list
        self.frequency_unit = frequency_unit
        self.confidence_list = confidence_list
        self.confidence_unit = confidence_unit

        self._remove_duplicates()

    def _remove_duplicates(self):
        new_frequency_list = []
        new_confidence_list = []
        confidence_list = (
            [[0 for _ in flist] for flist in self.frequency_list]
            if self.confidence_list is None
            else self.confidence_list
        )
        for flist, clist in zip(self.frequency_list, confidence_list):
            tmp_flist = []
            tmp_clist = []
            for f, c in zip(flist, clist):
                if f in tmp_flist:
                    continue
                tmp_flist.append(f)
                tmp_clist.append(c)

            new_frequency_list.append(tmp_flist)
            new_confidence_list.append(tmp_clist)

        self.frequency_list = new_frequency_list
        self.confidence_list = (
            None if self.confidence_list is None else new_confidence_list
        )

    def __add__(self, other):
        if other is None:
            return self

        if isinstance(other, F0Data):
            other = other.to_multif0()

        if not isinstance(other, MultiF0Data):
            raise TypeError("Unable to add type {} to MultiF0 data".format(type(other)))

        other_times = convert_time_units(other.times, other.time_unit, self.time_unit)
        if np.max(other_times) > np.max(self.times):
            data_resamp = self.resample(other_times, self.time_unit)
            times = other_times
            this_data = data_resamp
            other_data = other
        else:
            other_resamp = other.resample(self.times, self.time_unit)
            times = self.times
            this_data = self
            other_data = other_resamp

        this_frequency_list = [[f for f in flist] for flist in this_data.frequency_list]
        other_frequency_list = convert_pitch_units(
            other_data.frequency_list, other.frequency_unit, self.frequency_unit
        )

        for i, flist in enumerate(other_frequency_list):
            this_frequency_list[i].extend(flist)

        this_has_confidence = this_data.confidence_list is not None
        other_has_confidence = other_data.confidence_unit is not None
        this_confidence_unit = this_data.confidence_unit
        if this_has_confidence and other_has_confidence:
            this_confidence_list = [
                [c for c in clist] for clist in this_data.confidence_list
            ]
            other_confidence_list = convert_amplitude_units(
                other_data.confidence_list,
                other.confidence_unit,
                self.confidence_unit,
            )
            for i, clist in enumerate(other_confidence_list):
                this_confidence_list[i].extend(clist)
        elif not this_has_confidence and not other_has_confidence:
            this_confidence_list = None
        else:
            logging.warning(
                "Adding two MultiF0Data where one has confidence=None "
                + "and the other does not. The sum will have confidence=None."
            )
            this_confidence_list = None
            this_confidence_unit = None

        return MultiF0Data(
            times,
            self.time_unit,
            this_frequency_list,
            self.frequency_unit,
            this_confidence_list,
            this_confidence_unit,
        )

    def resample(self, times_new, times_new_unit):
        """Resample annotation to a new time scale. This function is adapted from:
        https://github.com/craffel/mir_eval/blob/master/mir_eval/multipitch.py#L104

        Args:
            times_new (np.array): array of new time scale values
            times_new_unit (str): units for new time scale, one of TIME_UNITS

        Returns:
            MultiF0Data: the resampled annotation
        """
        times = convert_time_units(self.times, self.time_unit, times_new_unit)
        n_times = len(self.times)

        # scipy's interpolate doesn't handle ragged arrays. Instead, we interpolate
        # the frequency index and then map back to the frequency values.
        # This only works because we're using a nearest neighbor interpolator!
        frequency_index = np.arange(0, n_times)

        # times are already ordered so assume_sorted=True for efficiency
        # since we're interpolating the index, fill_value is set to the first index
        # that is out of range. We handle this in the next line.
        new_frequency_index = scipy.interpolate.interp1d(
            times,
            frequency_index,
            kind="nearest",
            bounds_error=False,
            assume_sorted=True,
            fill_value=n_times,
        )(times_new)

        # create array of frequencies plus additional empty element at the end for
        # target time stamps that are out of the interpolation range
        freq_vals = self.frequency_list + [[]]

        # map interpolated indices back to frequency values
        frequencies_resampled = [freq_vals[i] for i in new_frequency_index.astype(int)]

        if self.confidence_list is not None:
            confidence_vals = self.confidence_list + [[]]
            confidence_resampled = [
                confidence_vals[i] for i in new_frequency_index.astype(int)
            ]
        else:
            confidence_resampled = None

        return MultiF0Data(
            times_new,
            times_new_unit,
            frequencies_resampled,
            self.frequency_unit,
            confidence_resampled,
            self.confidence_unit,
        )

    def to_sparse_index(
        self,
        time_scale,
        time_scale_unit,
        frequency_scale,
        frequency_scale_unit,
        amplitude_unit="binary",
    ):
        """
        Convert MultiF0 annotation to sparse matrix indices for a time-frequency matrix.

        Args:
            time_scale (np.array): times in units time_unit
            time_scale_unit (str): time scale units, one of TIME_UNITS
            frequency_scale (np.array): frequencies in frequency_unit
            frequency_scale_unit (str): frequency scale units, one of PITCH_UNITS
            amplitude_unit (str): amplitude units, one of AMPLITUDE_UNITS
                Defaults to "binary".

        Returns:
            * sparse_index (np.ndarray): Array of sparce indices [(time_index, frequency_index)]
            * amplitude (np.ndarray): Array of amplitude values for each index

        """
        multif0dat = self.resample(time_scale, time_scale_unit)
        frequencies = convert_pitch_units(
            multif0dat.frequency_list, self.frequency_unit, frequency_scale_unit
        )
        time_indexes = np.arange(len(time_scale))

        time_indexes_flattened = np.array(
            [t for (t, f_list) in zip(time_indexes, frequencies) for f in f_list]
        )
        frequencies_flattened = np.array([f for f_list in frequencies for f in f_list])
        if multif0dat.confidence_list is None:
            confidence_flattened = np.ones((len(time_indexes_flattened),))
            conf_unit = "binary"
        else:
            confidence_flattened = np.array(
                [c for c_list in multif0dat.confidence_list for c in c_list]
            )
            conf_unit = self.confidence_unit

        # get frequency indexes in matrix
        nonzero_freqs = (
            frequencies_flattened > 0
        )  # find indexes for frequencies not equal to 0
        frequencies_flattened[
            frequencies_flattened == 0
        ] = 1  # change zero frequency value to avoid NaN
        freq_indexes = closest_index(
            np.log(frequencies_flattened)[:, np.newaxis],
            np.log(frequency_scale)[:, np.newaxis],
        )

        # create sparse index
        index = [
            (t, f)
            for t, f in zip(
                time_indexes_flattened[nonzero_freqs], freq_indexes[nonzero_freqs]
            )
            if t != -1 and f != -1
        ]
        confidence_out = np.array(
            [
                c
                for c, t, f in zip(
                    confidence_flattened[nonzero_freqs],
                    time_indexes_flattened[nonzero_freqs],
                    freq_indexes[nonzero_freqs],
                )
                if t != -1 and f != -1
            ]
        )
        return (
            np.array(index),
            convert_amplitude_units(confidence_out, conf_unit, amplitude_unit),
        )

    def to_matrix(
        self,
        time_scale,
        time_scale_unit,
        frequency_scale,
        frequency_scale_unit,
        amplitude_unit="binary",
    ):
        """Convert f0 data to a matrix (piano roll) defined by a time and frequency scale

        Args:
            time_scale (np.array): times in units time_unit
            time_scale_unit (str): time scale units, one of TIME_UNITS
            frequency_scale (np.array): frequencies in frequency_unit
            frequency_scale_unit (str): frequency scale units, one of PITCH_UNITS
            amplitude_unit (str): amplitude units, one of AMPLITUDE_UNITS
                Defaults to "binary".

        Returns:
            np.ndarray: 2D matrix of shape len(time_scale) x len(frequency_scale)
        """
        index, voicing = self.to_sparse_index(
            time_scale,
            time_scale_unit,
            frequency_scale,
            frequency_scale_unit,
            amplitude_unit,
        )
        matrix = np.zeros((len(time_scale), len(frequency_scale)))
        matrix[index[:, 0], index[:, 1]] = voicing
        return matrix

    def to_mir_eval(self):
        """Convert annotation into the format expected by mir_eval.multipitch.evaluate

        Returns:
            * times (np.ndarray): array of uniformly spaced time stamps in seconds
            * frequency_list (list): list of np.array of frequency values in Hz
        """
        times = convert_time_units(self.times, self.time_unit, "s")
        frequency_list = [
            convert_pitch_units(np.array(flist), self.frequency_unit, "hz")
            for flist in self.frequency_list
        ]
        return times, frequency_list


class NoteData(Annotation):
    """NoteData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        pitches (np.ndarray): array of pitches
        pitch_unit (str): note unit, one of PITCH_UNITS
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of AMPLITUDE_UNITS

    """

    def __init__(
        self,
        intervals: np.ndarray,
        interval_unit: str,
        pitches: np.ndarray,
        pitch_unit: str,
        confidence: Optional[np.ndarray] = None,
        confidence_unit: Optional[str] = None,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(pitches, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, pitches, confidence])
        validate_intervals(intervals, interval_unit)
        validate_pitches(pitches, pitch_unit)
        validate_confidence(confidence, confidence_unit)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.pitches = pitches
        self.pitch_unit = pitch_unit
        self.confidence = confidence
        self.confidence_unit = confidence_unit

        self._remove_duplicates()

    @property
    def notes(self) -> np.ndarray:
        logging.warning(
            "Deprecation warning: NoteData.notes will be removed in a future version."
            + "Use NoteData.pitches"
        )
        return self.pitches

    def _remove_duplicates(self):
        # deduplicate if matching interval and pitch
        unq, unq_idx = np.unique(
            np.hstack([self.intervals, self.pitches[:, np.newaxis]]),
            axis=0,
            return_index=True,
        )
        self.intervals = unq[:, :2]
        self.pitches = unq[:, 2]
        if self.confidence is not None:
            self.confidence = self.confidence[unq_idx]

    def __add__(self, other):

        if other is None:
            return self

        if not isinstance(other, NoteData):
            raise TypeError("Unable to add type {} to NoteData".format(type(other)))
        # convert to the current units
        intervals = convert_time_units(
            other.intervals, other.interval_unit, self.interval_unit
        )
        pitches = convert_pitch_units(other.pitches, other.pitch_unit, self.pitch_unit)

        if other.confidence is None and self.confidence is None:
            new_confidence = None
            new_confidence_unit = None
        elif other.confidence is not None and self.confidence is not None:
            new_confidence = np.concatenate(
                [
                    self.confidence,
                    convert_amplitude_units(
                        other.confidence,
                        other.confidence_unit,
                        self.confidence_unit,
                    ),
                ]
            )
            new_confidence_unit = self.confidence_unit
        else:
            logging.warning(
                "Adding two NoteData objects but one has confidence=None and "
                + "the other does not. The resulting confidence will be None"
            )
            new_confidence = None
            new_confidence_unit = None

        return NoteData(
            np.vstack([self.intervals, intervals]),
            self.interval_unit,
            np.concatenate([self.pitches, pitches]),
            self.pitch_unit,
            new_confidence,
            new_confidence_unit,
        )

    def to_sparse_index(
        self,
        time_scale: np.ndarray,
        time_scale_unit: str,
        frequency_scale: np.ndarray,
        frequency_scale_unit: str,
        amplitude_unit: str = "binary",
        onsets_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert note annotations to indexes of a sparse matrix (piano roll)

        Args:
            time_scale (np.array): array of matrix time stamps in seconds
            time_scale_unit (str): units for time scale values, one of TIME_UNITS
            frequency_scale (np.array): array of matrix frequency values in seconds
            frequency_scale_unit (str): units for frequency scale values, one of PITCH_UNITS
            amplitude_unit (str): units for amplitude values, one of AMPLITUDE_UNITS.
                Defaults to "binary".
            onsets_only (bool, optional): If True, returns an onset piano roll.
                Defaults to False.

        Returns:
            * sparse_index (np.ndarray): Array of sparce indices [(time_index, frequency_index)]
            * amplitude (np.ndarray): Array of amplitude values for each index

        """
        intervals = convert_time_units(
            self.intervals, self.interval_unit, time_scale_unit
        )
        freqs_hz = convert_pitch_units(
            self.pitches, self.pitch_unit, frequency_scale_unit
        )

        if self.confidence is not None:
            confidence = convert_amplitude_units(
                self.confidence, self.confidence_unit, amplitude_unit
            )
        else:
            confidence = convert_amplitude_units(
                np.ones((freqs_hz.shape)), "binary", amplitude_unit
            )

        time_index_0 = closest_index(
            intervals[:, 0, np.newaxis], time_scale[:, np.newaxis]
        )
        freq_indexes = closest_index(
            np.log(freqs_hz)[:, np.newaxis], np.log(frequency_scale)[:, np.newaxis]
        )
        if onsets_only:
            onset_index = []
            confidences = []
            for t0, f, c in zip(time_index_0, freq_indexes, confidence):
                if t0 == -1 or f == -1:
                    continue
                onset_index.append([t0, f])
                confidences.append(c)
            return np.array(onset_index), np.array(confidences)

        time_index_1 = closest_index(
            intervals[:, 1, np.newaxis], time_scale[:, np.newaxis]
        )
        max_idx = len(time_scale) - 1
        sparse_index = []
        confidences = []
        for t0, t1, f, c in zip(time_index_0, time_index_1, freq_indexes, confidence):
            if f == -1 or (t0 == -1 and t1 == -1):
                continue

            t_start = max([t0, 0])
            t_end = (t1 if t1 != -1 else max_idx) + 1

            sparse_index.extend([[t, f] for t in range(t_start, t_end)])
            confidences.extend([c for _ in range(t_start, t_end)])

        return np.array(sparse_index), np.array(confidences)

    def to_matrix(
        self,
        time_scale: np.ndarray,
        time_scale_unit: str,
        frequency_scale: np.ndarray,
        frequency_scale_unit: str,
        amplitude_unit: str = "binary",
        onsets_only: bool = False,
    ) -> np.ndarray:
        """Convert f0 data to a matrix (piano roll) defined by a time and frequency scale

        Args:
            time_scale (np.ndarray): array of matrix time stamps in seconds
            time_scale_unit (str): units for time scale values, one of TIME_UNITS
            frequency_scale (np.ndarray): array of matrix frequency values in seconds
            frequency_scale_unit (str): units for frequency scale values, one of PITCH_UNITS
            onsets_only (bool, optional): If True, returns an onset piano roll.
                Defaults to False.

        Returns:
            np.ndarray: 2D matrix of shape len(time_scale) x len(frequency_scale)
        """
        index, voicing = self.to_sparse_index(
            time_scale,
            time_scale_unit,
            frequency_scale,
            frequency_scale_unit,
            amplitude_unit,
            onsets_only,
        )
        matrix = np.zeros((len(time_scale), len(frequency_scale)))
        matrix[index[:, 0], index[:, 1]] = voicing
        return matrix

    def to_multif0(
        self, time_hop: float, time_hop_unit: str, max_time: Optional[float] = None
    ) -> MultiF0Data:
        """Convert note annotation to multiple f0 format.

        Args:
            time_hop (float): time between time stamps in multif0 annotation
            time_hop_unit (str): unit for time_hop, and resulting multif0 data.
                One of TIME_UNITS
            max_time (float, optional): Maximum time stamp in time_hop units.
                Defaults to None, in which case the maximum note interval
                time is used.

        Returns:
            MultiF0Data: multif0 annotation
        """
        intervals = convert_time_units(
            self.intervals, self.interval_unit, time_hop_unit
        )
        note_time_max = np.max(intervals[:, 1])
        max_time = note_time_max if not max_time else max_time
        if max_time < note_time_max:
            raise ValueError(
                "max_time = {} cannot be smaller than the last note interval = {}".format(
                    max_time, note_time_max
                )
            )
        times = np.arange(0, max_time + time_hop, time_hop)
        frequency_list: List[List[float]] = [[] for _ in times]
        confidence_list: List[List[float]] = [[] for _ in times]
        if self.confidence is not None:

            for t0, t1, pch, conf in zip(
                intervals[:, 0], intervals[:, 1], self.pitches, self.confidence
            ):
                for i in range(
                    int(np.round(t0 / time_hop)), int(np.round(t1 / time_hop)) + 1
                ):
                    frequency_list[i].append(pch)
                    confidence_list[i].append(conf)
        else:
            for t0, t1, pch in zip(intervals[:, 0], intervals[:, 1], self.pitches):
                for i in range(
                    int(np.round(t0 / time_hop)), int(np.round(t1 / time_hop)) + 1
                ):
                    frequency_list[i].append(pch)

        return MultiF0Data(
            times,
            time_hop_unit,
            frequency_list,
            self.pitch_unit,
            None if self.confidence is None else confidence_list,
            self.confidence_unit,
        )

    def to_mir_eval(self):
        """Convert data to the format expected by mir_eval.transcription.evaluate and
        mir_eval.transcription_velocity.evaluate

        Returns:
            * intervals (np.ndarray) - (n x 2) array of intervals of start time, end time in seconds
            * pitches (np.ndarray) - array of pitch values in hz
            * velocity (optional, np.ndarray) - array of velocity values between 0 and 127
        """
        intervals = convert_time_units(self.intervals, self.interval_unit, "s")
        pitches = convert_pitch_units(self.pitches, self.pitch_unit, "hz")
        velocity = (
            None
            if self.confidence is None
            else convert_amplitude_units(
                self.confidence, self.confidence_unit, "velocity"
            )
        )
        return intervals, pitches, velocity


class KeyData(Annotation):
    """KeyData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        keys (list): list key labels (as strings)
        key_unit (str): key unit, one of KEY_UNITS

    """

    def __init__(self, intervals, interval_unit, keys, key_unit):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(keys, list, str)
        validate_lengths_equal([intervals, keys])
        validate_intervals(intervals, interval_unit)
        validate_key_labels(keys, key_unit)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.keys = keys
        self.key_unit = key_unit


class LyricData(Annotation):
    """LyricData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        lyrics (list): list of lyrics (as strings)
        lyric_unit (str): lyric unit, one of LYRIC_UNITS

    """

    def __init__(
        self,
        intervals,
        interval_unit,
        lyrics,
        lyric_unit,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(lyrics, list, str)
        validate_lengths_equal([intervals, lyrics])
        validate_intervals(intervals, interval_unit)
        validate_unit(lyric_unit, LYRIC_UNITS)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.lyrics = lyrics
        self.lyric_unit = lyric_unit

    @property
    def pronunciations(self):
        logging.warning(
            "LyricData.pronunciations will be removed in a future version. "
            + "Use LyricData.lyrics"
        )
        return self.lyrics


class TempoData(Annotation):
    """TempoData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        tempos (list): array of tempo values (as floats)
        tempo_unit (str): tempo unit, one of TEMPO_UNITS
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of AMPLITUDE_UNITS

    """

    def __init__(
        self,
        intervals,
        interval_unit,
        tempos,
        tempo_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(tempos, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, tempos, confidence])
        validate_intervals(intervals, interval_unit)
        validate_tempos(tempos, tempo_unit)
        validate_confidence(confidence, confidence_unit)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.tempos = tempos
        self.tempo_unit = tempo_unit
        self.confidence = confidence
        self.confidence_unit = confidence_unit

    @property
    def value(self):
        logging.warning(
            "Deprecation warning: TempoData.value will be removed in future versions. Use TempoData.tempos instead."
        )
        return self.tempos


class EventData(Annotation):
    """EventData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        interval_unit (str): interval units, one of TIME_UNITS
        events (list): list of event labels (as strings)
        event_unit (str): event units, one of EVENT_UNITS

    """

    def __init__(self, intervals, interval_unit, events, event_unit):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(events, list, str)
        validate_lengths_equal([intervals, events])
        validate_intervals(intervals, interval_unit)
        validate_unit(event_unit, EVENT_UNITS)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.events = events
        self.event_unit = event_unit


def convert_time_units(times, time_unit, target_time_unit):
    """Convert a time array from time_unit to target_time_unit

    Args:
        times (np.ndarray): array of time values in units time_unit
        time_unit (str): time unit, one of TIME_UNITS
        target_time_unit (str): new time unit, one of TIME_UNITS

    Raises:
        ValueError: If time units are not convertable

    Returns:
        np.ndarray: times in units target_time_unit
    """
    if time_unit == "ticks" and target_time_unit == "ticks":
        return times

    def _to_seconds(times, time_unit):
        """Convert times in time_unit to seconds"""
        if time_unit == "s":
            return times
        if time_unit == "ms":
            return times / 1000.0
        raise NotImplementedError

    def _from_seconds(times_sec, target_time_unit):
        """Convert times in seconds to target_time_unit"""
        if target_time_unit == "s":
            return times_sec
        if target_time_unit == "ms":
            return times_sec * 1000.0
        raise NotImplementedError

    try:
        return _from_seconds(_to_seconds(times, time_unit), target_time_unit)
    except NotImplementedError:
        raise NotImplementedError(
            "Conversion of time in units {} to {} is not supported".format(
                time_unit, target_time_unit
            )
        )


def convert_pitch_units(pitches, pitch_unit, target_pitch_unit):
    """Convert pitch values from pitch_unit to target_pitch_unit

    Args:
        pitches (np.array): array of pitch values
        pitch_unit (str): unit of pitch, one of PITCH_UNITS
        target_pitch_unit (str): target unit of pitch, one of PITCH_UNITS

    Raises:
        NotImplementedError: If conversion between given units is not supported

    Returns:
        np.array: array of pitch values in target_pitch_unit
    """
    # if input is a nested list, call this function recursively
    if isinstance(pitches, list) and isinstance(pitches[0], list):
        return [
            []
            if len(plist) == 0
            else list(convert_pitch_units(plist, pitch_unit, target_pitch_unit))
            for plist in pitches
        ]

    if pitch_unit == "pc" and target_pitch_unit == "pc":
        return pitches

    def _to_hz(pitches, pitch_unit):
        """Convert pitches in pitch_unit to Hz"""
        if pitch_unit == "hz":
            return pitches

        if pitch_unit == "midi":
            zero_idx = pitches == 0
            pitches_hz = librosa.midi_to_hz(pitches)
            pitches_hz[zero_idx] = 0
            return pitches_hz

        if pitch_unit == "note_name":
            return librosa.note_to_hz(pitches)

        raise NotImplementedError

    def _from_hz(pitches_hz, target_pitch_unit):
        """Convert pitches int Hz to target_pitch_unit"""
        if target_pitch_unit == "hz":
            return pitches_hz

        if target_pitch_unit == "midi":
            zero_idx = pitches_hz == 0
            pitches_midi = librosa.hz_to_midi(pitches_hz)
            pitches_midi[zero_idx] = 0
            return pitches_midi

        if target_pitch_unit == "note_name":
            return librosa.hz_to_note(pitches_hz)

        raise NotImplementedError

    try:
        return _from_hz(_to_hz(pitches, pitch_unit), target_pitch_unit)
    except NotImplementedError:
        raise NotImplementedError(
            "Conversion of pitch in units {} to {} is not supported".format(
                pitch_unit, target_pitch_unit
            )
        )


def convert_amplitude_units(amplitude, amplitude_unit, target_amplitude_unit):
    """Convert amplitude values to likelihoods

    Args:
        amplitude (np.array): array of amplitude values
        amplitude_unit (str): unit of amplitude, one of AMPLITUDE_UNITS
        target_amplitude_unit (str): target unit of amplitude, one of AMPLITUDE_UNITS

    Raises:
        NotImplementedError: If conversion is not supported

    Returns:
        np.array: array of amplitude values as in target amplitude unit
    """
    # if input is a nested list, call this function recursively
    if isinstance(amplitude, list) and isinstance(amplitude[0], list):
        return [
            []
            if len(alist) == 0
            else list(
                convert_amplitude_units(
                    np.array(alist), amplitude_unit, target_amplitude_unit
                )
            )
            for alist in amplitude
        ]

    def _to_likelihood(amplitude, amplitude_unit):
        if amplitude_unit in ["likelihood", "binary"]:
            return amplitude
        if amplitude_unit == "velocity":
            return amplitude / 127.0
        raise NotImplementedError

    def _from_likelihood(amplitude, target_amplitude_unit):
        if target_amplitude_unit == "likelihood":
            return amplitude
        if target_amplitude_unit == "binary":
            return np.ceil(amplitude)
        if target_amplitude_unit == "velocity":
            return amplitude * 127.0
        raise NotImplementedError

    try:
        return _from_likelihood(
            _to_likelihood(amplitude, amplitude_unit), target_amplitude_unit
        )
    except NotImplementedError:
        raise NotImplementedError(
            "Conversion of amplitude in units {} to {} is not supported".format(
                amplitude_unit, target_amplitude_unit
            )
        )


def closest_index(input_array, target_array):
    """Get array of indices of target_array that are closest to the input_array

    Args:
        input_array (np.ndarray): (n x 2) array of input values
        target_array (np.ndarray): (m x 2) array of target values)

    Returns:
        np.ndarray: array of shape (n x 1) of indexes into target_array
    """
    indexes = np.argmin(
        scipy.spatial.distance.cdist(input_array, target_array),
        axis=1,
    )
    indexes[input_array[:, 0] > np.max(target_array[:, 0])] = -1
    indexes[input_array[:, 0] < np.min(target_array[:, 0])] = -1

    return indexes


def validate_array_like(
    array_like, expected_type, expected_dtype, check_child=False, none_allowed=False
):
    """Validate that array-like object is well formed

    If array_like is None, validation passes automatically.

    Args:
        array_like (array-like): object to validate
        expected_type (type): expected type, either list or np.ndarray
        expected_dtype (type): expected dtype
        check_child (bool): if True, checks if all elements of array are children of expected_dtype
        none_allowed (bool): if True, allows array to be None

    Raises:
        TypeError: if type/dtype does not match expected_type/expected_dtype
        ValueError: if array is empty but it shouldn't be

    """
    if array_like is None:
        if none_allowed:
            return
        else:
            raise ValueError("array_like cannot be None")

    assert expected_type in [
        list,
        np.ndarray,
    ], "expected type must be a list or np.ndarray"

    if not isinstance(array_like, expected_type):
        raise TypeError(
            f"Object should be a {expected_type}, but is a {type(array_like)}"
        )

    if expected_type == list and not all(
        isinstance(n, expected_dtype)
        for n in array_like
        if not ((n is None) and none_allowed)
    ):
        raise TypeError(f"List elements should all have type {expected_dtype}")

    if (
        expected_type == np.ndarray
        and array_like.dtype != expected_dtype
        and expected_dtype is not None
    ):
        raise TypeError(
            f"Array should have dtype {expected_dtype} but has {array_like.dtype}"
        )

    if np.asarray(array_like).size == 0:
        raise ValueError("Object should not be empty, use None instead")


def validate_lengths_equal(array_list):
    """Validate that arrays in list are equal in length

    Some arrays may be None, and the validation for these are skipped.

    Args:
        array_list (list): list of array-like objects

    Raises:
        ValueError: if arrays are not equal in length

    """
    if len(array_list) == 1:
        return

    for att1, att2 in zip(array_list[:-1], array_list[1:]):
        if att1 is None or att2 is None:
            continue

        if not len(att1) == len(att2):
            raise ValueError("Arrays have unequal length")


def validate_tempos(tempo, tempo_unit):
    """Validate if tempos are well-formed

    Args:
        tempo (list): list of tempo values
        tempo_unit (str): tempo unit, one of TEMPO_UNITS

    Raises:
        ValueError: if tempos are not well-formed
    """
    validate_unit(tempo_unit, TEMPO_UNITS)
    if (tempo < 0).any():
        raise ValueError("tempos must be positive")


def validate_beat_positions(positions, position_unit):
    """Validate if positions is well-formed.

    Args:
        positions (np.ndarray): an array of positions values
        positions_unit (str): one of BEAT_POSITION_UNITS

    Raises:
        ValueError: if positions values are incompatible with the unit

    """
    validate_unit(position_unit, BEAT_POSITION_UNITS)

    position_shape = np.shape(positions)
    if len(position_shape) != 1:
        raise ValueError(
            f"positions should be 1d, but array has shape {position_shape}"
        )

    if (positions < 0).any():
        raise ValueError("beat positions must be positive. Found values below 0.")

    if position_unit in ["bar_index", "global_index"] and not np.array_equal(
        np.floor(positions), positions
    ):
        raise ValueError(
            "measure index or global indexes should be integers. "
            + "Found fractional values."
        )

    # we expect no more than 32 beats per bar - this can be changed if a need arises!
    if position_unit == "bar_index" and np.max(positions) > 32:
        raise ValueError(
            "beats with bar_index units should have indexes "
            + "which start from 1 at the beginning of every measure. "
            + "Found values > 16."
        )

    if position_unit == "bar_fraction" and np.max(positions) > 1:
        raise ValueError(
            "beats with bar_fraction units should be between 0 and 1. "
            + "Found values above 1."
        )


def validate_confidence(confidence, confidence_unit):
    """Validate if confidence is well-formed.

    If confidence is None, validation passes automatically

    Args:
        confidence (np.ndarray): an array of confidence values
        confidence_unit (str): one of AMPLITUDE_UNITS

    Raises:
        ValueError: if confidence values are incompatible with the unit

    """
    if confidence is None:
        return

    validate_unit(confidence_unit, AMPLITUDE_UNITS)
    if isinstance(confidence[0], list):
        confidence_flat = [c for subconf in confidence for c in subconf]
    else:
        confidence_flat = confidence

    if confidence_unit == "likelihood" and (
        any([c < 0 for c in confidence_flat]) or any([c > 1 for c in confidence_flat])
    ):
        raise ValueError(
            "confidence with unit 'likelihood' should be between 0 and 1. "
            + "Found values outside [0, 1]."
        )

    if confidence_unit == "energy" and any([c < 0 for c in confidence_flat]):
        raise ValueError(
            "confidence with unit 'energy' should be nonnegative. "
            + "Found negative values."
        )

    if confidence_unit == "binary" and any([c not in [0, 1] for c in confidence_flat]):
        raise ValueError(
            "confidence with unit 'binary' should only have values of 0 or 1. "
            + "Found non-binary values."
        )

    if confidence_unit == "velocity" and (
        any([c < 0 for c in confidence_flat]) or any([c > 127 for c in confidence_flat])
    ):
        raise ValueError(
            "confidence with unit 'velocity' should be between 0 and 127. "
            + "Found values outside [0, 127]."
        )


def validate_voicing(voicing, voicing_unit):
    """Validate if voicing is well-formed.

    Args:
        voicing (np.ndarray): an array of voicing values
        voicing_unit (str): one of VOICING_UNITS

    Raises:
        ValueError: if voicing values are incompatible with the unit

    """
    validate_unit(voicing_unit, VOICING_UNITS)

    voicing_shape = np.shape(voicing)
    if len(voicing_shape) != 1:
        raise ValueError(f"voicings should be 1d, but array has shape {voicing_shape}")

    if voicing_unit == "likelihood" and (
        any([c < 0 for c in voicing]) or any([c > 1 for c in voicing])
    ):
        raise ValueError(
            "voicing with unit 'likelihood' should be between 0 and 1. "
            + "Found values outside [0, 1]."
        )

    if voicing_unit == "binary" and any([c not in [0, 1] for c in voicing]):
        raise ValueError(
            "voicing with unit 'binary' should only have values of 0 or 1. "
            + "Found non-binary values."
        )


def validate_pitches(pitches, pitch_unit):
    """Validate if pitches are well-formed.

    Args:
        pitches (np.ndarray): an array of pitch values
        pitch_unit (str): pitch unit, one of PITCH_UNITS

    Raises:
        ValueError: if pitches do not correspond to the unit

    """
    validate_unit(pitch_unit, PITCH_UNITS)
    if pitch_unit in ["hz", "midi"] and np.any(
        [np.any(np.array(p) < 0) for p in pitches]
    ):
        raise ValueError(
            "pitches should be positive numbers. "
            + "Unvoiced frames should be indicated using the confidence field, "
            + "rather than negative pitch values."
        )

    if pitch_unit == "midi" and np.any([np.any(np.array(p) > 127) for p in pitches]):
        raise ValueError("pitches in midi format cannot be larger than 127. ")

    if pitch_unit in ["pc", "note_name"]:
        try:
            librosa.note_to_midi(pitches)
        except:
            raise ValueError("invalid format for unit pc or note_name")


def validate_chord_labels(chords, chord_unit):
    """Validate that chord labels conform to chord_unit namespace

    Args:
        chords (list): list of chord labels as strings
        chord_unit (str): chord namespace, e.g. "harte"

    Raises:
        ValueError: If chords don't conform to namespace

    """
    validate_unit(chord_unit, CHORD_UNITS)

    if chord_unit in ["harte", "jams"]:
        if chord_unit == "harte":
            pattern = namespace("chord_harte")["properties"]["value"]["pattern"]
        elif chord_unit == "jams":
            pattern = namespace("chord")["properties"]["value"]["pattern"]

        matches = [re.match(pattern, c) for c in chords]
        if not all(matches):
            non_matches = [c for (c, m) in zip(chords, matches) if not m]
            raise ValueError(
                "chords {} don't conform to chord_unit {}".format(
                    non_matches, chord_unit
                )
            )


def validate_key_labels(keys, key_unit):
    """Validate that key labels conform to key_unit namespace

    Args:
        keys (list): list of key labels as strings
        key_unit (str): key namespace, e.g. "harte"

    Raises:
        ValueError: If keys don't conform to namespace

    """
    validate_unit(key_unit, KEY_UNITS)

    if key_unit == "key_mode":
        pattern = namespace("key_mode")["properties"]["value"]["pattern"]
        matches = [re.match(pattern, c) for c in keys]
        if not all(matches):
            non_matches = [k for (k, m) in zip(keys, matches) if not m]
            raise ValueError(
                "keys {} don't conform to key_unit key-mode".format(non_matches)
            )


def validate_times(times, time_unit):
    """Validate if times are well-formed.

    If times is None, validation passes automatically

    Args:
        times (np.ndarray): an array of time stamps
        time_unit (str): one of TIME_UNITS

    Raises:
        ValueError: if times have negative values or are non-increasing

    """
    if times is None:
        return

    validate_unit(time_unit, TIME_UNITS)

    time_shape = np.shape(times)
    if len(time_shape) != 1:
        raise ValueError(f"Times should be 1d, but array has shape {time_shape}")

    if (times < 0).any():
        raise ValueError("times should be positive numbers")

    if (times[1:] - times[:-1] <= 0).any():
        raise ValueError("times should be strictly increasing")


def validate_intervals(intervals, interval_unit):
    """Validate if intervals are well-formed.

    If intervals is None, validation passes automatically

    Args:
        intervals (np.ndarray): (n x 2) array
        interval_unit (str): interval unit, one of TIME_UNITS

    Raises:
        ValueError: if intervals have an invalid shape, have negative values
        or if end times are smaller than start times.

    """
    if intervals is None:
        return

    validate_unit(interval_unit, TIME_UNITS)

    # validate that intervals have the correct shape
    interval_shape = np.shape(intervals)
    if len(interval_shape) != 2 or interval_shape[1] != 2:
        raise ValueError(
            f"Intervals should be arrays with two columns, but array has {interval_shape}"
        )

    # validate that time stamps are all positive numbers
    if (intervals < 0).any():
        raise ValueError(f"Interval values should be nonnegative numbers")

    # validate that end times are bigger than start times
    elif (intervals[:, 1] - intervals[:, 0] < 0).any():
        raise ValueError(f"Interval start times must be smaller than end times")


def validate_unit(unit, unit_values, allow_none=False):
    """Validate that the given unit is one of the allowed unit values.

    Args:
        unit (str): the unit name
        unit_values (dict): dictionary of possible unit values
        allow_none (bool): if true, allows unit=None to pass validation

    Raises:
        ValueError: If the given unit is not one of the allowed unit valuess
    """
    if allow_none and not unit:
        return

    if unit not in unit_values:
        raise ValueError("unit={} is not one of {}".format(unit, unit_values))


def validate_uniform_times(times):
    time_diffs = np.diff(times)
    median_diff = np.median(time_diffs)
    if any(np.abs(time_diffs - median_diff) > 0.01):
        raise ValueError(
            "time stamps should be uniformly spaced, but found non-uniform spacing"
        )
