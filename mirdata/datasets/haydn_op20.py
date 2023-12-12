"""Haydn op20 Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    This dataset accompanies the Master Thesis from Nestor Napoles. It is a manually-annotated corpus of harmonic
    analysis in harm syntax.

    This dataset contains 30 pieces composed by Joseph Haydn in symbolic format, which have each been manually
    annotated with harmonic analyses.
"""
import logging
import os
from typing import List, Optional, TextIO

import numpy as np
from deprecated.sphinx import deprecated

from mirdata import core, download_utils, io, jams_utils

try:
    import music21
except ImportError:
    logging.error(
        "In order to use haydn_op20 you must have music21 installed. "
        "Please reinstall mirdata using `pip install 'mirdata[haydn_op20]'"
    )
    raise ImportError

from mirdata.annotations import ChordData, KeyData

BIBTEX = """
@dataset{nestor_napoles_lopez_2017_1095630,
  author={N\'apoles L\'opez, N\'estor},
  title={{Joseph Haydn - String Quartets Op.20 - Harmonic Analysis Annotations Dataset}},
  month=dec,
  year=2017,
  publisher={Zenodo},
  version={v1.1-alpha},
  doi={10.5281/zenodo.1095630},
  url={https://doi.org/10.5281/zenodo.1095630}
}"""

INDEXES = {
    "default": "1.3",
    "test": "1.3",
    "1.3": core.Index(filename="haydn_op20_index_1.3.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="haydnop20v1.3_annotated.zip",
        url="https://github.com/napulen/haydn_op20_harm/releases/download/v1.3/haydnop20v1.3_annotated.zip",
        checksum="1c65c8da312e1c9dda681d0496bf527f",
        destination_dir=".",
    )
}
LICENSE_INFO = "Creative Commons Attribution Non Commercial Share Alike 4.0 International."


class Track(core.Track):
    """Haydn op20 track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        title (str): title of the track
        track_id (str): track id
        humdrum_annotated_path (str): path to humdrum annotated score

    Cached Properties:
        keys (KeyData): annotated local keys.
        keys_music21 (list): annotated local keys.
        roman_numerals (list): annotated roman_numerals.
        chords (ChordData): annotated chords.
        chords_music21 (list): annotated chords.
        duration (int): relative duration
        midi_path (str): path to midi
        score (music21.stream.Score): music21 score
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)
        self.humdrum_annotated_path = self.get_path("annotations")
        self.title = os.path.splitext(self._track_paths["annotations"][0])[0]

    @core.cached_property
    def score(self) -> music21.stream.Score:
        return load_score(self.humdrum_annotated_path)

    @core.cached_property
    def keys(self) -> Optional[KeyData]:
        return load_key(self.humdrum_annotated_path)

    @core.cached_property
    def keys_music21(self) -> Optional[List[dict]]:
        return load_key_music21(self.humdrum_annotated_path)

    @core.cached_property
    def roman_numerals(self) -> Optional[List[dict]]:
        return load_roman_numerals(self.humdrum_annotated_path)

    @core.cached_property
    def chords(self) -> Optional[ChordData]:
        return load_chords(self.humdrum_annotated_path)

    @core.cached_property
    def chords_music21(self) -> Optional[List[dict]]:
        return load_chords_music21(self.humdrum_annotated_path)

    @core.cached_property
    def duration(self) -> int:
        return self.chords_music21[-1]["time"]

    @core.cached_property
    def midi_path(self) -> Optional[str]:
        logging.warning(
            "midi_path is deprecated as of 0.3.4 and will be removed in a future version."
        )
        return convert_and_save_to_midi(self.humdrum_annotated_path)

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            metadata={
                "duration": self.duration,
                "title": self.title,
                "key": self.keys,  # format is not the expected by keydata jams namespace
                "chord": self.chords,  # format is not the expected by chorddata jams namespace
                "keys_music21": self.keys_music21,
                "chords_music21": self.chords_music21,
                "roman_numerals": self.roman_numerals,
                "midi_path": self.midi_path,
                "humdrum_annotated_path": self.humdrum_annotated_path,
            }
        )


def _split_score_annotations(fhandle: TextIO):
    """Load haydn op20 score and annotations divided.

    Args:
        fhandle (str or file-like): path to hrm annotations

    Returns:
        music21.stream.Score: score in music21 format
        list: list of roman numerals [(time in seconds, roman numeral)]
    """
    score = music21.converter.parse(fhandle.name, format="humdrum")

    rna = {rn.offset: rn for rn in list(score.flat.getElementsByClass("RomanNumeral"))}
    score.remove(list(rna.values()), recurse=True)
    rna_clean = [(offset, rn) for offset, rn in rna.items() if rn]
    return score, rna_clean


@io.coerce_to_string_io
def load_score(fhandle: TextIO):
    """Load haydn op20 score with annotations from a file with music21 format
    (music21.stream.Score).

    Args:
        fhandle (str or file-like): path to score

    Returns:
        music21.stream.Score: score in music21 format
    """
    score, rna = _split_score_annotations(fhandle)
    return score


def _load_key_base(fhandle, resolution):
    """Load haydn op20 key data from a file in music21 format.

    Args:
        fhandle (str or file-like): path to key annotations
        resolution (int): the number of pulses, or ticks, per quarter note (PPQ)

    Returns:
        list: musical key data and relative time (offset (Music21Object.offset) * resolution) [(time in PPQ, local key)]
    """
    _, rna = _split_score_annotations(fhandle)
    annotations = []
    for offset, rn in rna:
        time = int(round(float(offset * resolution)))
        tonicizedKey = rn.secondaryRomanNumeralKey
        key = tonicizedKey or rn.key
        annotations.append({"time": time, "key": key})
    return annotations


def _format_key_string(key_string):
    """Format a key string to match key_mode format.

    Args:
        key_string (str): unformatted key string

    Returns:
        str: key_mode format key string
    """
    return key_string.replace("-", "b").replace(" ", ":")


@io.coerce_to_string_io
def load_key(fhandle: TextIO, resolution=28):
    """Load haydn op20 key data from a file.

    Args:
        fhandle (str or file-like): path to key annotations
        resolution (int): the number of pulses, or ticks, per quarter note (PPQ)

    Returns:
        KeyData: loaded key data
    """
    keys = _load_key_base(fhandle, resolution)
    start_times = [0]
    end_times = []
    key_names = [_format_key_string(str(keys[0]["key"]))]
    for i, k in enumerate(keys):
        this_key_string = _format_key_string(str(k["key"]))
        # if this is a new key, add it to the list
        if this_key_string != key_names[-1]:
            end_times.append(keys[i]["time"] - 1)
            start_times.append(keys[i]["time"])
            key_names.append(this_key_string)
    end_times.append(keys[-1]["time"])
    return KeyData(
        np.array([start_times, end_times]).astype(float).T,
        "ticks",
        key_names,
        "key_mode",
    )


@io.coerce_to_string_io
def load_key_music21(fhandle: TextIO, resolution=28):
    """Load haydn op20 key data from a file in music21 format.

    Args:
        fhandle (str or file-like): path to key annotations
        resolution (int): the number of pulses, or ticks, per quarter note (PPQ)

    Returns:
        list: musical key data and relative time (offset (Music21Object.offset) * resolution) [(time in PPQ, local key)]
    """
    return _load_key_base(fhandle, resolution)


@deprecated(
    reason="convert_and_save_to_midi is deprecated and will be removed in a future version",
    version="0.3.4",
)
@io.coerce_to_string_io
def convert_and_save_to_midi(fpath: TextIO):
    """Convert to midi file and return the midi path.

    Args:
        fpath (str or file-like): path to score file

    Returns:
        str: midi file path
    """
    midi_path = os.path.splitext(fpath.name)[0] + ".midi"
    score, _ = _split_score_annotations(fpath)
    score.write("midi", fp=midi_path)
    return midi_path


@io.coerce_to_string_io
def load_roman_numerals(fhandle: TextIO, resolution=28):
    """Load haydn op20 roman numerals data from a file.

    Args:
        fhandle (str or file-like): path to roman numeral annotations
        resolution (int): the number of pulses, or ticks, per quarter note (PPQ)

    Returns:
        list: musical roman numerals data and relative time (offset (Music21Object.offset) * resolution) [(time in PPQ, roman numerals)]
    """
    _, rna = _split_score_annotations(fhandle)
    annotations = []
    for offset, rn in rna:
        time = int(round(float(offset * resolution)))
        figure = rn.figure
        annotations.append({"time": time, "roman_numeral": figure})
    return annotations


def _load_chords_base(fhandle: TextIO, resolution: int = 28):
    """Load haydn op20 chords data from a file in music21 format.

    Args:
        fhandle (str or file-like): path to chord annotations
        resolution (int): the number of pulses, or ticks, per quarter note (PPQ)

    Returns:
        list: musical chords data and relative time (offset (Music21Object.offset) * resolution) [(time in PPQ, chord)]
    """
    _, rna = _split_score_annotations(fhandle)
    annotations = []
    for offset, rn in rna:
        time = int(round(float(offset * resolution)))
        chord = rn.pitchedCommonName
        annotations.append({"time": time, "chord": chord})
    return annotations


@io.coerce_to_string_io
def load_chords(fhandle: TextIO, resolution: int = 28):
    """Load haydn op20 chords data from a file.

    Args:
        fhandle (str or file-like): path to chord annotations
        resolution (int): the number of pulses, or ticks, per quarter note (PPQ)

    Returns:
        ChordData: chord annotations
    """
    chords = _load_chords_base(fhandle, resolution)
    start_times, end_times, chord_names = [0], [], [str(chords[0]["chord"])]
    for ii, k in enumerate(chords):
        if str(k["chord"]) != chord_names[-1]:
            end_times.append(chords[ii]["time"] - 1)
            start_times.append(chords[ii]["time"])
            chord_names.append(str(chords[ii]["chord"]))
    end_times.append(chords[-1]["time"])
    return ChordData(
        np.array([start_times, end_times]).astype(float).T, "ticks", chord_names, "open"
    )


@io.coerce_to_string_io
def load_chords_music21(fhandle: TextIO, resolution: int = 28):
    """Load haydn op20 chords data from a file in music21 format.

    Args:
        fhandle (str or file-like): path to chord annotations
        resolution (int): the number of pulses, or ticks, per quarter note (PPQ)

    Returns:
        list: musical chords data and relative time (offset (Music21Object.offset) * resolution) [(time in PPQ, chord)]
    """
    return _load_chords_base(fhandle, resolution)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The haydn op20 dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            name="haydn_op20",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.haydn_op20.load_score", version="0.3.4")
    def load_score(self, *args, **kwargs):
        return load_score(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.haydn_op20.load_key_music21", version="0.3.4")
    def load_key_music21(self, *args, **kwargs):
        return load_key_music21(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.haydn_op20.load_key", version="0.3.4")
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.haydn_op20.load_chords", version="0.3.4")
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.haydn_op20.load_chords_music21", version="0.3.4")
    def load_chords_music21(self, *args, **kwargs):
        return load_chords_music21(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.haydn_op20.load_roman_numerals", version="0.3.4")
    def load_roman_numerals(self, *args, **kwargs):
        return load_roman_numerals(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.haydn_op20.convert_and_save_to_midi",
        version="0.3.4",
    )
    def load_midi_path(self, *args, **kwargs):
        return convert_and_save_to_midi(*args, **kwargs)
