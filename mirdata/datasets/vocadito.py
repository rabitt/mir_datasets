"""Vocadito Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    vocadito is a dataset of 40 short excerpts of solo, monophonic singing. The excerpts are sung in 7 different languages by singers with varying of levels of training, and are recorded on a variety of devices.

    Annotations are labeled by trained musicians. For each excerpt, we provide:

    frame-level f0 annotations
    2 versions of note annotations (from 2 different annotators)
    lyrics
    language

    For more details, please visit: https://zenodo.org/record/5578807
"""
import csv
import os
from typing import BinaryIO, List, Optional, TextIO, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@techreport{bittner2021vocadito,
      title={vocadito: A dataset of solo vocals with $f_0$, note, and lyric annotations}, 
      author={Rachel M. Bittner and Katherine Pasalo and Juan José Bosch and Gabriel Meseguer-Brocal and David Rubinstein},
      year={2021},
      institution={Spotify},
      number={2110.05580},
      note={https://arxiv.org/abs/2110.05580}
}
"""

INDEXES = {
    "default": "1",
    "test": "1",
    "1": core.Index(filename="vocadito_index_1.json"),
}

REMOTES = {
    "zenodo": download_utils.RemoteFileMetadata(
        filename="Vocadito.zip",
        url="https://zenodo.org/record/5578807/files/vocadito.zip?download=1",
        checksum="dea40fd18f14d899643c4ba221b33a46",
    )
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Track(core.Track):
    """Vocadito Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        f0_path (str): path to the track's f0 annotation file
        lyrics_path (str): path to the track's lyric annotation file
        notes_a1_path (str): path to the track's note annotation file for annotator A1
        notes_a2_path (str): path to the track's note annotation file for annotator A2
        track_id (str): track id
        singer_id (str): singer id
        average_pitch_midi (int): Average pitch in midi, computed from the f0 annotation
        language (str): The track's language. May contain multiple languages.

    Cached Properties:
        f0 (F0Data): human-annotated singing voice pitch
        lyrics (List[List[str]]): human-annotated lyrics
        notes_a1 (NoteData): human-annotated notes by annotator A1
        notes_a2 (NoteData): human-annotated notes by annotator A2
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.f0_path = self.get_path("f0")
        self.lyrics_path = self.get_path("lyrics")
        self.notes_a1_path = self.get_path("notesA1")
        self.notes_a2_path = self.get_path("notesA2")

        self.audio_path = self.get_path("audio")

    @property
    def singer_id(self):
        return self._track_metadata.get("singer_id")

    @property
    def average_pitch_midi(self):
        return self._track_metadata.get("average_pitch_midi")

    @property
    def language(self):
        return self._track_metadata.get("language")

    @core.cached_property
    def f0(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_path)

    @core.cached_property
    def lyrics(self) -> Optional[List[List[str]]]:
        return load_lyrics(self.lyrics_path)

    @core.cached_property
    def notes_a1(self) -> Optional[annotations.NoteData]:
        return load_notes(self.notes_a1_path)

    @core.cached_property
    def notes_a2(self) -> Optional[annotations.NoteData]:
        return load_notes(self.notes_a2_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """Solo vocal audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.f0, None)],
            note_data=[
                (self.notes_a1, "notes - Annotator 1"),
                (self.notes_a2, "notes - Annotator 2"),
            ],
            metadata={
                "singer_id": self.singer_id,
                "average_pitch_midi": int(self.average_pitch_midi),
                "language": self.language,
                "track_id": self.track_id,
                "lyrics": self.lyrics,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load vocadito vocal audio.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_f0(fhandle: TextIO) -> annotations.F0Data:
    """Load a vocadito f0 annotation.

    Args:
        fhandle (str or file-like): File-like object or path to f0 annotation file

    Raises:
        IOError: If f0_path does not exist

    Returns:
        F0Data: the f0 annotation data
    """
    times_frequencies = np.genfromtxt(fhandle, delimiter=",")
    return annotations.F0Data(
        times=times_frequencies[:, 0],
        time_unit="s",
        frequencies=times_frequencies[:, 1],
        frequency_unit="hz",
        voicing=(times_frequencies[:, 1] > 0).astype(np.float64),
        voicing_unit="binary",
    )


@io.coerce_to_string_io
def load_notes(fhandle: TextIO) -> Optional[annotations.NoteData]:
    """Load a note annotation file.

    Args:
        fhandle (str or file-like): str or file-like to note annotation file

    Raises:
        IOError: if file doesn't exist

    Returns:
        NoteData: note annotation
    """
    notes = np.genfromtxt(fhandle, delimiter=",")
    return annotations.NoteData(
        intervals=np.column_stack((notes[:, 0], notes[:, 0] + notes[:, 2])),
        interval_unit="s",
        pitches=notes[:, 1],
        pitch_unit="hz",
    )


@io.coerce_to_string_io
def load_lyrics(fhandle: TextIO) -> List[List[str]]:
    """Load a lyrics annotation.

    Args:
        fhandle (str or file-like): File-like object or path to lyric annotation file

    Raises:
        IOError: if lyrics_path does not exist

    Returns:
        LyricData: lyric annotation data
    """
    return list(csv.reader(fhandle, delimiter=" "))


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The vocadito dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="vocadito",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "vocadito_metadata.csv")
        try:
            with open(metadata_path, "r") as fhandle:
                return {
                    row["track_id"]: {
                        "singer_id": row["singer_id"],
                        "average_pitch_midi": int(row["average_pitch"]),
                        "language": row["language"],
                    }
                    for row in csv.DictReader(fhandle)
                }
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
