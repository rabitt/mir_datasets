"""MedleyDB pitch Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    MedleyDB Pitch is a pitch-tracking subset of the MedleyDB dataset
    containing only f0-annotated, monophonic stems.

    MedleyDB is a dataset of annotated, royalty-free multitrack recordings.
    MedleyDB was curated primarily to support research on melody extraction,
    addressing important shortcomings of existing collections. For each song
    we provide melody f0 annotations as well as instrument activations for
    evaluating automatic instrument recognition.

    For more details, please visit: https://medleydb.weebly.com
"""

import csv
import json
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{bittner2014medleydb,
    Author = {Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan P},
    Booktitle = {International Society of Music Information Retrieval (ISMIR)},
    Month = {October},
    Title = {Medley{DB}: A Multitrack Dataset for Annotation-Intensive {MIR} Research},
    Year = {2014}
}"""
INDEXES = {
    "default": "3.0",
    "test": "3.0",
    "2.0": core.Index(filename="medleydb_pitch_index_2.0.json", partial_download=[]),
    "3.0": core.Index(filename="medleydb_pitch_index_3.0.json"),
}
REMOTES = {
    "notes_pyin": download_utils.RemoteFileMetadata(
        filename="medleydb-pitch-pyin-notes.zip",
        url="https://zenodo.org/record/4728793/files/medleydb-pitch-pyin-notes.zip?download=1",
        checksum="464af0c8db7b6e70d87f833eb551a8fb",
    )
}
DOWNLOAD_INFO = """
    To download this dataset, visit:
    https://zenodo.org/record/2620624#.XKZc7hNKh24
    and request access.

    Once downloaded, unzip the file MedleyDB-Pitch.zip
    and copy the result to:
    {}
"""

LICENSE_INFO = "Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)."


class Track(core.Track):
    """medleydb_pitch Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): artist
        audio_path (str): path to the audio file
        genre (str): genre
        instrument (str): instrument of the track
        notes_pyin_path (str): path to the pyin note annotation file
        pitch_path (str): path to the pitch annotation file
        title (str): title
        track_id (str): track id

    Cached Properties:
        pitch (F0Data): human annotated pitch
        notes_pyin (NoteData): notes estimated by the pyin algorithm.
            Not available in version 2.0
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.pitch_path = self.get_path("pitch")
        self.notes_pyin_path = self.get_path("notes_pyin")
        self.audio_path = self.get_path("audio")

    @property
    def instrument(self):
        return self._track_metadata.get("instrument")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def genre(self):
        return self._track_metadata.get("genre")

    @core.cached_property
    def pitch(self) -> Optional[annotations.F0Data]:
        return load_pitch(self.pitch_path)

    @core.cached_property
    def notes_pyin(self) -> Optional[annotations.NoteData]:
        return load_notes(self.notes_pyin_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

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
            f0_data=[(self.pitch, "annotated pitch")],
            note_data=[(self.notes_pyin, "pyin note estimate")],
            metadata=self._track_metadata,
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a MedleyDB audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_pitch(fhandle: TextIO) -> annotations.F0Data:
    """Load a MedleyDB pitch annotation file.

    Args:
        fhandle (str or file-like): str or file-like to pitch annotation file

    Raises:
        IOError: if the path doesn't exist

    Returns:
        F0Data: pitch annotation
    """
    times = []
    freqs = []
    voicing = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        times.append(float(line[0]))
        freq_val = float(line[1])
        freqs.append(freq_val)
        voicing.append(float(freq_val > 0))

    return annotations.F0Data(
        np.array(times), "s", np.array(freqs), "hz", np.array(voicing), "binary"
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
    intervals = []
    freqs = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        start_time = float(line[0])
        intervals.append([start_time, start_time + float(line[1])])
        freqs.append(float(line[2]))

    # if file is empty, return None
    if len(intervals) == 0:
        return None

    return annotations.NoteData(np.array(intervals), "s", np.array(freqs), "hz")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The medleydb_pitch dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="medleydb_pitch",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "medleydb_pitch_metadata.json")

        try:
            with open(metadata_path, "r") as fhandle:
                metadata = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata

    @deprecated(reason="Use mirdata.datasets.medleydb_pitch.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.medleydb_pitch.load_pitch", version="0.3.4")
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.medleydb_pitch.load_notes", version="0.3.4")
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)
