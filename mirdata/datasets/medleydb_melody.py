"""MedleyDB melody Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    MedleyDB melody is a subset of the MedleyDB dataset containing only
    the mixtures and melody annotations.

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

from mirdata import annotations, core, io, jams_utils

BIBTEX = """@inproceedings{bittner2014medleydb,
    Author = {Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan P},
    Booktitle = {International Society of Music Information Retrieval (ISMIR)},
    Month = {October},
    Title = {Medley{DB}: A Multitrack Dataset for Annotation-Intensive {MIR} Research},
    Year = {2014}
}"""
INDEXES = {
    "default": "5.0",
    "test": "5.0",
    "5.0": core.Index(filename="medleydb_melody_index_5.0.json"),
}

DOWNLOAD_INFO = """
    To download this dataset, visit:
    https://zenodo.org/record/2628782#.XKZdABNKh24
    and request access.

    Once downloaded, unzip the file MedleyDB-Melody.zip
    and copy the result to:
    {}
"""

LICENSE_INFO = "Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)."


class Track(core.Track):
    """medleydb_melody Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): artist
        audio_path (str): path to the audio file
        genre (str): genre
        is_excerpt (bool): True if the track is an excerpt
        is_instrumental (bool): True of the track does not contain vocals
        melody1_path (str): path to the melody1 annotation file
        melody2_path (str): path to the melody2 annotation file
        melody3_path (str): path to the melody3 annotation file
        n_sources (int): Number of instruments in the track
        title (str): title
        track_id (str): track id

    Cached Properties:
        melody1 (F0Data): the pitch of the single most predominant source (often the voice)
        melody2 (F0Data): the pitch of the predominant source for each point in time
        melody3 (MultiF0Data): the pitch of any melodic source. Allows for more than one f0 value at a time
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.melody1_path = self.get_path("melody1")
        self.melody2_path = self.get_path("melody2")
        self.melody3_path = self.get_path("melody3")

        self.audio_path = self.get_path("audio")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def genre(self):
        return self._track_metadata.get("genre")

    @property
    def is_excerpt(self):
        return self._track_metadata.get("is_excerpt")

    @property
    def is_instrumental(self):
        return self._track_metadata.get("is_instrumental")

    @property
    def n_sources(self):
        return self._track_metadata.get("n_sources")

    @core.cached_property
    def melody1(self) -> Optional[annotations.F0Data]:
        return load_melody(self.melody1_path)

    @core.cached_property
    def melody2(self) -> Optional[annotations.F0Data]:
        return load_melody(self.melody2_path)

    @core.cached_property
    def melody3(self) -> Optional[annotations.MultiF0Data]:
        return load_melody3(self.melody3_path)

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
        # jams does not support multiF0, so we skip melody3
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.melody1, "melody1"), (self.melody2, "melody2")],
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
def load_melody(fhandle: TextIO) -> annotations.F0Data:
    """Load a MedleyDB melody1 or melody2 annotation file.

    Args:
        fhandle (str or file-like): File-like object or path to a melody annotation file

    Raises:
        IOError: if melody_path does not exist

    Returns:
        F0Data: melody data
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

    times = np.array(times)  # type: ignore
    freqs = np.array(freqs)  # type: ignore
    voicing = np.array(voicing)  # type: ignore
    return annotations.F0Data(times, "s", freqs, "hz", voicing, "binary")


@io.coerce_to_string_io
def load_melody3(fhandle: TextIO) -> annotations.MultiF0Data:
    """Load a MedleyDB melody3 annotation file.

    Args:
        fhandle (str or file-like): File-like object or melody 3 melody annotation path

    Raises:
        IOError: if melody_path does not exist

    Returns:
        MultiF0Data: melody 3 annotation data
    """
    times = []
    freqs_list = []
    conf_list = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        times.append(float(line[0]))
        freqs_list.append([float(v) for v in line[1:] if float(v) != 0])
        conf_list.append([1.0 for v in line[1:] if float(v) != 0])

    times = np.array(times)  # type: ignore
    melody_data = annotations.MultiF0Data(times, "s", freqs_list, "hz", conf_list, "binary")
    return melody_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The medleydb_melody dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="medleydb_melody",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "medleydb_melody_metadata.json")

        try:
            with open(metadata_path, "r") as fhandle:
                metadata = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata

    @deprecated(reason="Use mirdata.datasets.medleydb_melody.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.medleydb_melody.load_melody", version="0.3.4")
    def load_melody(self, *args, **kwargs):
        return load_melody(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.medleydb_melody.load_melody3", version="0.3.4")
    def load_melody3(self, *args, **kwargs):
        return load_melody3(*args, **kwargs)
