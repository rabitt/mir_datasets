"""Tonality classicalDB Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Tonality classicalDB Dataset includes 881 classical musical pieces across different styles from s.XVII to s.XX
    annotated with single-key labels.

    Tonality classicalDB Dataset was created as part of:

    .. code-block:: latex

        Gómez, E. (2006). PhD Thesis. Tonal description of music audio signals.
        Department of Information and Communication Technologies.

    This dataset is mainly intended to assess the performance of computational key estimation algorithms in classical music.

    2020 note: The audio is privates. If you don't have the original audio collection, you could create it from your private collection
    because most of the recordings are well known. To this end, we provide musicbrainz metadata. Moreover, we have added the spectrum and
    HPCP chromagram of each audio.

    This dataset can be used with mirdata library:
    https://github.com/mir-dataset-loaders/mirdata

    Spectrum features have been computed as is shown here:
    https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_spectrum_features.ipynb

    HPCP chromagram has been computed as is shown here:
    https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_HPCP_features.ipynb

    Musicbrainz metadata has been computed as is shown here:
    https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_musicbrainz_metadata.ipynb
"""

import csv
import json
from typing import Any, BinaryIO, Dict, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """@article{gomez2006tonal,
  title={Tonal description of music audio signals},
  author={G{\'o}mez, Emilia},
  journal={Department of Information and Communication Technologies},
  year={2006}
}"""
INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="tonality_classicaldb_index_1.0.json"),
}
REMOTES = {
    "keys": download_utils.RemoteFileMetadata(
        filename="keys.zip",
        url="https://zenodo.org/record/4283868/files/keys.zip?download=1",
        checksum="5d58978783de846f9cb337352e7d2612",
        destination_dir=".",
    ),
    "musicbrainz_metadata": download_utils.RemoteFileMetadata(
        filename="musicbrainz_metadata.zip",
        url="https://zenodo.org/record/4283868/files/musicbrainz_metadata.zip?download=1",
        checksum="4a77ecc6a9410a59feeffa1152cb6edc",
        destination_dir=".",
    ),
    "HPCPs": download_utils.RemoteFileMetadata(
        filename="HPCPs.zip",
        url="https://zenodo.org/record/4283868/files/HPCPs.zip?download=1",
        checksum="66d1ca70376109a42d0bac1306691599",
        destination_dir=".",
    ),
    "spectrums": download_utils.RemoteFileMetadata(
        filename="spectrums.zip",
        url="https://zenodo.org/record/4283868/files/spectrums.zip?download=1",
        checksum="63a79033d608ba95fb559a33e2f70d3a",
        destination_dir=".",
    ),
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the Tonality classicalDB dataset are not available
    for download. If you have the tonality classicalDB audio dataset, place the contents into
    a folder called classicaldb with the following structure:
        > classicaldb/
            > audio/
            > keys/
            > spectrums/
            > HPCPs/
            > musicbrainz_metadata/
    and copy the folder to {} directory
"""

LICENSE_INFO = "Creative Commons Attribution Non Commercial Share Alike 4.0 International."


class Track(core.Track):
    """tonality_classicaldb track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        key_path (str): key annotation path
        title (str): title of the track
        track_id (str): track id

    Cached Properties:
        key (str): key annotation
        spectrum (np.array): computed audio spectrum
        hpcp (np.array): computed hpcp
        musicbrainz_metadata (dict): MusicBrainz metadata
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.key_path = self.get_path("key")
        self.spectrum_path = self.get_path("spectrum")
        self.musicbrainz_path = self.get_path("mb")
        self.hpcp_path = self.get_path("HPCP")

        self.audio_path = self.get_path("audio")

        self.title = self.audio_path.replace(".wav", "").split("/")[-1]

    @core.cached_property
    def key(self) -> Optional[str]:
        return load_key(self.key_path)

    @core.cached_property
    def spectrum(self) -> Optional[np.ndarray]:
        return load_spectrum(self.spectrum_path)

    @core.cached_property
    def hpcp(self) -> Optional[np.ndarray]:
        return load_hpcp(self.hpcp_path)

    @core.cached_property
    def musicbrainz_metadata(self) -> Optional[Dict[Any, Any]]:
        return load_musicbrainz(self.musicbrainz_path)

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
            metadata={
                "title": self.title,
                "key": self.key,
                "spectrum": self.spectrum,
                "hpcp": self.hpcp,
                "musicbrainz_metatada": self.musicbrainz_metadata,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Tonality classicalDB audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_key(fhandle: TextIO) -> str:
    """Load Tonality classicalDB format key data from a file.

    Args:
        fhandle (str or file-like): File-like object or path to key annotation file

    Returns:
        str: musical key data
    """
    reader = csv.reader(fhandle, delimiter="\n")
    key = next(reader)[0]

    return key.replace("\t", " ")


@io.coerce_to_string_io
def load_spectrum(fhandle: TextIO) -> np.ndarray:
    """Load Tonality classicalDB spectrum data from a file.

    Args:
        fhandle (str or file-like): File-like object or path to spectrum file

    Returns:
        np.ndarray: spectrum data
    """
    data = json.load(fhandle)
    spectrum = [list(map(complex, x)) for x in data["spectrum"]]
    return np.array(spectrum)


@io.coerce_to_string_io
def load_hpcp(fhandle: TextIO) -> np.ndarray:
    """Load Tonality classicalDB HPCP feature from a file.

    Args:
        fhandle (str or file-like): File-like object or path to HPCP file

    Returns:
        np.ndarray: loaded HPCP data
    """
    data = json.load(fhandle)
    return np.array(data["hpcp"])


@io.coerce_to_string_io
def load_musicbrainz(fhandle: TextIO) -> Dict[Any, Any]:
    """Load Tonality classicalDB musicbraiz metadata from a file.

    Args:
        fhandle (str or file-like): File-like object or path to musicbrainz metadata file

    Returns:
        dict: musicbrainz metadata
    """
    return json.load(fhandle)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The tonality_classicaldb dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="tonality_classicaldb",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.tonality_classicaldb.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.tonality_classicaldb.load_key", version="0.3.4")
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.tonality_classicaldb.load_spectrum",
        version="0.3.4",
    )
    def load_spectrum(self, *args, **kwargs):
        return load_spectrum(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.tonality_classicaldb.load_hpcp", version="0.3.4")
    def load_hpcp(self, *args, **kwargs):
        return load_hpcp(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.tonality_classicaldb.load_musicbrainz",
        version="0.3.4",
    )
    def load_musicbrainz(self, *args, **kwargs):
        return load_musicbrainz(*args, **kwargs)
