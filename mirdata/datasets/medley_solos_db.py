"""Medley-solos-DB Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Medley-solos-DB is a cross-collection dataset for automatic musical instrument
    recognition in solo recordings. It consists of a training set of 3-second audio
    clips, which are extracted from the MedleyDB dataset (Bittner et al., ISMIR 2014)
    as well as a test set of 3-second clips, which are extracted from the solosDB
    dataset (Essid et al., IEEE TASLP 2009).

    Each of these clips contains a single instrument among a taxonomy of eight:

        0. clarinet,
        1. distorted electric guitar,
        2. female singer,
        3. flute,
        4. piano,
        5. tenor saxophone,
        6. trumpet, and
        7. violin.

    The Medley-solos-DB dataset is the dataset that is used in the benchmarks of
    musical instrument recognition in the publications of Lostanlen and Cella
    (ISMIR 2016) and Andén et al. (IEEE TSP 2019).
"""

import csv
import os
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{lostanlen2019ismir,
    title={Deep Convolutional Networks in the Pitch Spiral for Musical Instrument Recognition},
    author={Lostanlen, Vincent and Cella, Carmine Emanuele},
    booktitle={International Society of Music Information Retrieval (ISMIR)},
    year={2016}
}"""

INDEXES = {
    "default": "1.2",
    "test": "1.2",
    "1.2": core.Index(filename="medley_solos_db_index_1.2.json"),
}

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="Medley-solos-DB_metadata.csv",
        url="https://zenodo.org/record/3464194/files/Medley-solos-DB_metadata.csv?download=1",
        checksum="fda6a589c56785f2195c9227809c521a",
        destination_dir="annotation",
    ),
    "audio": download_utils.RemoteFileMetadata(
        filename="Medley-solos-DB.tar.gz",
        url="https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz?download=1",
        checksum="f5facf398793ef5c1f80c013afdf3e5f",
        destination_dir="audio",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International."


class Track(core.Track):
    """medley_solos_db Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        instrument (str): instrument encoded by its English name
        instrument_id (int): instrument encoded as an integer
        song_id (int): song encoded as an integer
        subset (str): either equal to 'train', 'validation', or 'test'
        track_id (str): track id
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

    @property
    def instrument(self):
        return self._track_metadata.get("instrument")

    @property
    def instrument_id(self):
        return self._track_metadata.get("instrument_id")

    @property
    def song_id(self):
        return self._track_metadata.get("song_id")

    @property
    def subset(self):
        return self._track_metadata.get("subset")

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
        return jams_utils.jams_converter(audio_path=self.audio_path, metadata=self._track_metadata)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Medley Solos DB audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=22050, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The medley_solos_db dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="medley_solos_db",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "annotation", "Medley-solos-DB_metadata.csv")

        metadata_index = {}
        try:
            with open(metadata_path, "r") as fhandle:
                csv_reader = csv.DictReader(fhandle, delimiter=",")
                for row in csv_reader:
                    metadata_index[str(row["uuid4"])] = {
                        "subset": row["subset"],
                        "instrument": row["instrument"],
                        "instrument_id": int(row["instrument_id"]),
                        "song_id": int(row["song_id"]),
                    }
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.medley_solos_db.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
