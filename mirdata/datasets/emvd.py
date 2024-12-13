"""Extreme Metal Vocals Dataset (EMVD) Loader

.. admonition:: Dataset Info
    :class: dropdown

    EMVD contains 760 tracks ranging in length from 1 to 30 seconds, for a total of 100 minutes of audio. The dataset contains vocal recordings from 27 different singers performing both clean and distorted vocal techniques without accompaniment or post-processing. Singers perform three different sustained vowels (a, i, u) as well as lyrics of their choosing.

    Tracks are annotated with a classification of the clean or extreme vocal technique used, the pitch range, and a subjective rating of the singer's performance. Four different train/eval/validation splits are provided.

    Additionally, metadata is provided for each singer and the circumstances of the recording.

    The dataset was collected by researchers of ENS Louis-Lumière, Université de Toulouse, and Ecole Centrale de Nantes, and first published in [1]. It is available under the Creative Commons Attribution 4.0 International License.

    For more details and download info, please visit:

    - https://zenodo.org/records/8406322

    - https://github.com/modantailleur/ExtremeMetalVocalsDataset

    [1] Tailleur, Modan, et al. "EMVD dataset: a dataset of extreme vocal distortion techniques used in heavy metal." arXiv preprint arXiv:2406.17732 (2024).

"""

import csv
import os
from typing import BinaryIO, Optional, Tuple, Dict

import librosa
import numpy as np

from smart_open import open
from mirdata import io, jams_utils, core, download_utils

BIBTEX = """
@misc{tailleur2024emvddatasetdatasetextreme,
      title={EMVD dataset: a dataset of extreme vocal distortion techniques used in heavy metal}, 
      author={Modan Tailleur and Julien Pinquier and Laurent Millot and Corsin Vogel and Mathieu Lagrange},
      year={2024},
      eprint={2406.17732},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2406.17732}, 
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="emvd_index_1.0.json",
        url="https://zenodo.org/records/14456645/files/emvd_index_1.0.json?download=1",
        checksum="b105d81130061aa616988ff467cafd5e",
    ),
    "sample": core.Index(filename="emvd_index_1.0_sample.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="audio.tar.xz",
        url="https://zenodo.org/records/8406322/files/audio.tar.xz?download=1",
        checksum="179f9d3aca33d1f4fb6b3d3c47192e73",
    ),
    "metadata_files": download_utils.RemoteFileMetadata(
        filename="metadata_files.csv",
        url="https://zenodo.org/records/8406322/files/metadata_files.csv?download=1",
        checksum="322685ad3df2a66eec581bc2bea8c1a0",
    ),
    "metadata_singers": download_utils.RemoteFileMetadata(
        filename="metadata_singers.csv",
        url="https://zenodo.org/records/8406322/files/metadata_singers.csv?download=1",
        checksum="cacc9a39929f96855779c5361529cc8a",
    ),
    "split_kfolds": download_utils.RemoteFileMetadata(
        filename="split_kfolds.csv",
        url="https://zenodo.org/records/8406322/files/split_kfolds.csv?download=1",
        checksum="b8f964968b5ff8d40a88f7d7e5f3a6c8",
    ),
}


LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Track(core.Track):
    """emvd Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        track_id (str): track id
        duration (float): duration of the track in seconds
        name (str): Name of the technique or effect used, or None if it does not fit a category
        range (str): One of ['Low', 'Mid', 'High'], or None if not applicable
        rank (str): A rating of the singer's performance. One of ["0", "1", "2"] (higher is better), or None if not applicable. The dataset authors recommend not using tracks with rank "0" for machine learning applications, and the tracks will be marked as "unused" in the splits.
        singer_id (str): The singer's ID
        vocalization_type (str): One of ['Technique', 'Effect', 'Other']
        vowel (str): One of ['a', 'i', 'u'] for single vowel recordings, or 'lyrics', or None if not applicable

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_path = self.get_path("audio")

    @property
    def singer_id(self) -> str:
        return self._track_metadata.get("singer_id")

    @property
    def vocalization_type(self) -> str:
        return self._track_metadata.get("type")

    @property
    def name(self) -> Optional[str]:
        return self._track_metadata.get("name")

    @property
    def range(self) -> Optional[str]:
        return self._track_metadata.get("range")

    @property
    def vowel(self):
        return self._track_metadata.get("vowel")

    @property
    def rank(self) -> Optional[str]:
        return self._track_metadata.get("authors_rank")

    @property
    def duration(self) -> float:
        return float(self._track_metadata.get("duration(s)").replace(",", "."))

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """solo vocal audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "singer_id": self.singer_id,
                "vocalization_type": self.vocalization_type,
                "name": self.name,
                "range": self.range,
                "vowel": self.vowel,
                "rank": self.rank,
                "duration": self.duration,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load MDB-stem-synth audio

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate

    """
    return librosa.load(fhandle, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Extreme Metal Vocals Dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="emvd",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata_files.csv")

        try:
            with open(metadata_path, "r") as fhandle:
                reader = csv.reader(fhandle)
                keys = next(reader)[1:]
                return {
                    os.path.splitext(row[0])[0]: {
                        key: (val if val != "-" else None)
                        for key, val in zip(keys, row[1:])
                    }
                    for row in reader
                }
        except FileNotFoundError:
            raise FileNotFoundError("Metadata file not found. Did you run .download()?")

    @core.cached_property
    def _singer_metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata_singers.csv")
        try:
            with open(metadata_path, "r") as fhandle:
                reader = csv.reader(fhandle)
                keys = next(reader)[1:]
                return {
                    row[0]: {key: val for key, val in zip(keys, row[1:])}
                    for row in reader
                }
        except FileNotFoundError:
            raise FileNotFoundError(
                "Singer metadata not found. Did you run .download()?"
            )

    @core.cached_property
    def _kfold_splits(self):
        split_path = os.path.join(self.data_home, "split_kfolds.csv")

        try:
            with open(split_path, "r") as fhandle:
                reader = csv.reader(fhandle)
                folds = next(reader)[1:]

                splits = {
                    fold: {
                        "train": list(),
                        "valid": list(),
                        "eval": list(),
                        "unused": list(),
                    }
                    for fold in folds
                }

                for row in reader:
                    track_id = os.path.splitext(row[0])[0]
                    for fold, split in zip(folds, row[1:]):
                        if split == "-":
                            split = "unused"
                        splits[fold][split].append(track_id)

        except FileNotFoundError:
            raise FileNotFoundError(
                "Split Kfolds file not found. Did you run .download()?"
            )

        return splits

    def get_track_splits(self, split_id: str = "split0"):
        """Get the split of the dataset

        Args:
            split_id (str): which split to use (one of "split0", "split1", "split2", "split3")

        Returns:
            dict: split of the dataset

        """
        if split_id not in self._kfold_splits:
            raise ValueError(
                f"split_id must be one of {list(self._kfold_splits.keys())}"
            )

        return self._kfold_splits[split_id]

    def get_singer_info(self, singer_id: str) -> Dict[str, str]:
        """Get singer metadata

        Args:
            singer_id (str): singer_id

        Returns:
            dict: singer metadata

        """
        return self._singer_metadata[singer_id]
