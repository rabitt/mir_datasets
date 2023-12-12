"""Freesound One-Shot Percussive Sounds Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Introduction:

    This dataset contains 10254 one-shot (single event) percussive sounds from freesound.org, a timbral
    analysis computed by two different extractors (FreesoundExtractor from Essentia and AudioCommons Extractor),
    and a list of tags. There is also metadata information about the audio file, since the audio specifications
    are not the same along all the dataset tracks. The analysis data was used to train the generative model
    for "Neural Percussive Synthesis Parameterised by High-Level Timbral Features".

    Dataset Construction:

    To collect this dataset, the following steps were performed:
    * Freesound was queried with words associated with percussive instruments, such as "percussion", "kick",
    "wood" or "clave". Only sounds with less than one second of effective duration were selected.
    * This stage retrieved some audio clips that contained multiple sound events or that were of low quality.
    Therefore, we listened to all the retrieved sounds and manually discarded the sounds presenting one of these
    characteristics. For this, the percussive-annotator was used (https://github.com/xavierfav/percussive-annotator).
    This tool allows the user to annotate a dataset that focuses on percussive sounds.
    * The sounds were then cut or padded to have 1-second length, normalized and downsampled to 16kHz.
    * Finally, the sounds were analyzed with the AudioCommons Extractor, to obtain the AudioCommons timbral
    descriptors.

    Authors and Contact:

    This dataset was developed by António Ramires, Pritish Chadna, Xavier Favory, Emilia Gómez and Xavier Serra.
    Any questions related to this dataset please contact:
    António Ramires (antonio.ramires@upf.edu / aframires@gmail.com)

    Acknowledgements:

    This work has received funding from the European Union's Horizon 2020 research and innovation programme under
    the Marie Skłodowska-Curie grant agreement No. 765068 (MIP-Frontiers).
    This work has received funding from the European Union's Horizon 2020 research and innovation programme under
    grant agreement No. 770376 (TROMPA).
"""

import json
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """
@inproceedings{ramires2020, 
    author = "Antonio Ramires and Pritish Chandna and Xavier Favory and Emilia Gómez and Xavier Serra",
    title = "Neural Percussive Synthesis Parametrerised by High-Level Timbral Features",
    booktitle = "Proc. of the IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)",
    year = "2020"
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="freesound_one_shot_percussive_sounds_index_1.0.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="one_shot_percussive_sounds.zip",
        url="https://zenodo.org/record/3665275/files/one_shot_percussive_sounds.zip?download=1",
        checksum="278994c2a7b92a24a4daad99f40c13db",
    ),
    "analysis": download_utils.RemoteFileMetadata(
        filename="analysis.zip",
        url="https://zenodo.org/record/3665275/files/analysis.zip?download=1",
        checksum="c67ce39d5aa6c6a7f88eedf7eb7d933e",
    ),
    "sound_info_analysis": download_utils.RemoteFileMetadata(
        filename="sound_info_analysis.json",
        url="https://zenodo.org/record/4687854/files/sound_info_analysis.json?download=1",
        checksum="b51913a801bd59c2583d5f0e6f3c05b9",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="licenses.txt",
        url="https://zenodo.org/record/3665275/files/licenses.txt?download=1",
        checksum="25f95a0e38d3ac4ae868f56c378fbccb",
    ),
    "readme": download_utils.RemoteFileMetadata(
        filename="README.md",
        url="https://zenodo.org/record/3665275/files/README.md?download=1",
        checksum="afec91c033db607e2fc83c09940abd15",
    ),
}

LICENSE_INFO = """
The dataset is licensed under The Creative Commons Attribution Non Commercial Share Alike 4.0 International.
Please check the specific license of each sound by running track.license
"""


class Track(core.Track):
    """Freesound one-shot percussive sounds track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/freesound_one_shot_percussive_sounds`

    Attributes:
        file_metadata_path (str): local path where the analysis file is stored and from where we get the file metadata
        audio_path(str): local path where audio file is stored
        track_id (str): track id
        filename (str): filename of the track
        username (str): username of the Freesound uploader of the track
        license (str): link to license of the track file
        tags (list): list of tags of the track
        freesound_preview_urls (dict): dict of Freesound previews urls of the track
        freesound_analysis (str): dict of analysis parameters computed in Freesound using Essentia extractor
        audiocommons_analysis (str): dict of analysis parameters computed using AudioCommons Extractor

    Cached Properties:
        file_metadata (dict): metadata parameters of the track file in form of Python dictionary
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.file_metadata_path = self.get_path("analysis")
        self.audio_path = self.get_path("audio")

    @property
    def tags(self):
        return self._track_metadata.get("tags")

    @property
    def freesound_analysis(self):
        return self._track_metadata.get("analysis")

    @property
    def audiocommons_analysis(self):
        return self._track_metadata.get("ac_analysis")

    @property
    def freesound_preview_urls(self):
        return self._track_metadata.get("previews")

    @property
    def filename(self):
        return self._track_metadata.get("name")

    @property
    def username(self):
        return self._track_metadata.get("username")

    @property
    def license(self):
        return self._track_metadata.get("license")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    @core.cached_property
    def file_metadata(self) -> Optional[dict]:
        return load_file_metadata(self.file_metadata_path)

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        jams_metadata = dict(self._track_metadata)
        jams_metadata.update(self.file_metadata)
        return jams_utils.jams_converter(audio_path=self.audio_path, metadata=jams_metadata)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load the track audio file.

    Args:
        fhandle (str): path to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=16000, mono=True)


@io.coerce_to_string_io
def load_file_metadata(fhandle: TextIO) -> Optional[dict]:
    """Extract file metadata from analysis json file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to f0 annotation file

    Returns:
        analysis: track analysis dict
    """
    file_metadata = json.load(fhandle)
    # Dropping analysis keys that are included in dataset general metadata files
    keys_to_drop = [
        "loudness",
        "dynamic_range",
        "temporal_centroid",
        "log_attack_time",
        "single_event",
        "hardness",
        "depth",
        "brightness",
        "roughness",
        "warmth",
        "sharpness",
        "boominess",
        "reverb",
    ]

    for key in keys_to_drop:
        file_metadata.pop(key)

    return file_metadata


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Freesound One-Shot Percussive Sounds dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="freesound_one_shot_percussive_sounds",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        license_path = os.path.join(self.data_home, "licenses.txt")
        sound_info_path = os.path.join(self.data_home, "sound_info_analysis.json")

        metadata = {}
        try:
            with open(sound_info_path, "r", errors="ignore") as f:
                sound_info = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        for track in sound_info:
            track_id = str(track.pop("id"))
            metadata[track_id] = track

        try:
            with open(license_path, "r", errors="ignore") as f:
                license_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Licenses file not found. Did you run .download()?")

        for track_key in license_dict.keys():
            metadata[track_key]["username"] = license_dict[track_key].get("username")
            metadata[track_key]["license"] = license_dict[track_key].get("license")

        return metadata

    @deprecated(
        reason="Use mirdata.datasets.freesound_one_shot_percussive_sounds.load_audio",
        version="0.3.4",
    )
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.freesound_one_shot_percussive_sounds.load_file_metadata",
        version="0.3.4",
    )
    def load_file_metadata(self, *args, **kwargs):
        return load_file_metadata(*args, **kwargs)
