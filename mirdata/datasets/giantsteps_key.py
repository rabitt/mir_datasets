"""giantsteps_key Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The GiantSteps+ EDM Key Dataset includes 600 two-minute sound excerpts from various EDM subgenres, annotated with
    single-key labels, comments and confidence levels by Daniel G. Camhi, and thoroughly revised and expanded by Ángel
    Faraldo at MTG UPF. Additionally, 500 tracks have been thoroughly analysed, containing pitch-class set descriptions,
    key changes, and additional modal changes. This dataset is a revision of the original  GiantSteps Key Dataset, available
    in Github (<https://github.com/GiantSteps/giantsteps-key-dataset>) and initially described in:

    .. code-block:: latex

        Knees, P., Faraldo, Á., Herrera, P., Vogl, R., Böck, S., Hörschläger, F., Le Goff, M. (2015).
        Two Datasets for Tempo Estimation and Key Detection in Electronic Dance Music Annotated from User Corrections.
        In Proceedings of the 16th International Society for Music Information Retrieval Conference, 364–370. Málaga, Spain.

    The original audio samples belong to online audio snippets from Beatport, an online music store for DJ's and Electronic
    Dance Music Producers (<http:\\www.beatport.com>). If this dataset were used in further research, we would appreciate
    the citation of the current DOI (10.5281/zenodo.1101082) and the following doctoral dissertation, where a detailed
    description of the properties of this dataset can be found:

    .. code-block:: latex

        Ángel Faraldo (2017). Tonality Estimation in Electronic Dance Music: A Computational and Musically Informed Examination.
        PhD Thesis. Universitat Pompeu Fabra, Barcelona.

    This dataset is mainly intended to assess the performance of computational key estimation algorithms in electronic dance
    music subgenres.

    All the data of this dataset is licensed with Creative Commons Attribution Share Alike 4.0 International.
"""

import json
from typing import Dict, List, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{knees2015two,
  title={Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections},
  author={Knees, Peter and Faraldo P{\'e}rez, {\'A}ngel and Boyer, Herrera and Vogl, Richard and B{\"o}ck, Sebastian and H{\"o}rschl{\"a}ger, Florian and Le Goff, Mickael and others},
  booktitle={Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR); 2015 Oct 26-30; M{\'a}laga, Spain.[M{\'a}laga]: International Society for Music Information Retrieval, 2015. p. 364-70.},
  year={2015},
  organization={International Society for Music Information Retrieval (ISMIR)}
}"""

INDEXES = {
    "default": "+",
    "test": "+",
    "+": core.Index(filename="giantsteps_key_index_+.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="audio.zip",
        url="https://zenodo.org/record/1095691/files/audio.zip?download=1",
        checksum="8ec9ade888d5a88ce435d7fda031929b",
        destination_dir=".",
    ),
    "keys": download_utils.RemoteFileMetadata(
        filename="keys.zip",
        url="https://zenodo.org/record/1095691/files/keys.zip?download=1",
        checksum="775b7d17e009f5818544cf505b6a96fd",
        destination_dir=".",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="original_metadata.zip",
        url="https://zenodo.org/record/1095691/files/original_metadata.zip?download=1",
        checksum="54181e0f34c35d9720439750d0b08091",
        destination_dir=".",
    ),
}

LICENSE_INFO = "Creative Commons Attribution Share Alike 4.0 International."


class Track(core.Track):
    """giantsteps_key track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        keys_path (str): key annotation path
        metadata_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

    Cached Properties:
        key (str): musical key annotation
        artists (list): list of artists involved
        genres (dict): genres and subgenres
        tempo (int): crowdsourced tempo annotations in beats per minute
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.keys_path = self.get_path("key")
        self.metadata_path = self.get_path("meta")

        self.audio_path = self.get_path("audio")

        self.title = self.audio_path.replace(".mp3", "").split("/")[-1]

    @core.cached_property
    def key(self) -> Optional[str]:
        return load_key(self.keys_path)

    @core.cached_property
    def artists(self) -> Optional[List[str]]:
        return load_artist(self.metadata_path)

    @core.cached_property
    def genres(self) -> Optional[Dict[str, List[str]]]:
        return load_genre(self.metadata_path)

    @core.cached_property
    def tempo(self) -> Optional[str]:
        return load_tempo(self.metadata_path)

    @property
    def audio(self) -> Tuple[np.ndarray, float]:
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
                "artists": self.artists,
                "genres": self.genres,
                "tempo": self.tempo,
                "title": self.title,
                "key": self.key,
            },
        )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(fpath: str) -> Tuple[np.ndarray, float]:
    """Load a giantsteps_key audio file.

    Args:
        fpath (str): str pointing to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fpath, sr=None, mono=True)


@io.coerce_to_string_io
def load_key(fhandle: TextIO) -> str:
    """Load giantsteps_key format key data from a file.

    Args:
        fhandle (str or file-like): File like object or string pointing to key annotation file

    Returns:
        str: loaded key data
    """
    return fhandle.readline()


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> str:
    """Load giantsteps_key tempo data from a file.

    Args:
        fhandle (str or file-like): File-like object or string pointing to metadata annotation file

    Returns:
        str: loaded tempo data
    """
    meta = json.load(fhandle)
    return meta["bpm"]


@io.coerce_to_string_io
def load_genre(fhandle: TextIO) -> Dict[str, List[str]]:
    """Load giantsteps_key genre data from a file.

    Args:
        fhandle (str or file-like): File-like object or path pointing to metadata annotation file

    Returns:
        dict: `{'genres': [...], 'subgenres': [...]}`
    """
    meta = json.load(fhandle)
    return {
        "genres": [genre["name"] for genre in meta["genres"]],
        "sub_genres": [genre["name"] for genre in meta["sub_genres"]],
    }


@io.coerce_to_string_io
def load_artist(fhandle: TextIO) -> List[str]:
    """Load giantsteps_key tempo data from a file.

    Args:
        fhandle (str or file-like): File-like object or path pointing to metadata annotation file

    Returns:
        list: list of artists involved in the track.
    """
    meta = json.load(fhandle)

    return [artist["name"] for artist in meta["artists"]]


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The giantsteps_key dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="giantsteps_key",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.giantsteps_key.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.giantsteps_key.load_key", version="0.3.4")
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.giantsteps_key.load_tempo", version="0.3.4")
    def load_tempo(self, *args, **kwargs):
        return load_tempo(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.giantsteps_key.load_genre", version="0.3.4")
    def load_genre(self, *args, **kwargs):
        return load_genre(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.giantsteps_key.load_artist", version="0.3.4")
    def load_artist(self, *args, **kwargs):
        return load_artist(*args, **kwargs)
