# -*- coding: utf-8 -*-
"""Da-TACOS Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Da-TACOS: a dataset for cover song identification and understanding. It contains two subsets,
    namely the benchmark subset (for benchmarking cover song identification systems) and the cover
    analysis subset (for analyzing the links among cover songs), with pre-extracted features and
    metadata for 15,000 and 10,000 songs, respectively. The annotations included in the metadata
    are obtained with the API of SecondHandSongs.com. All audio files we use to extract features
    are encoded in MP3 format and their sample rate is 44.1 kHz. Da-TACOS does not contain any
    audio files. For the results of our analyses on modifiable musical characteristics using the
    cover analysis subset and our initial benchmarking of 7 state-of-the-art cover song identification
    algorithms on the benchmark subset, you can look at our publication.

    For organizing the data, we use the structure of SecondHandSongs where each song is called a
    ‘performance’, and each clique (cover group) is called a ‘work’. Based on this, the file names
    of the songs are their unique performance IDs (PID, e.g. P_22), and their labels with respect
    to their cliques are their work IDs (WID, e.g. W_14).

    Metadata for each song includes:

        - performance title
        - performance artist
        - work title
        - work artist
        - release year
        - SecondHandSongs.com performance ID
        - SecondHandSongs.com work ID
        - whether the song is instrumental or not

    In addition, we matched the original metadata with MusicBrainz to obtain MusicBrainz ID (MBID),
    song length and genre/style tags. We would like to note that MusicBrainz related information is
    not available for all the songs in Da-TACOS, and since we used just our metadata for matching,
    we include all possible MBIDs for a particular songs.

    For facilitating reproducibility in cover song identification (CSI) research, we propose a framework
    for feature extraction and benchmarking in our supplementary repository: acoss. The feature extraction
    component is designed to help CSI researchers to find the most commonly used features for CSI in a
    single address. The parameter values we used to extract the features in Da-TACOS are shared in the
    same repository. Moreover, the benchmarking component includes our implementations of 7 state-of-the-art
    CSI systems. We provide the performance results of an initial benchmarking of those 7 systems on the
    benchmark subset of Da-TACOS. We encourage other CSI researchers to contribute to acoss with implementing
    their favorite feature extraction algorithms and their CSI systems to build up a knowledge base where
    CSI research can reach larger audiences.

    Pre-extracted features:

    The list of features included in Da-TACOS can be seen below. All the features are extracted with acoss
    repository that uses open-source feature extraction libraries such as Essentia, LibROSA, and Madmom.

    To facilitate the use of the dataset, we provide two options regarding the file structure.

    1. In da-tacos_benchmark_subset_single_files and da-tacos_coveranalysis_subset_single_files folders,
    we organize the data based on their respective cliques, and one file contains all the features for
    that particular song.

    .. code-block:: python

        {
            "chroma_cens": numpy.ndarray,
            "crema": numpy.ndarray,
            "hpcp": numpy.ndarray,
            "key_extractor": {
                "key": numpy.str_,
                "scale": numpy.str_,_
                "strength": numpy.float64
            },
            "madmom_features": {
                "novfn": numpy.ndarray,
                "onsets": numpy.ndarray,
                "snovfn": numpy.ndarray,
                "tempos": numpy.ndarray
            }
            "mfcc_htk": numpy.ndarray,
            "tags": list of (numpy.str_, numpy.str_)
            "label": numpy.str_,
            "track_id": numpy.str_
        }


    2. In da-tacos_benchmark_subset_FEATURE and da-tacos_coveranalysis_subset_FEATURE folders,
    the data is organized based on their cliques as well, but each of these folders contain only one
    feature per song. For instance, if you want to test your system that uses HPCP features, you can
    download da-tacos_benchmark_subset_hpcp to access the pre-computed HPCP features. An example for
    the contents in those files can be seen below:

    .. code-block:: python

        {
            "hpcp": numpy.ndarray,
            "label": numpy.str_,
            "track_id": numpy.str_
        }
"""
import json
import os
from typing import BinaryIO, Optional

import h5py
import numpy as np
from deprecated.sphinx import deprecated
from jams import JAMS
from smart_open import open

from mirdata import core, download_utils, io, jams_utils

LICENSE_INFO = """
Creative Commons Attribution Non Commercial Share Alike 4.0 International
"""
BIBTEX = """@inproceedings{yesiler2019,
    author = "Furkan Yesiler and Chris Tralie and Albin Correya and Diego F. Silva and Philip Tovstogan and Emilia G{\'{o}}mez and Xavier Serra",
    title = "{Da-TACOS}: A Dataset for Cover Song Identification and Understanding",
    booktitle = "Proc. of the 20th Int. Soc. for Music Information Retrieval Conf. (ISMIR)",
    year = "2019",
    pages = "327--334",
    address = "Delft, The Netherlands"
}"""
REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="da-tacos_metadata.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_metadata.zip?download=1",
        checksum="b8aed83c45687a6bac76de3da1799237",
    ),
    "benchmark_cens": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_cens.zip",
        url="https://zenodo.org/record/4717628/files/da-tacos_benchmark_subset_cens.zip?download=1",
        checksum="b32aab63ee401f0f8baec8aa35eb0975",
    ),
    "benchmark_crema": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_crema.zip",
        url=(
            "https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_crema.zip?download=1"
        ),
        checksum="c702a3b97a60081311bf8e7fae7b433b",
    ),
    "benchmark_hpcp": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_hpcp.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_hpcp.zip?download=1",
        checksum="f92cf3d00cc3195572381d6bbcc086de",
    ),
    "benchmark_key": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_key.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_key.zip?download=1",
        checksum="f4e6b05fa9ab46002357f371a8b0e97e",
    ),
    "benchmark_madmom": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_madmom.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_madmom.zip?download=1",
        checksum="8beb1d8fa39f95b79d5f502a41fd5f0c",
    ),
    "benchmark_mfcc": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_mfcc.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_mfcc.zip?download=1",
        checksum="a3be0cd80754043a8c238cf501062789",
    ),
    "coveranalysis_tags": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_tags.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_tags.zip?download=1",
        checksum="4b9d4cd5beca571e1d614c9a77580f8c",
    ),
    "coveranalysis_cens": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_cens.zip",
        url="https://zenodo.org/record/4717628/files/da-tacos_coveranalysis_subset_cens.zip?download=1",
        checksum="7eb56dd3a44fa7d90cc6643bc446e79b",
    ),
    "coveranalysis_crema": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_crema.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_crema.zip?download=1",
        checksum="70252fe115e1ab4c4d74698d4ad68f4b",
    ),
    "coveranalysis_hpcp": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_hpcp.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_hpcp.zip?download=1",
        checksum="961784fc2419214adf05504e9fc56cc2",
    ),
    "coveranalysis_key": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_key.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_key.zip?download=1",
        checksum="6e72db855bad5805a67382bd318eee9c",
    ),
    "coveranalysis_madmom": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_madmom.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_madmom.zip?download=1",
        checksum="42482eedfe9d9a8be9db3611b9d343b4",
    ),
    "coveranalysis_mfcc": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_mfcc.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_mfcc.zip?download=1",
        checksum="11371910cad7012daaa81a5fe9dfa1c0",
    ),
}
INDEXES = {
    "default": "1.1_full",
    "test": "1.1_full",
    "1.1_crema": core.Index(
        filename="da_tacos_index_1.1_crema.json",
        partial_download=["benchmark_crema", "coveranalysis_crema"],
    ),
    "1.1_full": core.Index(filename="da_tacos_index_1.1_full.json"),
}


class Track(core.Track):
    """da_tacos track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        subset (str): subset which the track belongs to
        work_id (str): id of work's original track
        label (str): alias of work_id
        performance_id (str): id of cover track
        cens_path (str): cens annotation path
        crema_path (str): crema annotation path
        hpcp_path (str): hpcp annotation path
        key_path (str): key annotation path
        madmom_path (str): madmom annotation path
        mfcc_path (str): mfcc annotation path
        tags_path (str): tags annotation path

    Properties:
        work_title (str): title of the work
        work_artist (str): original artist of the work
        performance_title (str): title of the performance
        performance_artist (str): artist of the performance
        release_year (str): release year
        is_instrumental (bool): True if the track is instrumental
        performance_artist_mbid (str): musicbrainz id of the performance artist
        mb_performances (dict): musicbrainz ids of performances

    Cached Properties:
        cens (np.ndarray): chroma-cens features
        hpcp (np.ndarray): hpcp features
        key (dict): key data, with keys 'key', 'scale', and 'strength'
        madmom (dict): dictionary of madmom analysis features
        mfcc (np.ndarray): mfcc features
        tags (list): list of tags
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.track_id = track_id
        self._data_home = data_home
        self.cens_path = self.get_path("cens")
        self.crema_path = self.get_path("crema")
        self.hpcp_path = self.get_path("hpcp")
        self.key_path = self.get_path("key")
        self.madmom_path = self.get_path("madmom")
        self.mfcc_path = self.get_path("mfcc")
        self.tags_path = self.get_path("tags")

        self.subset = self.track_id.split("#")[0]
        self.work_id = self.track_id.split("#")[1]
        self.label = self.work_id
        self.performance_id = self.track_id.split("#")[2]

    @property
    def work_title(self) -> str:
        return self._track_metadata.get("work_title")

    @property
    def work_artist(self) -> str:
        return self._track_metadata.get("work_artist")

    @property
    def performance_title(self) -> str:
        return self._track_metadata.get("perf_title")

    @property
    def performance_artist(self) -> str:
        return self._track_metadata.get("perf_artist")

    @property
    def release_year(self) -> str:
        return self._track_metadata.get("release_year")

    @property
    def is_instrumental(self) -> bool:
        return self._track_metadata.get("instrumental") == "Yes"

    @property
    def performance_artist_mbid(self) -> str:
        return self._track_metadata.get("perf_artist_mbid")

    @property
    def mb_performances(self) -> dict:
        return self._track_metadata.get("mb_performances")

    @core.cached_property
    def cens(self) -> Optional[np.ndarray]:
        return load_cens(self.cens_path)

    @core.cached_property
    def crema(self) -> Optional[np.ndarray]:
        return load_crema(self.crema_path)

    @core.cached_property
    def hpcp(self) -> Optional[np.ndarray]:
        return load_hpcp(self.hpcp_path)

    @core.cached_property
    def key(self) -> Optional[dict]:
        return load_key(self.key_path)

    @core.cached_property
    def madmom(self) -> Optional[dict]:
        return load_madmom(self.madmom_path)

    @core.cached_property
    def mfcc(self) -> Optional[np.ndarray]:
        return load_mfcc(self.mfcc_path)

    @core.cached_property
    def tags(self) -> Optional[list]:
        return load_tags(self.tags_path)

    def to_jams(self) -> JAMS:
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            metadata={
                "duration": 0.0,
                "work_id": self.work_id,
                "performance_id": self.performance_id,
                "subset": self.subset,
                "label": self.label,
                "cens": self.cens,
                "crema": self.crema,
                "hpcp": self.hpcp,
                "key": self.key,
                "madmom": self.madmom,
                "mfcc": self.mfcc,
                "tags": self.tags,
            }
        )


@io.coerce_to_bytes_io
def load_cens(fhandle: BinaryIO):
    """Load Da-TACOS cens features from a file.

    Args:
        fhandle (str or file-like): File-like object or path to chroma-cens file

    Returns:
        np.ndarray: cens features
    """
    with h5py.File(fhandle, "r") as open_h5_handle:
        return open_h5_handle["chroma_cens"][()]


@io.coerce_to_bytes_io
def load_crema(fhandle: BinaryIO):
    """Load Da-TACOS crema features from a file.

    Args:
        fhandle (str or file-like): File-like object or path to crema file

    Returns:
        np.ndarray: crema features
    """
    with h5py.File(fhandle, "r") as open_h5_fhandle:
        return open_h5_fhandle["crema"][()]


@io.coerce_to_bytes_io
def load_hpcp(fhandle: BinaryIO):
    """Load Da-TACOS hpcp features from a file.

    Args:
        fhandle (str or file-like): File-like object or path to hpcp file

    Returns:
        np.ndarray: hpcp features
    """
    with h5py.File(fhandle, "r") as open_h5_fhandle:
        return open_h5_fhandle["hpcp"][()]


def _dict_from_h5py(fhandle, record_key):
    """Loads dictionary information from an hdf5 file.

    Args:
        fhandle (file-like): Open file, in binary read mode
        record_key (str): the name of the record key to load

    Returns:
        dict: data loaded from record key of the open file
    """
    with h5py.File(fhandle, "r") as open_file:
        return {
            attr: open_file[record_key].attrs[attr]
            for attr in list(open_file[record_key].attrs.keys())
            if attr.lower() == attr
        }


@io.coerce_to_bytes_io
def load_key(fhandle: BinaryIO):
    """Load Da-TACOS key features from a file.

    Args:
        fhandle (str or file-like): File-like object or path to key file

    Returns:
        dict: key, mode and confidence

    Examples:
        {'key': 'C', 'scale': 'major', 'strength': 0.8449875116348267}
    """
    return _dict_from_h5py(fhandle, "key_extractor")


@io.coerce_to_bytes_io
def load_madmom(fhandle: BinaryIO):
    """Load Da-TACOS madmom features from a file.

    Args:
        fhandle (str or file-like): File-like object or path to madmom file

    Returns:
        dict: madmom features, with keys 'novfn', 'onsets', 'snovfn', 'tempos
    """
    return _dict_from_h5py(fhandle, "madmom_features")


@io.coerce_to_bytes_io
def load_mfcc(fhandle: BinaryIO):
    """Load Da-TACOS mfcc from a file.

    Args:
        fhandle (str or file-like): File-like object or path to mfcc file

    Returns:
        np.ndarray: array of mfccs over time
    """
    with h5py.File(fhandle, "r") as open_h5_fhandle:
        return open_h5_fhandle["mfcc_htk"][()]


@io.coerce_to_bytes_io
def load_tags(fhandle: BinaryIO):
    """Load Da-TACOS tags from a file.

    Args:
        fhandle (str or file-like): File-like object or path to tags file

    Returns:
        list: tags, in the form [(tag, confidence), ...]

    Example:
        [('rock', '0.127'), ('pop', '0.014'), ...]
    """
    with h5py.File(fhandle, "r") as open_file:
        return [
            (open_file["tags"][k].attrs["i0"], open_file["tags"][k].attrs["i1"])
            for k in open_file["tags"].keys()
        ]


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Da-TACOS dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="da_tacos",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_index = {}
        for subset in ["benchmark", "coveranalysis"]:
            path_subset = os.path.join(
                self.data_home,
                "da-tacos_metadata",
                "da-tacos_" + subset + "_subset_metadata.json",
            )
            try:
                with open(path_subset) as f:
                    meta = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Metadata file {} not found. Did you run .download()?".format(path_subset)
                )
            for work_id in meta.keys():
                for performance_id in meta[work_id].keys():
                    track_id = subset + "#" + work_id + "#" + performance_id
                    metadata_index[track_id] = meta[work_id][performance_id]

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.da_tacos.load_cens", version="0.3.4")
    def load_cens(self, *args, **kwargs):
        return load_cens(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.da_tacos.load_crema", version="0.3.4")
    def load_crema(self, *args, **kwargs):
        return load_crema(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.da_tacos.load_hpcp", version="0.3.4")
    def load_hpcp(self, *args, **kwargs):
        return load_hpcp(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.da_tacos.load_key", version="0.3.4")
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.da_tacos.load_mfcc", version="0.3.4")
    def load_mfcc(self, *args, **kwargs):
        return load_mfcc(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.da_tacos.load_madmom", version="0.3.4")
    def load_madmom(self, *args, **kwargs):
        return load_madmom(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.da_tacos.load_tags", version="0.3.4")
    def load_tags(self, *args, **kwargs):
        return load_tags(*args, **kwargs)

    def filter_index(self, search_key):
        """Load from Da-TACOS genre dataset the indexes that match with
        search_key.

        Args:
            search_key (str): regex to match with folds, mbid or genres

        Returns:
             dict: {`track_id`: track data}
        """
        data = {k: v for k, v in self._index["tracks"].items() if search_key in k}
        return data

    def benchmark_tracks(self):
        """Load from Da-TACOS dataset the benchmark subset tracks.

        Returns:
            dict: {`track_id`: track data}
        """
        return self.filter_index("benchmark#")

    def coveranalysis_tracks(self):
        """Load from Da-TACOS dataset the coveranalysis subset tracks.

        Returns:
            dict: {`track_id`: track data}
        """
        return self.filter_index("coveranalysis#")
