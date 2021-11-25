"""OpenMIC-2018 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    OpenMIC-2018 is a dataset of 20000 excerpts of polyphonic audio recordings
    in the Free Music Archive (FMA).
    This dataset was produced by Spotify and New York University.
    The collection has been partially annotated for the presence or absence of
    20 instrument categories by workers on a crowd-sourcing platform.

    Each excerpt is 10 seconds long, and no two excerpts come from the same
    recording.

    Each of the 20 instrument classes is guaranteed to have at least 500
    positive examples and 1500 observations (positive or negative) in total.
    The style and genre of recordings is mixed and biased with respect to
    the FMA, but designed to ensure sufficient representation of each
    instrument category.

    Note that the excerpts are partially annotated: only some of the instrument
    labels will be observed for any track, and many are unobserved.
    Annotations include a confidence rating derived from the inter-annotator
    agreement on the track.
    Raw (disaggregated) annotations are also provided.

    In addition to raw audio, pre-computed features generated by the
    VGGish model are provided.

    A pre-registered partition of the data has been constructed to ensure
    reproducible evaluation with approximately balanced class presentations.

    All excerpts are permissively licensed (Creative Commons or Public Domain).

    For more details, please visit: https://zenodo.org/record/1432913
"""
import json
import re
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Dict, List

import librosa
import numpy as np
import pandas as pd
from smart_open import open

from mirdata import download_utils, jams_utils, core, io

BIBTEX = """
@inproceedings{DBLP:conf/ismir/HumphreyDM18,
  author    = {Eric Humphrey and
               Simon Durand and
               Brian McFee},
  editor    = {Emilia G{\'{o}}mez and
               Xiao Hu and
               Eric Humphrey and
               Emmanouil Benetos},
  title     = {OpenMIC-2018: An Open Data-set for Multiple Instrument Recognition},
  booktitle = {Proceedings of the 19th International Society for Music Information
               Retrieval Conference, {ISMIR} 2018, Paris, France, September 23-27,
               2018},
  pages     = {438--444},
  year      = {2018},
  url       = {http://ismir2018.ircam.fr/doc/pdfs/248\_Paper.pdf},
  timestamp = {Thu, 12 Mar 2020 11:33:14 +0100},
  biburl    = {https://dblp.org/rec/conf/ismir/HumphreyDM18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
},
@dataset{humphrey_eric_j_2018_1432913,
  author       = {Humphrey, Eric J. and
                  Durand, Simon and
                  McFee, Brian},
  title        = {OpenMIC-2018},
  month        = sep,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.1432913},
  url          = {https://doi.org/10.5281/zenodo.1432913}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="openmic2018_index.json"),
}

REMOTES = {
    "remote_data": download_utils.RemoteFileMetadata(
        filename="openmic-2018-v1.0.0.tgz",
        url="https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz?download=1",
        checksum="e4ccf187e2bb5ab2e115416e8aafe7f4",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"

INSTRUMENTS = {
    "accordion": 0,
    "banjo": 1,
    "bass": 2,
    "cello": 3,
    "clarinet": 4,
    "cymbals": 5,
    "drums": 6,
    "flute": 7,
    "guitar": 8,
    "mallet_percussion": 9,
    "mandolin": 10,
    "organ": 11,
    "piano": 12,
    "saxophone": 13,
    "synthesizer": 14,
    "trombone": 15,
    "trumpet": 16,
    "ukulele": 17,
    "violin": 18,
    "voice": 19,
}


class Track(core.Track):
    """openmic2018 Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the audio file
        split (str): string identifier for train/test split
        track_id (str): track id
        vggish_path (str): path to pre-computed VGGish features
        artist (str): name of the artist
        audio (tuple): audio time series and sampling rate (mono, 44100 Hz)
        genres (list): list of strings denoting genres for this track
        instruments (dict): instrument name -> confidence (>0.5 indicates presence)
        start_time (float): time (in seconds) where this excerpt occurs in the full recording
        title (str): title of the track
        url (str): web address to access the original recording on the Free Music Archive
        vggish (tuple): np.ndarrays for frame times and pre-computed VGGish features

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        # -- add any dataset specific attributes here
        self.audio_path = self.get_path("audio")
        self.vggish_path = self.get_path("vggish")

    @property
    def split(self):
        """Get the pre-defined split"""
        return self._track_metadata.get("split")

    @property
    def instruments(self) -> Dict[str, float]:
        """The instrments for this track.

        Each observed instrument for the track receives a score between 0 and 1,
        corresponding to the number of annotators who believe the instrument to be
        present.

        A score less than 0.5 indicates that the instrument is probably not present.

        Returns:
            * dict : instrument name -> confidence score
        """

        scores = dict()
        for k in INSTRUMENTS:
            inst_score = self._track_metadata.get(k, np.nan)
            if np.isnan(inst_score):
                continue
            scores[k] = inst_score
        return scores

    @property
    def genres(self) -> Optional[List[str]]:
        """The FMA genres of the track.

        Returns:
            * genres: list of genre strings
        """

        return list(
            [g["genre_title"] for g in self._track_metadata.get("track_genres")]
        )

    @property
    def artist(self) -> Optional[str]:
        """The artist of the track.

        Returns:
            * artist
        """

        return self._track_metadata.get("artist_name")

    @property
    def title(self) -> Optional[str]:
        """The title of the track.

        Returns:
            * title
        """
        return self._track_metadata.get("track_title")

    @property
    def url(self) -> Optional[str]:
        """The URL on Free Music Archive for this track

        Returns:
            * str - URL
        """
        return self._track_metadata.get("track_url")

    @property
    def start_time(self) -> float:
        """The starting time (in seconds) of the selected excerpt within the track.

        Returns:
            * float - starting time
        """
        return self._track_metadata.get("start_time")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def vggish(self) -> Tuple[np.ndarray, np.ndarray]:
        """The track's pre-computed VGGish features

        Returns:
            * np.ndarray - time indices (seconds) for each frame
            * np.ndarray - VGGish features; shape=(n_frames, 128)
        """

        with open(self.vggish_path, "r") as fd:
            data = json.load(fd)
            times = np.asarray(data["time_points"])
            vgg = np.asarray(data["features"])
            return times, vgg

    # -- will be fed as beat_data=[(self.beats, None)], see jams_utils), and returns a jams
    # -- object with the annotations.
    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata=self._track_metadata,
            # tag_data=None,  # FIXME
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load an OpenMIC2018 audio file.

    Audio will be resampled to 44100 Hz and downmixed to mono.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    # -- load as 44100 mono
    return librosa.load(fhandle, sr=44100, mono=True)


# -- use this decorator so the docs are complete
@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The OpenMIC-2018 dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="openmic2018",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = Path(self.data_home) / "openmic-2018-metadata.csv"

        try:
            with open(metadata_path, "r") as fdesc:
                # index column is second to last
                metadata = pd.read_csv(fdesc, index_col=-2)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Metadata file {metadata_path} not found. " "Did you run .download?"
            ) from exc

        # genres column is a json object: expand it
        # the raw CSV file is not actually valid json, so we'll fix that with a
        # regexp
        str_fixer = re.compile("(?<!\\\\)'")
        metadata["track_genres"] = metadata["track_genres"].map(
            lambda x: json.loads(str_fixer.sub('"', x)), na_action="ignore"
        )

        # Create a column for splits
        metadata["split"] = pd.Series(index=metadata.index, data="")

        # Populate each split
        for split_file in (Path(self.data_home) / "partitions").rglob("*.csv"):
            split = split_file.stem
            with open(split_file, "r") as fdesc:
                split_df = pd.read_csv(
                    fdesc,
                    header=None,
                    index_col=0,
                )
                split_df["split"] = split
                metadata.update(split_df)

        # Tack on labels
        label_path = Path(self.data_home) / "openmic-2018-aggregated-labels.csv"
        with open(label_path, "r") as fdesc:
            labels = pd.read_csv(fdesc, index_col=0)
        # Pivot the labels into its own dataframe
        labels = labels.pivot_table(
            columns="instrument", values="relevance", index=labels.index
        )

        # Join to metadata
        metadata = metadata.join(labels)

        # Tack on individual annotations?

        return metadata.to_dict(orient="index")

    @core.cached_property
    def _class_map(self):
        class_path = Path(self.data_home) / "class-map.json"

        try:
            with open(class_path, "r") as fd:
                classes = json.load(fd)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Metadata file {class_path} not found. " "Did you run .download?"
            ) from exc

        return classes
