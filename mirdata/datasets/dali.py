"""DALI Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    DALI contains 5358 audio files with their time-aligned vocal melody.
    It also contains time-aligned lyrics at four levels of granularity: notes,
    words, lines, and paragraphs.

    For each song, DALI also provides additional metadata: genre, language, musician,
    album covers, or links to video clips.

    For more details, please visit: https://github.com/gabolsgabs/DALI

"""

import json
import gzip
import logging
import os
import pickle
from typing import BinaryIO, Optional, Tuple

from deprecated.sphinx import deprecated
import librosa
import numpy as np
from smart_open import open

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations
from mirdata import io

# this is the package, needed to load the annotations.
# DALI-dataset is only installed if the user explicitly declares
# they want dali when pip installing.
try:
    import DALI
except ImportError as E:
    raise ImportError(
        "An error occured when importing this dataset. Most likely this is due to a dependency not being installed, in this case DALI."
    )

BIBTEX = """@inproceedings{Meseguer-Brocal_2018,
    Title = {DALI: a large Dataset of synchronized Audio, LyrIcs and notes, automatically created using teacher-student
     machine learning paradigm.},
    Author = {Meseguer-Brocal, Gabriel and Cohen-Hadria, Alice and Peeters, Geoffroy},
    Booktitle = {19th International Society for Music Information Retrieval Conference},
    Editor = {ISMIR}, Month = {September},
    Year = {2018}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="dali_index_1.0.json"),
}

REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="dali_metadata.json",
        url="https://raw.githubusercontent.com/gabolsgabs/DALI/master/code/DALI/files/dali_v1_metadata.json",
        checksum="40af5059e7aa97f81b2654758094d24b",
        destination_dir=".",
    )
}
DOWNLOAD_INFO = """
    To download this dataset, visit:
    https://zenodo.org/record/2577915 and request access.
    Once downloaded, unzip the file DALI_v1.0.zip
    and place the result in:
    {}

    Use the function dali_code.get_audio you can find at:
    https://github.com/gabolsgabs/DALI for getting the audio
    and place them in "audio" folder with the following structure:
    > Dali
        > audio
        ...
"""

LICENSE_INFO = (
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License."
)


class Track(core.Track):
    """DALI melody Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        album (str): the track's album
        annotation_path (str): path to the track's annotation file
        artist (str): the track's artist
        audio_path (str): path to the track's audio file
        audio_url (str): youtube ID
        dataset_version (int): dataset annotation version
        ground_truth (bool): True if the annotation is verified
        language (str): sung language
        release_date (str): year the track was released
        scores_manual (int): manual score annotations
        scores_ncc (float): ncc score annotations
        title (str): the track's title
        track_id (str): the unique track id
        url_working (bool): True if the youtube url was valid

    Cached Properties:
        notes (NoteData): vocal notes
        words (LyricData): word-level lyrics
        lines (LyricData): line-level lyrics
        paragraphs (LyricData): paragraph-level lyrics
        annotation-object (DALI.Annotations): DALI annotation object

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.annotation_path = self.get_path("annot")

        self.audio_path = self.get_path("audio")

    @property
    def audio_url(self):
        return self._track_metadata.get("audio", {}).get("url")

    @property
    def url_working(self):
        return self._track_metadata.get("audio", {}).get("working")

    @property
    def ground_truth(self):
        return self._track_metadata.get("ground-truth")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def dataset_version(self):
        return self._track_metadata.get("dataset_version")

    @property
    def scores_ncc(self):
        return self._track_metadata.get("scores", {}).get("NCC")

    @property
    def scores_manual(self):
        return self._track_metadata.get("scores", {}).get("manual")

    @property
    def album(self):
        return self._track_metadata.get("metadata", {}).get("album")

    @property
    def release_date(self):
        return self._track_metadata.get("metadata", {}).get("release_date")

    @property
    def genres(self):
        return self._track_metadata.get("metadata", {}).get("genres")

    @property
    def language(self):
        return self._track_metadata.get("metadata", {}).get("language")

    @core.cached_property
    def notes(self) -> annotations.NoteData:
        return load_annotations_granularity(self.annotation_path, "notes")

    @core.cached_property
    def words(self) -> annotations.NoteData:
        return load_annotations_granularity(self.annotation_path, "words")

    @core.cached_property
    def lines(self) -> annotations.NoteData:
        return load_annotations_granularity(self.annotation_path, "lines")

    @core.cached_property
    def paragraphs(self) -> annotations.NoteData:
        return load_annotations_granularity(self.annotation_path, "paragraphs")

    @core.cached_property
    def annotation_object(self) -> DALI.Annotations:
        return load_annotations_class(self.annotation_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            lyrics_data=[
                (self.words, "word-aligned lyrics"),
                (self.lines, "line-aligned lyrics"),
                (self.paragraphs, "paragraph-aligned lyrics"),
            ],
            note_data=[(self.notes, "annotated vocal notes")],
            metadata=self._track_metadata,
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Optional[Tuple[np.ndarray, float]]:
    """Load a DALI audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


def load_annotations_granularity(annotations_path, granularity):
    """Load annotations at the specified level of granularity

    Args:
        annotations_path (str): path to a DALI annotation file
        granularity (str): one of 'notes', 'words', 'lines', 'paragraphs'

    Returns:
        NoteData for granularity='notes' or LyricData otherwise

    """
    try:
        with gzip.open(annotations_path, "rb") as f:
            output = pickle.load(f)
    except Exception as e:
        with gzip.open(annotations_path, "r") as f:
            output = pickle.load(f)
    text = []
    notes = []
    begs = []
    ends = []
    for annot in output.annotations["annot"][granularity]:
        notes.append(round(annot["freq"][0], 3))
        begs.append(round(annot["time"][0], 3))
        ends.append(round(annot["time"][1], 3))
        text.append(annot["text"])
    if granularity == "notes":
        annotation = annotations.NoteData(
            np.array([begs, ends]).T, "s", np.array(notes), "hz"
        )
    else:
        annotation = annotations.LyricData(np.array([begs, ends]).T, "s", text, "words")
    return annotation


def load_annotations_class(annotations_path):
    """Load full annotations into the DALI class object

    Args:
        annotations_path (str): path to a DALI annotation file

    Returns:
        DALI.annotations: DALI annotations object

    """
    try:
        with gzip.open(annotations_path, "rb") as f:
            output = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "annotations_path {} does not exist".format(annotations_path)
        )
    except Exception as e:
        with gzip.open(annotations_path, "r") as f:
            output = pickle.load(f)
    return output


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The dali dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="dali",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, os.path.join("dali_metadata.json"))

        try:
            with open(metadata_path, "r") as fhandle:
                metadata_index = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.dali.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.dali.load_annotations_granularity", version="0.3.4"
    )
    def load_annotations_granularity(self, *args, **kwargs):
        return load_annotations_granularity(*args, **kwargs)

    @deprecated(
        reason="Use mirdata.datasets.dali.load_annotations_class", version="0.3.4"
    )
    def load_annotations_class(self, *args, **kwargs):
        return load_annotations_class(*args, **kwargs)
