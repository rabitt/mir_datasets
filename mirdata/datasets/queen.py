"""Queen Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Queen Dataset includes chord, key, and segmentation
    annotations for 51 Queen songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf and
    http://isophonics.net/content/reference-annotations-queen.

    The CDs used in this dataset are:
    Queen: Greatest Hits I, Parlophone, 0777 7 8950424
    Queen: Greatest Hits II, Parlophone, CDP 7979712
    Queen: Greatest Hits III, Parlophone, 7243 52389421

    In the progress of labelling the chords, C4DM researchers used the following literature to verify their judgements:

    Queen, Greatest Hits I, International Music Publications Ltd, London, ISBN 0-571-52828-7

    Queen, Greatest Hits II, Queen Music Ltd./EMI Music Publishing (Barnes Music Engraving), ISBN 0-86175-465-4

    Acknowledgements
    We'd like to thank our student annotators:

    Eric Gyingy
    Diako Rasoul
    Felix Stiller
    Helena du Toit
    Vinh Ton
    Chuks Chiejine
"""

import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{mauch2009beatles,
    title={OMRAS2 metadata project 2009},
    author={Mauch, Matthias and Cannam, Chris and Davies, Matthew and Dixon, Simon and Harte,
    Christopher and Kolozali, Sefki and Tidhar, Dan and Sandler, Mark},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2009},
    series = {ISMIR}
}"""
LICENSE_INFO = "Unfortunately we couldn't find the license information for Queen dataset."

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="queen_index_1.0.json"),
}

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="Queen Annotations.tar.gz",
        url="http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz",
        checksum="fe11217d32bc222ae418425441974046",
        destination_dir="annotations",
    )
}


DOWNLOAD_INFO = """
        Unfortunately the audio files of Queen dataset are not available
        for download. If you have Queen dataset, place the contents into
        a folder called Queen with the following structure:
            > Queen/
                > annotations/
                > audio/
        and copy Queen folder to {}
"""


class Track(core.Track):
    """Queen track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        chords_path (str): chord annotation path
        keys_path (str): key annotation path
        sections_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

    Cached Properties:
        chords (ChordData): human-labeled chord annotations
        key (KeyData): local key annotations
        sections (SectionData): section annotations
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.chords_path = self.get_path("chords")
        self.keys_path = self.get_path("keys")
        self.sections_path = self.get_path("sections")
        self.audio_path = self.get_path("audio")

        self.title = os.path.basename(self.sections_path).split(".")[0]

    @core.cached_property
    def chords(self) -> Optional[annotations.ChordData]:
        return load_chords(self.chords_path)

    @core.cached_property
    def key(self) -> Optional[annotations.KeyData]:
        return load_key(self.keys_path)

    @core.cached_property
    def sections(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """The track's data in jams format.

        Returns:
            jams.JAMS: return track data in jam format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            section_data=[(self.sections, None)],
            chord_data=[(self.chords, None)],
            key_data=[(self.key, None)],
            metadata={"artist": "Queen", "title": self.title},
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Queen audio file.

    Args:
        fhandle (str): path to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=44100, mono=True)


@io.coerce_to_string_io
def load_chords(fhandle: TextIO) -> annotations.ChordData:
    """Load Queen format chord data from a file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a chord file

    Returns:
        (ChordData): loaded chord data
    """
    start_times, end_times, chords = [], [], []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        chords.append(line[2])

    return annotations.ChordData(np.array([start_times, end_times]).T, "s", chords, "harte")


@io.coerce_to_string_io
def load_key(fhandle: TextIO) -> annotations.KeyData:
    """Load Queen format key data from a file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a key file

    Returns:
        (KeyData): loaded key data
    """
    start_times, end_times, keys = [], [], []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        if line[2] == "Key":
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            keys.append(line[3])

    return annotations.KeyData(np.array([start_times, end_times]).T, "s", keys, "key_mode")


@io.coerce_to_string_io
def load_sections(fhandle: TextIO) -> annotations.SectionData:
    """Load Queen format section data from a file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a section file

    Returns:
        (SectionData): loaded section data
    """
    start_times, end_times, sections = [], [], []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        sections.append(line[3])

    return annotations.SectionData(np.array([start_times, end_times]).T, "s", sections, "open")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """Queen dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="queen",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.queen.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.queen.load_key", version="0.3.4")
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.queen.load_chords", version="0.3.4")
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.queen.load_sections", version="0.3.4")
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)
