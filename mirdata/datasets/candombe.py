"""Candombe Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    This is a dataset of Candombe recordings with annotated beats and downbeats, totaling over 2 hours of audio.
    It comprises 35 complete performances by renowned players, in groups of three to five drums.
    Recording sessions were conducted in studio, in the context of musicological research over the past two decades.
    A total of 26 tambor players took part, belonging to different generations and representing all the important traditional Candombe styles.
    The audio files are stereo with a sampling rate of 44.1 kHz and 16-bit precision.
    The location of beats and downbeats was annotated by an expert, adding to more than 4700 downbeats.

    The audio is provided as .flac files and the annotations as .csv files.
    The values in the first column of the csv file are the time instants of the beats.
    The numbers on the second column indicate both the bar number and the beat number within the bar.
    For instance, 1.1, 1.2, 1.3 and 1.4 are the four beats of the first bar. Hence, each label ending with .1 indicates a downbeat.
    Another set of annotations are provided as .beats files in which the bar numbers are removed.
"""
import csv
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@inproceedings{Nunes2015,
    author = {Leonardo Nunes and Martín Rocamora and Luis Jure and Luiz W. P. Biscainho},
    title = {{Beat and Downbeat Tracking Based on Rhythmic Patterns Applied to the Uruguayan Candombe Drumming}},
    booktitle = {Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR 2015)},
    month = {Oct.},
    address = {Málaga, Spain},
    pages = {264--270},
    year = {2015}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="candombe_index_1.0.json"),
}


REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="candombe_annotations.zip",
        url="https://zenodo.org/record/6533068/files/candombe_annotations.zip",
        checksum="f78aff60aa413cb4960c0c77cc31c243",
        destination_dir=None,
    ),
    "audio": download_utils.RemoteFileMetadata(
        filename="candombe_audio.zip",
        url="https://zenodo.org/record/6533068/files/candombe_audio.zip",
        checksum="ccd7f437024807b1a52c0818aa0b7f06",
        destination_dir=None,
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Track(core.Track):
    """Candombe Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to audio file
        beats_path (str): path to beats file

    Cached Properties:
        beats (BeatData): beat annotations
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.beats_path = self.get_path("beats")

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        """The track's beats.

        Returns:
            BeatData: loaded beat data
        """
        return load_beats(self.beats_path)

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
            audio_path=self.audio_path, beat_data=[(self.beats, None)], metadata=None
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a candombe audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load a candombe beats file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        BeatData: loaded beat data
    """
    reader = csv.reader(fhandle, delimiter=",")
    times = []
    beats = []
    for line in reader:
        times.append(float(line[0]))
        beats.append(int(line[1].split(".")[1]))

    beat_data = annotations.BeatData(
        times=np.array(times),
        time_unit="s",
        positions=np.array(beats),
        position_unit="bar_index",
    )
    return beat_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The candombe dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="candombe",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )
