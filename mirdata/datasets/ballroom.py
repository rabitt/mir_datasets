"""Ballroom Rhythm Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Ballroom Rhythm Dataset is a comprehensive collection of rhythm annotations for ballroom dance music. This dataset is designed for tasks such as beat tracking, rhythm analysis, and tempo estimation in ballroom dance music. It includes annotations for beats and bars corresponding to different dance styles within the ballroom genre.

    **Dataset Overview:**

    The dataset offers beat and bar annotations for various ballroom dance styles, such as Waltz, Tango, Viennese Waltz, Slow Foxtrot, Quickstep, Samba, Cha-Cha-Cha, Rumba, Paso Doble, and Jive. These annotations are provided in a format that includes beat time in seconds and beat ID, facilitating precise rhythm analysis.

    **Beat and Bar Annotations:**

    The beat annotations are structured as `.beats` files, where each line represents a beat with its timestamp and beat ID. For example, a line `9.430022675 3` indicates that the third beat of a bar is located at 9.43 seconds. This format is particularly useful for identifying downbeats, as they correspond to beats with ID = 1.

    **Annotation Methodology:**

    The dataset's annotations are based on the tempo guidelines of each ballroom dance style. Initial annotations were generated using a beat tracker, and then manually adjusted for accuracy. This method ensures that the annotations reflect the characteristic rhythms of each dance style.

    **Applications:**

    The Ballroom Rhythm Dataset is ideal for developing and testing algorithms for beat tracking, tempo estimation, and rhythm analysis in ballroom dance music. It can also be used for educational purposes, offering insights into the rhythmic structures of various ballroom dance styles.

    **Acknowledgments and References:**

    This dataset was created with the collaboration of experts in ballroom dance music. We extend our gratitude to those who contributed their knowledge and expertise to this project. For detailed information on the dataset and its creation, please refer to the associated research papers and documentation.

"""

import os
import csv
import logging
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, io, jams_utils


BIBTEX = """
@ARTICLE{1678001,
    author={Gouyon, F. and Klapuri, A. and Dixon, S. and Alonso, M. and Tzanetakis, G. and Uhle, C. and Cano, P.},
    journal={IEEE Transactions on Audio, Speech, and Language Processing}, 
    title={An experimental comparison of audio tempo induction algorithms}, 
    year={2006},
    volume={14},
    number={5},
    pages={1832-1844},
    doi={10.1109/TSA.2005.858509}}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="ballroom_full_index_1.0.json"),
}

REMOTES = None

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """The files of this dataset are shared under request. Please go to: https://zenodo.org/record/1264742 and request access, stating
    the research-related use you will give to the dataset. Once the access is granted (it may take, at most, one day or two), please download 
    the dataset with the provided Zenodo link and uncompress and store the datasets to a desired location, and use such location to initialize the 
    dataset as follows: compmusic_hindustani_rhythm = mirdata.initialize("compmusic_hindustani_rhythm", data_home="/path/to/home/folder/of/dataset").
    """


class Track(core.Track):
    """Ballroom Rhythm class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        beats_path (srt): path to beats file
        tempo_path (srt): path to tempo file

    """

    def __init__(
        self,
        track_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        # Audio path
        self.audio_path = self.get_path("audio")

        # Annotations paths
        self.beats_path = self.get_path("beats")
        self.tempo_path = self.get_path("tempo")

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def tempo(self) -> Optional[float]:
        return load_tempo(self.tempo_path)

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
            beat_data=[(self.beats, "beats")],
            tempo_data=[(self.tempo, "tempo")],
            metadata=None,
        )


def load_audio(audio_path):
    """Load an audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None
    return librosa.load(audio_path, sr=44100, mono=False)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO):
    """Load beats

    Args:
        fhandle (str or file-like): Local path where the beats annotation is stored.

    Returns:
        BeatData: beat annotations

    """
    beat_times = []
    beat_positions = []

    reader = csv.reader(fhandle, delimiter=" ")
    for line in reader:
        beat_times.append(float(line[0]))
        beat_positions.append(int(line[1]))

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(
        np.array(beat_times), "s", np.array(beat_positions), "bar_index"
    )


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> float:
    """Load tempo

    Args:
        fhandle (str or file-like): Local path where the tempo annotation is stored.

    Returns:
        float: tempo annotation

    """
    reader = csv.reader(fhandle, delimiter=",")
    return float(next(reader)[0])


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The ballroom dataset

    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="ballroom",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
            download_info=DOWNLOAD_INFO,
        )
