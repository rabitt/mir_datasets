"""Four-Way Tabla Stroke Transcription and Classification Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Four-Way Tabla Dataset includes audio recordings of tabla solo with onset annotations for particular
    strokes types. This dataset was published in 2021 in the context of ISMIR2021 (Online), and may be used for
    tasks related to tabla analysis, including problems such as onset detection and stroke classification.

    Total audio samples: We do have a total of 226 samples for training and 10 for testing. Each audio has
    an approximate duration of 1 minute.

    Audio specifications:

    * Sampling frequency: 44.1 kHz
    * Bit-depth: 16 bit
    * Audio format: .wav

    Dataset usage: This dataset may be used for the data-driven research of tabla stroke transcription and
    identification. In this dataset, four important tabla characteristic strokes are considered.

    Dataset structure: The dataset is split in two subsets, containing training and testing samples. Within each
    subset, there is a folder containing the audios, and another folder containing the onset annotations. The onset
    annotations are organized in a folder per each stroke type: b, d, rb, rt. Therefore, the paths to onsets would
    look like:

    .. code-block:: bash

        train/onsets/<StrokeType>/<ID>.onsets

    The dataset is made available by CompMusic under a Creative Commons
    Attribution 3.0 Unported (CC BY 3.0) License.
"""

import csv
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@article{RohitMA2021,
    author = {M.A, Rohit and Bhattacharjee, Amitrajit and Rao, Preeti},
    journal = {Proc. of the 22nd Int. Society for Music Information Retrieval Conf., Online, 2021},
    title = {{Four-way Classification of Tabla Strokes with Models Adapted from Automatic Drum Transcription}},
    year = {2021}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="four_way_tabla_index.json"),
}

REMOTES = {
    "remote_data": download_utils.RemoteFileMetadata(
        filename="4way-tabla-ismir21-dataset.zip",
        url="https://zenodo.org/record/7110248/files/4way-tabla-ismir21-dataset.zip?download=1",
        checksum="fcddb565f260d4170877f70a7c33d69d",  # the md5 checksum
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International License."


class Track(core.Track):
    """Four-Way Tabla track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.

    Attributes:
        track_id (str): track id
        audio_path (str): audio path
        onsets_b_path (str): path to B onsets
        onsets_d_path (str): path to D onsets
        onsets_rb_path (str): path to RB onsets
        onsets_rt_path (str): path to RT onsets
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

        self.audio_path = self.get_path("audio")
        self.onsets_b_path = self.get_path("onsets_b")
        self.onsets_d_path = self.get_path("onsets_d")
        self.onsets_rb_path = self.get_path("onsets_rb")
        self.onsets_rt_path = self.get_path("onsets_rt")

        self.train = True if "train" in self.audio_path else False

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    @property
    def onsets_b(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke B.

        Returns:
            * annotations.BeatData - onsets annotation
        """
        return load_onsets(self.onsets_b_path)

    @property
    def onsets_d(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke D.

        Returns:
            * annotations.BeatData - onsets annotation
        """
        return load_onsets(self.onsets_d_path)

    @property
    def onsets_rb(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke RB.

        Returns:
            * annotations.BeatData - onsets annotation
        """
        return load_onsets(self.onsets_rb_path)

    @property
    def onsets_rt(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke RT.

        Returns:
            * annotations.BeatData - onsets annotation
        """
        return load_onsets(self.onsets_rt_path)

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[
                (self.onsets_b, "onsets_b"),
                (self.onsets_d, "onsets_d"),
                (self.onsets_rb, "onsets_rb"),
                (self.onsets_rt, "onsets_rt"),
            ],
            metadata={"train": self.train},
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Mridangam Stroke Dataset audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=44100, mono=True)


@io.coerce_to_string_io
def load_onsets(fhandle):
    """Load stroke onsets.

    Args:
        fhandle (str or file-like): Local path where the pitch annotation is stored.

    Returns:
        EventData: onset annotations
    """
    onsets = []
    reader = csv.reader(fhandle, delimiter="\n")
    for line in reader:
        onsets.append(float(line[0]))

    if not onsets:
        return None
    onsets = np.array(onsets)

    # All strokes are out of an annotated metric
    beat_position = np.zeros(onsets.shape)

    return annotations.BeatData(onsets, "s", beat_position, "global_index")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Four-Way Tabla dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="four_way_tabla",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )
