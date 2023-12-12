"""Mridangam Stroke Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Mridangam Stroke dataset is a collection of individual strokes of
    the Mridangam in various tonics. The dataset comprises of 10 different
    strokes played on Mridangams with 6 different tonic values. The audio
    examples were recorded from a professional Carnatic percussionist in a
    semi-anechoic studio conditions by Akshay Anantapadmanabhan.

    Total audio samples: 6977

    Used microphones:

    * SM-58 microphones
    * H4n ZOOM recorder.

    Audio specifications:

    * Sampling frequency: 44.1 kHz
    * Bit-depth: 16 bit
    * Audio format: .wav

    The dataset can be used for training models for each Mridangam stroke. The
    presentation of the dataset took place on the IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP 2013) on May 2013.
    You can read the full publication here: https://repositori.upf.edu/handle/10230/25756

    Mridangam Dataset is annotated by storing the informat of each track in their filenames.
    The structure of the filename is:

    .. code-block:: bash

        <TrackID>__<AuthorName>__<StrokeName>-<Tonic>-<InstanceNum>.wav

    The dataset is made available by CompMusic under a Creative Commons
    Attribution 3.0 Unported (CC BY 3.0) License.

    For more details, please visit: https://compmusic.upf.edu/mridangam-stroke-dataset
"""

import os
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """@article{Anantapadmanabhan2013,
    author = {Anantapadmanabhan, Akshay and Bellur, Ashwin and Murthy, Hema A.},
    doi = {10.1109/ICASSP.2013.6637633},
    isbn = {9781479903566},
    issn = {15206149},
    journal = {ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings},
    keywords = {Hidden Markov models, Modal Analysis, Mridangam, Non-negative Matrix Factorization,
    automatic transcription},
    pages = {181--185},
    title = {{Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization}},
    year = {2013}
}"""

INDEXES = {
    "default": "1.5",
    "test": "1.5",
    "1.5": core.Index(filename="mridangam_stroke_index_1.5.json"),
}

REMOTES = {
    "remote_data": download_utils.RemoteFileMetadata(
        filename="mridangam_stroke_1.5.zip",
        url="https://zenodo.org/record/4068196/files/mridangam_stroke_1.5.zip?download=1",
        checksum="39af55b2476b94c7946bec24331ec01a",  # the md5 checksum
    )
}


STROKE_DICT = {
    "bheem",
    "cha",
    "dheem",
    "dhin",
    "num",
    "ta",
    "tha",
    "tham",
    "thi",
    "thom",
}


TONIC_DICT = {"B", "C", "C#", "D", "D#", "E"}

LICENSE_INFO = "Creative Commons Attribution 3.0 Unported (CC BY 3.0) License."


class Track(core.Track):
    """Mridangam Stroke track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.

    Attributes:
        track_id (str): track id
        audio_path (str): audio path
        stroke_name (str): name of the Mridangam stroke present in Track
        tonic (str): tonic of the stroke in the Track
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

        # Parse stroke name annotation from audio file name
        self.stroke_name = self.audio_path.split("__")[2].split("-")[0]
        assert self.stroke_name in STROKE_DICT, "Stroke {} not in stroke dictionary".format(
            self.stroke_name
        )

        # Parse tonic annotation from audio file name
        self.tonic = os.path.basename(os.path.dirname(self.audio_path))
        assert self.tonic in TONIC_DICT, "Tonic {} not in tonic dictionary".format(self.tonic)

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
            audio_path=self.audio_path,
            tags_open_data=[(self.stroke_name, "stroke_name")],
            metadata={"tonic": self.tonic},
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


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The mridangam_stroke dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="mridangam_stroke",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.mridangam_stroke.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
