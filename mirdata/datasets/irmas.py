"""IRMAS Loader.

.. admonition:: Dataset Info
    :class: dropdown

    IRMAS: a dataset for instrument recognition in musical audio signals

    This dataset includes musical audio excerpts with annotations of the predominant instrument(s) present.
    It was used for the evaluation in the following article:

    .. code-block:: latex

        Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. “A Comparison of Sound Segregation Techniques for
        Predominant Instrument Recognition in Musical Audio Signals”, in Proc. ISMIR (pp. 559-564), 2012.

    IRMAS is intended to be used for training and testing methods for the automatic recognition of predominant
    instruments in musical audio. The instruments considered are: cello, clarinet, flute, acoustic guitar,
    electric guitar, organ, piano, saxophone, trumpet, violin, and human singing voice.
    This dataset is derived from the one compiled by Ferdinand Fuhrmann in his PhD thesis, with the difference
    that we provide audio data in stereo format, the annotations in the testing dataset are limited to specific
    pitched instruments, and there is a different amount and lenght of excerpts from the original dataset.


    The dataset is split into training and test data.

    **Training data**

    Total audio samples: 6705
    They are excerpts of 3 seconds from more than 2000 distinct recordings.

    Audio specifications

    * Sampling frequency: 44.1 kHz
    * Bit-depth: 16 bit
    * Audio format: .wav

    IRMAS Dataset trainig samples are annotated by storing the information of each track in their filenames.

    * Predominant instrument:

        * The annotation of the predominant instrument of each excerpt is both in the name of the containing
          folder, and in the file name: cello (cel), clarinet (cla), flute (flu), acoustic guitar (gac),
          electric guitar (gel), organ (org), piano (pia), saxophone (sax), trumpet (tru), violin (vio),
          and human singing voice (voi).
        * The number of files per instrument are: cel(388), cla(505), flu(451), gac(637), gel(760), org(682),
          pia(721), sax(626), tru(577), vio(580), voi(778).

    * Drum presence

        * Additionally, some of the files have annotation in the filename regarding the presence ([dru])
          or non presence([nod]) of drums.

    * The annotation of the musical genre:

        * country-folk ([cou_fol])
        * classical ([cla]),
        * pop-rock ([pop_roc])
        * latin-soul ([lat_sou])
        * jazz-blues ([jaz_blu]).

    **Testing data**

    Total audio samples: 2874

    Audio specifications

    * Sampling frequency: 44.1 kHz
    * Bit-depth: 16 bit
    * Audio format: .wav

    IRMAS Dataset testing samples are annotated by the following basis:

    * Predominant instrument:

        The annotations for an excerpt named: “excerptName.wav” are given in “excerptName.txt”. More than one
        instrument may be annotated in each excerpt, one label per line. This part of the dataset contains excerpts
        from a diversity of western musical genres, with varied instrumentations, and it is derived from the original
        testing dataset from Fuhrmann (http://www.dtic.upf.edu/~ffuhrmann/PhD/).
        Instrument nomenclatures are the same as the training dataset.

    Dataset compiled by Juan J. Bosch, Ferdinand Fuhrmann, Perfecto Herrera,
    Music Technology Group - Universitat Pompeu Fabra (Barcelona).

    The IRMAS dataset is offered free of charge for non-commercial use only. You can not redistribute it nor modify it.
    This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

    For more details, please visit: https://www.upf.edu/web/mtg/irmas
"""
import csv
import os
from typing import BinaryIO, List, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """
@dataset{juan_j_bosch_2014_1290750,
  author       = {Juan J. Bosch and Ferdinand Fuhrmann and Perfecto Herrera},
  title        = {{IRMAS: a dataset for instrument recognition in musical audio signals}},
  month        = sep,
  year         = 2014,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1290750},
  url          = {https://doi.org/10.5281/zenodo.1290750}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="irmas_index_1.0.json"),
}

REMOTES = {
    "training_data": download_utils.RemoteFileMetadata(
        filename="IRMAS-TrainingData.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1",
        checksum="4fd9f5ed5a18d8e2687e6360b5f60afe",
    ),
    "testing_data_1": download_utils.RemoteFileMetadata(
        filename="IRMAS-TestingData-Part1.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip?download=1",
        checksum="5a2e65520dcedada565dff2050bb2a56",
    ),
    "testing_data_2": download_utils.RemoteFileMetadata(
        filename="IRMAS-TestingData-Part2.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip?download=1",
        checksum="afb0c8ea92f34ee653693106be95c895",
    ),
    "testing_data_3": download_utils.RemoteFileMetadata(
        filename="IRMAS-TestingData-Part3.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip?download=1",
        checksum="9b3fb2d0c89cdc98037121c25bd5b556",
    ),
}


INST_DICT = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]

GENRE_DICT = ["cou_fol", "cla", "pop_roc", "lat_sou", "jaz_blu"]

LICENSE_INFO = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License."


class Track(core.Track):
    """IRMAS track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Mridangam-Stroke`

    Attributes:
        track_id (str): track id
        predominant_instrument (list): Training tracks predominant instrument
        train (bool): flag to identify if the track is from the training of the testing dataset
        genre (str): string containing the namecode of the genre of the track.
        drum (bool): flag to identify if the track contains drums or not.
        split (str): data split ("train" or "test")

    Cached Properties:
        instrument (list): list of predominant instruments as str
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.annotation_path = self.get_path("annotation")

        self.audio_path = self.get_path("audio")
        self._audio_filename = self._track_paths["audio"][0]

        self.split = "train" if "__" in track_id else "test"

        # Dataset attributes
        self.predominant_instrument = (
            os.path.basename(os.path.dirname(self.audio_path)) if self.split == "train" else None
        )
        if self.split == "train" and (
            "dru" in self._audio_filename or "nod" in self._audio_filename
        ):
            self.genre = self._audio_filename.split(".")[0].split("[")[3].split("]")[0]
            self.drum = [True if "dru" in self._audio_filename else False][0]
        elif self.split == "train" and not (
            "dru" in self._audio_filename or "nod" in self._audio_filename
        ):
            self.genre = self._audio_filename.split(".")[0].split("[")[2].split("]")[0]
            self.drum = None
        else:
            self.genre = None
            self.drum = None

        self.train = True if self.split == "train" else False

    @core.cached_property
    def instrument(self):
        if self.predominant_instrument is not None:
            return [self.predominant_instrument]

        return load_pred_inst(self.annotation_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio signal.

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file
        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """The track's data in jams format.

        Returns:
            jams.JAMS: return track data in jam format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "instrument": self.instrument,
                "genre": self.genre,
                "drum": self.drum,
                "train": self.train,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a IRMAS dataset audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=44100, mono=False)


@io.coerce_to_string_io
def load_pred_inst(fhandle: TextIO) -> List[str]:
    """Load predominant instrument of track.

    Args:
        fhandle (str or file-like): File-like object or path where the test annotations are stored.

    Returns:
        list(str): test track predominant instrument(s) annotations
    """
    pred_inst = []
    reader = csv.reader(fhandle, delimiter=" ")
    for line in reader:
        inst_code = line[0][:3]
        assert inst_code in INST_DICT, "Instrument {} not in instrument dictionary".format(
            inst_code
        )
        pred_inst.append(inst_code)

    return pred_inst


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The irmas dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="irmas",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.irmas.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.irmas.load_pred_inst", version="0.3.4")
    def load_pred_inst(self, *args, **kwargs):
        return load_pred_inst(*args, **kwargs)
