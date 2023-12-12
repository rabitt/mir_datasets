"""TinySOL Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    TinySOL is a dataset of 2913 samples, each containing a single musical note from one of 14
    different instruments:

    - Bass Tuba
    - French Horn
    - Trombone
    - Trumpet in C
    - Accordion
    - Contrabass
    - Violin
    - Viola
    - Violoncello
    - Bassoon
    - Clarinet in B-flat
    - Flute
    - Oboe
    - Alto Saxophone

    These sounds were originally recorded at Ircam in Paris (France) between 1996
    and 1999, as part of a larger project named Studio On Line (SOL). Although SOL
    contains many combinations of mutes and extended playing techniques, TinySOL
    purely consists of sounds played in the so-called "ordinary" style, and in
    absence of mute.

    TinySOL can be used for education and research purposes. In particular, it can
    be employed as a dataset for training and/or evaluating music information
    retrieval (MIR) systems, for tasks such as instrument recognition or
    fundamental frequency estimation. For this purpose, we provide an official 5-fold
    split of TinySOL as a metadata attribute. This split has been carefully balanced
    in terms of instrumentation, pitch range, and dynamics. For the sake of research
    reproducibility, we encourage users of TinySOL to adopt this split and report
    their results in terms of average performance across folds.

    We encourage TinySOL users to subscribe to the Ircam Forum so that they can
    have access to larger versions of SOL.

    For more details, please visit: https://www.orch-idea.org/
"""

import csv
import os
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{cella2020preprint,
  author={Cella, Carmine Emanuele and Ghisi, Daniele and Lostanlen, Vincent and
  Lévy, Fabien and Fineberg, Joshua and Maresz, Yan},
  title={{OrchideaSOL}: {A} dataset of extended
  instrumental techniques for computer-aided orchestration},
  bootktitle={Under review},
  year={2020}
}"""

INDEXES = {
    "default": "6.0",
    "test": "6.0",
    "6.0": core.Index(filename="tinysol_index_6.0.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="TinySOL.tar.gz",
        url="https://zenodo.org/record/3685367/files/TinySOL.tar.gz?download=1",
        checksum="36030a7fe389da86c3419e5ee48e3b7f",
        destination_dir="audio",
    ),
    "annotations": download_utils.RemoteFileMetadata(
        filename="TinySOL_metadata.csv",
        url="https://zenodo.org/record/3685367/files/TinySOL_metadata.csv?download=1",
        checksum="a86c9bb115f69e61f2f25872e397fc4a",
        destination_dir="annotation",
    ),
}

STRING_ROMAN_NUMERALS = {1: "I", 2: "II", 3: "III", 4: "IV"}

LICENSE_INFO = "Creative Commons Attribution 4.0 International Public License."


class Track(core.Track):
    """Tinysol Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path of the audio file
        dynamics (str): dynamics abbreviation. Ex: pp, mf, ff, etc.
        dynamics_id (int): pp=0, p=1, mf=2, f=3, ff=4
        family (str): instrument family encoded by its English name
        instance_id (int): instance ID. Either equal to 0, 1, 2, or 3.
        instrument_abbr (str): instrument abbreviation
        instrument_full (str): instrument encoded by its English name
        is_resampled (bool): True if this sample was pitch-shifted from a neighbor; False if it was genuinely recorded.
        pitch (str): string containing English pitch class and octave number
        pitch_id (int): MIDI note index, where middle C ("C4") corresponds to 60
        string_id (NoneType): string ID. By musical convention, the first
            string is the highest. On wind instruments, this is replaced by `None`.
        technique_abbr (str): playing technique abbreviation
        technique_full (str): playing technique encoded by its English name
        track_id (str): track id
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

    @property
    def split(self):
        return self._track_metadata.get("Fold")

    @property
    def family(self):
        return self._track_metadata.get("Family")

    @property
    def instrument_abbr(self):
        return self._track_metadata.get("Instrument (abbr.)")

    @property
    def instrument_full(self):
        return self._track_metadata.get("Instrument (in full)")

    @property
    def technique_abbr(self):
        return self._track_metadata.get("Technique (abbr.)")

    @property
    def technique_full(self):
        return self._track_metadata.get("Technique (in full)")

    @property
    def pitch(self):
        return self._track_metadata.get("Pitch")

    @property
    def pitch_id(self):
        return self._track_metadata.get("Pitch ID")

    @property
    def dynamics(self):
        return self._track_metadata.get("Dynamics")

    @property
    def dynamics_id(self):
        return self._track_metadata.get("Dynamics ID")

    @property
    def instance_id(self):
        return self._track_metadata.get("Instance ID")

    @property
    def string_id(self):
        return self._track_metadata.get("String ID")

    @property
    def is_resampled(self):
        return self._track_metadata.get("Resampled")

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
        return jams_utils.jams_converter(audio_path=self.audio_path, metadata=self._track_metadata)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a TinySOL audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The tinysol dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="tinysol",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "annotation", "TinySOL_metadata.csv")

        metadata_index = {}
        try:
            with open(metadata_path, "r") as fhandle:
                csv_reader = csv.reader(fhandle, delimiter=",")
                next(csv_reader)
                for row in csv_reader:
                    key = os.path.splitext(os.path.split(row[0])[1])[0]
                    metadata_index[key] = {
                        "Fold": int(row[1]),
                        "Family": row[2],
                        "Instrument (abbr.)": row[3],
                        "Instrument (in full)": row[4],
                        "Technique (abbr.)": row[5],
                        "Technique (in full)": row[6],
                        "Pitch": row[7],
                        "Pitch ID": int(row[8]),
                        "Dynamics": row[9],
                        "Dynamics ID": int(row[10]),
                        "Instance ID": int(row[11]),
                        "Resampled": (row[13] == "TRUE"),
                    }
                    if len(row[12]) > 0:
                        metadata_index[key]["String ID"] = int(float(row[12]))
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.tinysol.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
