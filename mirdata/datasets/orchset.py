"""ORCHSET Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Orchset is intended to be used as a dataset for the development and
    evaluation of melody extraction algorithms. This collection contains
    64 audio excerpts focused on symphonic music with their corresponding
    annotation of the melody.

    For more details, please visit: https://zenodo.org/record/1289786#.XREpzaeZPx6
"""

import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@article{bosch2016evaluation,
    title={Evaluation and combination of pitch estimation methods for melody extraction in symphonic classical music},
    author={Bosch, Juan J and Marxer, Ricard and G{\'o}mez, Emilia},
    journal={Journal of New Music Research},
    volume={45},
    number={2},
    pages={101--117},
    year={2016},
    publisher={Taylor \\& Francis}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="orchset_index_1.0.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="Orchset_dataset_0.zip",
        url="https://zenodo.org/record/1289786/files/Orchset_dataset_0.zip?download=1",
        checksum="cf6fe52d64624f61ee116c752fb318ca",
        unpack_directories=["Orchset"],
    )
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial Share Alike 4.0 International."


class Track(core.Track):
    """Orchset Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        alternating_melody (bool): True if the melody alternates between instruments
        audio_path_mono (str): path to the mono audio file
        audio_path_stereo (str): path to the stereo audio file
        composer (str): the work's composer
        contains_brass (bool): True if the track contains any brass instrument
        contains_strings (bool): True if the track contains any string instrument
        contains_winds (bool): True if the track contains any wind instrument
        excerpt (str): True if the track is an excerpt
        melody_path (str): path to the melody annotation file
        only_brass (bool): True if the track contains brass instruments only
        only_strings (bool): True if the track contains string instruments only
        only_winds (bool): True if the track contains wind instruments only
        predominant_melodic_instruments (list): List of instruments which play the melody
        track_id (str): track id
        work (str): The musical work

    Cached Properties:
        melody (F0Data): melody annotation
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.melody_path = self.get_path("melody")

        self.audio_path_mono = self.get_path("audio_mono")
        self.audio_path_stereo = self.get_path("audio_stereo")

    @property
    def composer(self):
        return self._track_metadata.get("composer")

    @property
    def work(self):
        return self._track_metadata.get("work")

    @property
    def excerpt(self):
        return self._track_metadata.get("excerpt")

    @property
    def predominant_melodic_instruments(self):
        return self._track_metadata.get("predominant_melodic_instruments-normalized")

    @property
    def alternating_melody(self):
        return self._track_metadata.get("alternating_melody")

    @property
    def contains_winds(self):
        return self._track_metadata.get("contains_winds")

    @property
    def contains_strings(self):
        return self._track_metadata.get("contains_strings")

    @property
    def contains_brass(self):
        return self._track_metadata.get("contains_brass")

    @property
    def only_strings(self):
        return self._track_metadata.get("only_strings")

    @property
    def only_winds(self):
        return self._track_metadata.get("only_winds")

    @property
    def only_brass(self):
        return self._track_metadata.get("only_brass")

    @core.cached_property
    def melody(self) -> Optional[annotations.F0Data]:
        return load_melody(self.melody_path)

    @property
    def audio_mono(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio (mono)

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file
        """
        return load_audio_mono(self.audio_path_mono)

    @property
    def audio_stereo(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio (stereo)

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file
        """
        return load_audio_stereo(self.audio_path_stereo)

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path_mono,
            f0_data=[(self.melody, "annotated melody")],
            metadata=self._track_metadata,
        )


@io.coerce_to_bytes_io
def load_audio_mono(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load an Orchset audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_bytes_io
def load_audio_stereo(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load an Orchset audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the stereo audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=False)


@io.coerce_to_string_io
def load_melody(fhandle: TextIO) -> annotations.F0Data:
    """Load an Orchset melody annotation file.

    Args:
        fhandle (str or file-like): File-like object or path to melody annotation file

    Raises:
        IOError: if melody_path doesn't exist

    Returns:
        F0Data: melody annotation data
    """

    times = []
    freqs = []
    voicing = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        times.append(float(line[0]))
        freqs.append(float(line[1]))
        voicing.append(0.0 if line[1] == "0" else 1.0)

    melody_data = annotations.F0Data(
        np.array(times), "s", np.array(freqs), "hz", np.array(voicing), "binary"
    )
    return melody_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The orchset dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="orchset",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        predominant_inst_path = os.path.join(
            self.data_home, "Orchset - Predominant Melodic Instruments.csv"
        )

        try:
            with open(predominant_inst_path, "r") as fhandle:
                reader = csv.reader(fhandle, delimiter=",")
                raw_data = []
                for line in reader:
                    if line[0] == "excerpt":
                        continue
                    raw_data.append(line)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        tf_dict = {"TRUE": True, "FALSE": False}

        metadata_index = {}
        for line in raw_data:
            track_id = line[0].split(".")[0]

            id_split = track_id.split(".")[0].split("-")
            if id_split[0] == "Musorgski" or id_split[0] == "Rimski":
                id_split[0] = "-".join(id_split[:2])
                id_split.pop(1)

            melodic_instruments = [s.split(",") for s in line[1].split("+")]
            melodic_instruments = [
                item.lower() for sublist in melodic_instruments for item in sublist
            ]
            for i, inst in enumerate(melodic_instruments):
                if inst == "string":
                    melodic_instruments[i] = "strings"
                elif inst == "winds (solo)":
                    melodic_instruments[i] = "winds"
            melodic_instruments = sorted(list(set(melodic_instruments)))

            metadata_index[track_id] = {
                "predominant_melodic_instruments-raw": line[1],
                "predominant_melodic_instruments-normalized": melodic_instruments,
                "alternating_melody": tf_dict[line[2]],
                "contains_winds": tf_dict[line[3]],
                "contains_strings": tf_dict[line[4]],
                "contains_brass": tf_dict[line[5]],
                "only_strings": tf_dict[line[6]],
                "only_winds": tf_dict[line[7]],
                "only_brass": tf_dict[line[8]],
                "composer": id_split[0],
                "work": "-".join(id_split[1:-1]),
                "excerpt": id_split[-1][2:],
            }

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.orchset.load_audio_mono", version="0.3.4")
    def load_audio_mono(self, *args, **kwargs):
        return load_audio_mono(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.orchset.load_audio_stereo", version="0.3.4")
    def load_audio_stereo(self, *args, **kwargs):
        return load_audio_stereo(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.orchset.load_melody", version="0.3.4")
    def load_melody(self, *args, **kwargs):
        return load_melody(*args, **kwargs)
