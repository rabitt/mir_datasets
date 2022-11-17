"""EGFxSet Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    EGFxSet (Electric Guitar Effects dataset) features recordings for all clean tones in a 22-fret Stratocaster,
    recorded with 5 different pickup configurations, also processed through 12 popular guitar effects.
    Our dataset was recorded in real hardware, making it relevant for music information retrieval tasks on real music.
    We also include annotations for parameter settings of the effects we used.

    This dataset was conceived during Iran Roman's "Deep Learning for Music Information Retrieval" course
    imparted in the postgraduate studies in music technology at the UNAM (Universidad Nacional Autónoma de México). 
    The result is a combined effort between two UNAM postgraduate students (Hegel Pedroza and Gerardo Meza) and Iran Roman(NYU).    

    EGFxSet is a dataset of 8,970 audio files with a 5-second duration each,
    summing a total time of - 12 hours and 28 minutes -.
    
    All possible 138 notes of a standard tuning 22 frets guitar were recorded in each one of the 5 pickup configurations,
    giving a total of 690 clean tone audio files ( 58 min ).

    The 690 audio files were processed through 12 different audio effects employing actual guitar gear (no emulations),
    summing a total of 8,280 proceed audio files ( 11 hours 30 min ).

    The effects employed were divided into four categories, and each category comprised three different effects.
    Sometimes there were employed more than one effect from a same guitar equipment.

    Categories, Models and Effects:

        Distortion:
            Boss BD-2: Blues Driver
            Ibanez Minitube Screamer: Tube Screamer
            ProCo RAT2: Distortion

        Modulation:
            Boss CE-3: Chorus
            MXR Phase 45: Phaser
            Mooer E-Lady: Flangergh pr checkout 556

        Delays:
            Line6 DL-4:
                        Digital Delay
                        Tape Echo
                        Sweep Echo

        Reverb:
            Orange CR-60 Combo Amplifier:
                                        Plate Reverb
                                        Hall Reverb
                                        Spring Reverb



    Annotations are labeled by a trained electric guitar musician. For each tone, we provide:

    Guitar string number
    Fret number
    Guitar pickup configuration
    Effect name
    Effect type
    Hardware model
    Knob names
    Knob types
    Knob settings

    For more details, please visit https://zenodo.org/record/7044411
"""
import csv
import os
from typing import BinaryIO, Optional, Tuple
from ast import literal_eval

import librosa
import numpy as np
from smart_open import open

from mirdata import core, download_utils, jams_utils, io

BIBTEX = """
@article{pedrozaegfxset,
  title={EGFxSet: Electric guitar tones processed through real effects of distortion, modulation, delay and reverb},
  author={Pedroza, Hegel and Meza, Gerardo and Roman, Iran}
}
"""

INDEXES = {
    "default": "1",
    "test": "1",
    "1": core.Index(filename="egfxset_index_1.json"),
}

REMOTES = {
    "bluesDriver": download_utils.RemoteFileMetadata(
        filename="BluesDriver.zip",
        url="https://zenodo.org/record/7044411/files/BluesDriver.zip?download=1",
        checksum="b1d6dce9064a25a1cff2a0c40c30a2e4",
    ),
    "chorus": download_utils.RemoteFileMetadata(
        filename="Chorus.zip",
        url="https://zenodo.org/record/7044411/files/Chorus.zip?download=1",
        checksum="3698b3b1756917f93dadf27517d48479",
    ),
    "clean": download_utils.RemoteFileMetadata(
        filename="Clean.zip",
        url="https://zenodo.org/record/7044411/files/Clean.zip?download=1",
        checksum="cdb1b401960f56becc8640387910e78a",
    ),
    "digitalDelay": download_utils.RemoteFileMetadata(
        filename="Digital-Delay.zip",
        url="https://zenodo.org/record/7044411/files/Digital-Delay.zip?download=1",
        checksum="4a25d57bcb0083667bade7b3c42460bc",
    ),
    "flanger": download_utils.RemoteFileMetadata(
        filename="Flanger.zip",
        url="https://zenodo.org/record/7044411/files/Flanger.zip?download=1",
        checksum="f3f7b39c895a400d35c5b1314a1122bd",
    ),
    "hallReverb": download_utils.RemoteFileMetadata(
        filename="Hall-Reverb.zip",
        url="https://zenodo.org/record/7044411/files/Hall-Reverb.zip?download=1",
        checksum="c173bebdcbed50d4bc8803e0b30d6517",
    ),
    "phaser": download_utils.RemoteFileMetadata(
        filename="Phaser.zip",
        url="https://zenodo.org/record/7044411/files/Phaser.zip?download=1",
        checksum="1842e2643dd34d7285a77506ca540df3",
    ),
    "plateReverb": download_utils.RemoteFileMetadata(
        filename="Plate-Reverb.zip",
        url="https://zenodo.org/record/7044411/files/Plate-Reverb.zip?download=1",
        checksum="abbcc68d692f323e8af6aeb8d478c40d",
    ),
    "rat": download_utils.RemoteFileMetadata(
        filename="RAT.zip",
        url="https://zenodo.org/record/7044411/files/RAT.zip?download=1",
        checksum="afe9fc757a51d04126c23159706f4e8e",
    ),
    "spring-Reverb": download_utils.RemoteFileMetadata(
        filename="Spring-Reverb.zip",
        url="https://zenodo.org/record/7044411/files/Spring-Reverb.zip?download=1",
        checksum="21afc47594ed8f37db008f313e50c634",
    ),
    "sweepEcho": download_utils.RemoteFileMetadata(
        filename="Sweep-Echo.zip",
        url="https://zenodo.org/record/7044411/files/Sweep-Echo.zip?download=1",
        checksum="ea6dda440e9af6a19173facdf2bf17ac",
    ),
    "tapeEcho": download_utils.RemoteFileMetadata(
        filename="TapeEcho.zip",
        url="https://zenodo.org/record/7044411/files/TapeEcho.zip?download=1",
        checksum="77adf4a6a8ed4eb566b2b8e77735c7dc",
    ),
    "tubeScreamer": download_utils.RemoteFileMetadata(
        filename="TubeScreamer.zip",
        url="https://zenodo.org/record/7044411/files/TubeScreamer.zip?download=1",
        checksum="b9c46ed65037d0bd17bdf82dc3125beb",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="egfxset_metadata.csv",
        url="https://zenodo.org/record/7044411/files/egfxset_metadata.csv?download=1",
        checksum="ec8d160fe79469c7de8cad528d7d35e1",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Track(core.Track):
    """EGFxSet Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        track_id (str): track id
        stringfret_tuple (str): the tuple of the note recorded
        note (str): the notename of the file
        midinote (int): the midinote value
        pickup_configuration: the pickup used in the recording
        effect (str): the effect recorded
        model (str): the model of the hardware used
        effect_type (str) the type of effect used (distortion, modulation, delay or reverb)
        knob_names (list): an array with the knob names of the effect used or "None" when the recording is a clean effect sound
        knob_type (list): an array with the type of knobs of the effect used or "None" when the recording is a clean effect sound
        setting (list): the setting of the effect recorded or "None" when the recording is a clean effect sound
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

    @property
    def stringfret_tuple(self):
        return self._track_metadata.get("String-fret Tuple")

    @property
    def note(self):
        return self._track_metadata.get("Note")

    @property
    def midinote(self):
        return self._track_metadata.get("Midinote")

    @property
    def pickup_configuration(self):
        return self._track_metadata.get("Pickup Configuration")

    @property
    def effect(self):
        return self._track_metadata.get("Effect")

    @property
    def model(self):
        return self._track_metadata.get("Model")

    @property
    def effect_type(self):
        return self._track_metadata.get("Effect Type")

    @property
    def knob_names(self):
        return self._track_metadata.get("Knob Names")

    @property
    def knob_type(self):
        return self._track_metadata.get("Knob Type")

    @property
    def setting(self):
        return self._track_metadata.get("Setting")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """Solo guitar audio (mono)

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
            audio_path=self.audio_path, metadata=self._track_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load EGFxSet guitar audio

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate

    """
    return librosa.load(fhandle, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The EGFxSet dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="egfxset",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "egfxset_metadata.csv")
        metadata_index = {}
        tracknames = self.track_ids
        cuerdas = {"1": 64, "2": 59, "3": 55, "4": 50, "5": 45, "6": 40}

        try:
            with open(metadata_path, "r") as fhandle:
                csv_reader = csv.reader(fhandle, delimiter=",")
                next(csv_reader)
                for row in csv_reader:
                    key = os.path.splitext(os.path.split(row[0])[1])[0]
                    for track in tracknames:

                        noteCord = track.split("/")[1].split("-")
                        if len(noteCord) != 2:
                            noteCord = track.split("/")[1].split(".")

                        if track[:3].upper() == "RAT" and key == "distortion":
                            metadata_index[track] = {
                                "String-fret Tuple": track.split("/")[1],
                                "Note": librosa.midi_to_note(
                                    (cuerdas[noteCord[0]] + int(float(noteCord[1])))
                                ),
                                "Midinote": (
                                    cuerdas[noteCord[0]] + int(float(noteCord[1]))
                                ),
                                "Pickup Configuration": track.split("_")[1].split("/")[
                                    0
                                ],
                                "Effect": row[0],
                                "Model": row[1],
                                "Effect Type": row[2],
                                "Knob Names": literal_eval(row[3]),
                                "Knob Type": literal_eval(row[4]),
                                "Setting": literal_eval(row[5]),
                            }

                        if track[:2].upper() == key[:2].upper():
                            metadata_index[track] = {
                                "String-fret Tuple": track.split("/")[1],
                                "Note": librosa.midi_to_note(
                                    (cuerdas[noteCord[0]] + int(float(noteCord[1])))
                                ),
                                "Midinote": (
                                    cuerdas[noteCord[0]] + int(float(noteCord[1]))
                                ),
                                "Pickup Configuration": track.split("_")[1].split("/")[
                                    0
                                ],
                                "Effect": row[0],
                                "Model": row[1],
                                "Effect Type": row[2],
                                "Knob Names": literal_eval(row[3]),
                                "Knob Type": literal_eval(row[4]),
                                "Setting": literal_eval(row[5]),
                            }

                        if track[:2].upper() == "CL":
                            metadata_index[track] = {
                                "String-fret Tuple": track.split("/")[1],
                                "Note": librosa.midi_to_note(
                                    (cuerdas[noteCord[0]] + int(float(noteCord[1])))
                                ),
                                "Midinote": (
                                    cuerdas[noteCord[0]] + int(float(noteCord[1]))
                                ),
                                "Pickup Configuration": track.split("_")[1].split("/")[
                                    0
                                ],
                                "Effect": "clean",
                                "Model": "None",
                                "Effect Type": "None",
                                "Knob Names": "None",
                                "Knob Type": "None",
                                "Setting": "None",
                            }

        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata_index

    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
