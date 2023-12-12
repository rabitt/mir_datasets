# -*- coding: utf-8 -*-
"""Good-Sounds Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Good-Sounds dataset is born of the collaboration between the Music Technology Group and Korg. Good-Sounds
    [2, 16] is carried out recording a training dataset of single note excerpts including six classes of sounds per
    studied instrument. Twelve different instruments are recorded, as is shown in Table 2. For each instrument,
    the complete range of playable semitones is captured several times with various tonal characteristics. There
    are two classes: Good and Bad sounds. Bad sounds are divided into five sub-classes, one for each musical dimension
    stated by the expert musicians. Bad sounds are composed by examples of note recordings
    that are intentionally badly played. The last class includes examples of note recordings that are considered to
    be well played.

    This dataset was created in the context of the Pablo project, partially funded by KORG Inc. It contains monophonic
    recordings of two kind of exercises: single notes and scales.
    The recordings were made in the Universitat Pompeu Fabra / Phonos recording studio by 15 different professional
    musicians, all of them holding a music degree and having some expertise in teaching. 12 different instruments were
    recorded using one or up to 4 different microphones (depending on the recording session). For all the instruments
    the whole set of playable semitones in the instrument is recorded several times with different tonal characteristics.
    Each note is recorded into a separate mono .flac audio file of 48kHz and 32 bits. The tonal characteristics are
    explained both in the the following section and the related publication. The database is meant for organizing the
    sounds in a handy way. It is organised in four different entities: sounds, takes, packs and ratings.
"""
import json
import os
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{romani2015real,
  title={A Real-Time System for Measuring Sound Goodness in Instrumental Sounds},
  author={Romani Picas, Oriol and Parra Rodriguez, Hector and Dabiri, Dara and Tokuda, Hiroshi and Hariya, Wataru and Oishi, Koji and Serra, Xavier},
  booktitle={Audio Engineering Society Convention 138},
  year={2015},
  organization={Audio Engineering Society}
}"""
INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="good_sounds_index_1.0.json"),
}
REMOTES = {
    "packs": download_utils.RemoteFileMetadata(
        filename="packs.json",
        url="https://zenodo.org/record/4588740/files/packs.json?download=1",
        checksum="3b512c280f8be64ccb59b0b294e84610",
        destination_dir=".",
    ),
    "ratings": download_utils.RemoteFileMetadata(
        filename="ratings.json",
        url="https://zenodo.org/record/4588740/files/ratings.json?download=1",
        checksum="b50b95fc7eb996b31a7cd290070f8059",
        destination_dir=".",
    ),
    "sounds": download_utils.RemoteFileMetadata(
        filename="sounds.json",
        url="https://zenodo.org/record/4588740/files/sounds.json?download=1",
        checksum="a60d90a964fd567ebc2e4b4e3f2990f2",
        destination_dir=".",
    ),
    "takes": download_utils.RemoteFileMetadata(
        filename="takes.json",
        url="https://zenodo.org/record/4588740/files/takes.json?download=1",
        checksum="318e840031397314907e7f9420f2abeb",
        destination_dir=".",
    ),
    "audios": download_utils.RemoteFileMetadata(
        filename="good-sounds.zip",
        url="https://zenodo.org/record/820937/files/good-sounds.zip?download=1",
        checksum="2137bbb2d32c1d60aa51e1301225f541",
        destination_dir=".",
    ),
}


class Track(core.Track):
    """GOOD-SOUNDS Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): Path to the audio file

    Cached Properties:
        ratings_info (dict):  A dictionary containing the entity Ratings.
        Some musicians self-rated their performance in a 0-10 goodness scale for the user evaluation of the first project
        prototype. Please read the paper for more detailed information.
            - id
            - mark: the rate or score.
            - type: the klass of the sound. Related to the tags of the sound.
            - created_at
            - comments
            - sound_id
            - rater: the musician who rated the sound.
        pack_info (dict): A dictionary containing the entity Pack. A pack is a group of sounds from the same recording
            session. The audio files are organised in the sound_files directory in subfolders with the pack name to which
            they belong. The following metadata is associated with the entity Pack.
            - id
            - name
            - description
        sound_info (dict): A dictionary containing the entity Sound. A sound can have several takes as some of them were
            recorded using different microphones at the same time. The following
            metadata is associated with the entity Sound.
            - id
            - instrument: flute, cello, clarinet, trumpet, violin, sax_alto, sax_tenor, sax_baritone, sax_soprano, oboe, piccolo, bass
            - note
            - octave
            - dynamics: for some sounds, the musical notation of the loudness level (p, mf, f..)
            - recorded_at: recording date and time
            - location: recording place
            - player: the musician who recorded. For detailed information about the musicians contact us.
            - bow_velocity: for some string instruments the velocity of the bow (slow, medieum, fast)
            - bridge_position: for some string instruments the position of the bow (tasto, middle, ponticello)
            - string: for some string instruments the number of the string in which the sound it's played (1: lowest in pitch)
            - csv_file: used for creation of the DB
            - csv_id: used for creation of the DB
            - pack_filename: used for creation of the DB
            - pack_id: used for creation of the DB
            - attack: for single notes, manual annotation of the onset in samples.
            - decay: for single notes, manual annotation of the decay in samples.
            - sustain: for single notes, manual annotation of the beginnig of the sustained part in samples.
            - release: for single notes, manual annotation of the beginnig of the release part in samples.
            - offset: for single notes, manual annotation of the offset in samples
            - reference: 1 if sound was used to create the models in the good-sounds project, 0 if not.
            - Other tags regarding tonal characteristics are also available.
            - comments: if any
            - semitone: midi note
            - pitch_reference: the reference pitch
            - klass: user generated tags of the tonal qualities of the sound. They also contain information about the exercise, that could be single note or scale.
            * "good-sound":  good examples of single note
            * "bad": bad example of one of the sound attributes defined in the project (please read the papers for a detailed explanation)
            * "scale-good": good example of a one octave scale going up and down (15 notes). If the scale is minor a tagged "minor" is also available.
            * "scale-bad": bad example scale of one of the sounds defined in the project. (15 notes up and down).
        take_info (dict): A dictionary containing the entity Take. A sound can have several takes as some of them were
            recorded using different microphones at the same time. Each take has an associated audio file.
            The annotations.
            Each take has an associated audio file. The following
            metadata is associated with the entity Sound.
            - id
            - microphone
            - filename: the name of the associated audio file
            - original_filename:
            - freesound_id: for some sounds uploaded to freesound.org
            - sound_id: the id of the sound in the DB
            - goodsound_id: for some of the sounds available in good-sounds.org
        microphone (str): the microphone used to record the take.
        instrument (str): the instrument recorded (flute, cello, clarinet, trumpet, violin, sax_alto, sax_tenor, sax_baritone, sax_soprano, oboe, piccolo, bass).
        klass (str): user generated tags of the tonal qualities of the sound. They also contain information about the exercise, that could be single note or scale.
            * "good-sound":  good examples of single note
            * "bad": bad example of one of the sound attributes defined in the project (please read the papers for a detailed explanation)
            * "scale-good": good example of a one octave scale going up and down (15 notes). If the scale is minor a tagged "minor" is also available.
            * "scale-bad": bad example scale of one of the sounds defined in the project. (15 notes up and down).
        semitone (int): midi note
        pitch_reference (int): the reference pitch
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.track_id = track_id
        self.audio_path = self.get_path("audio")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    @core.cached_property
    def sound_info(self) -> dict:
        return self._metadata()["sounds"][str(self._metadata()["takes"][self.track_id]["sound_id"])]

    @core.cached_property
    def take_info(self) -> dict:
        take_info = self._metadata()["takes"][self.track_id]
        take_info["filename"] = self.audio_path
        return take_info

    @core.cached_property
    def ratings_info(self) -> list:
        sound_id = str(self._metadata()["takes"][self.track_id]["sound_id"])
        return list(
            filter(
                lambda rating: rating["sound_id"] == sound_id,
                self._metadata()["ratings"].values(),
            )
        )

    @core.cached_property
    def pack_info(self) -> dict:
        return self._metadata()["packs"][
            str(
                self._metadata()["sounds"][
                    str(self._metadata()["takes"][self.track_id]["sound_id"])
                ]["pack_id"]
            )
        ]

    @core.cached_property
    def microphone(self) -> str:
        return self.take_info["microphone"]

    @core.cached_property
    def instrument(self) -> str:
        return self.sound_info["instrument"]

    @core.cached_property
    def klass(self) -> str:
        return self.sound_info["klass"]

    @core.cached_property
    def semitone(self) -> str:
        return self.sound_info["semitone"]

    @core.cached_property
    def pitch_reference(self) -> str:
        return self.sound_info["pitch_reference"]

    def to_jams(self):
        # Initialize top-level JAMS container
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "sound": self.sound_info,
                "take": self.take_info,
                "ratings": self.ratings_info,
                "pack": self.pack_info,
                "microphone": self.microphone,
                "instrument": self.instrument,
                "klass": self.klass,
                "semitone": self.semitone,
                "pitch_reference": self.pitch_reference,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a GOOD-SOUNDS audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The GOOD-SOUNDS dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="good_sounds",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info="Creative Commons Attribution Share Alike 4.0 International.",
        )

    @core.cached_property
    def _metadata(self):
        packs = os.path.join(self.data_home, "packs.json")
        ratings = os.path.join(self.data_home, "ratings.json")
        sounds = os.path.join(self.data_home, "sounds.json")
        takes = os.path.join(self.data_home, "takes.json")

        try:
            with open(packs, "r") as fhandle:
                packs = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Packs metadata not found. Did you run .download()?")

        try:
            with open(ratings, "r") as fhandle:
                ratings = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Ratings metadata not found. Did you run .download()?")

        try:
            with open(sounds, "r") as fhandle:
                sounds = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Sounds metadata not found. Did you run .download()?")

        try:
            with open(takes, "r") as fhandle:
                takes = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Takes metadata not found. Did you run .download()?")

        return {"packs": packs, "ratings": ratings, "sounds": sounds, "takes": takes}

    @deprecated(reason="Use mirdata.datasets.good_sounds.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
