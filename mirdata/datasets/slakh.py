"""Slakh Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Synthesized Lakh (Slakh) Dataset is a dataset of multi-track audio and aligned
    MIDI for music source separation and multi-instrument automatic transcription.
    Individual MIDI tracks are synthesized from the Lakh MIDI Dataset v0.1 using
    professional-grade sample-based virtual instruments, and the resulting audio is
    mixed together to make musical mixtures.

    The original release of Slakh, called Slakh2100,
    contains 2100 automatically mixed tracks and accompanying, aligned MIDI files,
    synthesized from 187 instrument patches categorized into 34 classes, totaling
    145 hours of mixture data.

    This loader supports two versions of Slakh:
    - Slakh2100-redux: a deduplicated version of slakh2100 containing 1710 multitracks
    - baby-slakh: a mini version with 16k wav audio and only the first 20 tracks

    This dataset was created at Mitsubishi Electric Research Labl (MERL) and
    Interactive Audio Lab at Northwestern University by Ethan Manilow,
    Gordon Wichern, Prem Seetharaman, and Jonathan Le Roux.

    For more information see http://www.slakh.com/
"""
import os
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import yaml
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@inproceedings{manilow2019cutting,
  title={Cutting Music Source Separation Some {Slakh}: A Dataset to Study the Impact of Training Data Quality and Quantity},
  author={Manilow, Ethan and Wichern, Gordon and Seetharaman, Prem and Le Roux, Jonathan},
  booktitle={Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2019},
  organization={IEEE}
}
"""

INDEXES = {
    "default": "2100-redux",
    "test": "baby",
    "2100-redux": core.Index(
        filename="slakh_index_2100-redux.json", partial_download=["2100-redux"]
    ),
    "baby": core.Index(filename="slakh_index_baby.json", partial_download=["baby"]),
}

REMOTES = {
    "2100-redux": download_utils.RemoteFileMetadata(
        filename="slakh2100_flac_redux.tar.gz",
        url="https://zenodo.org/record/4599666/files/slakh2100_flac_redux.tar.gz?download=1",
        checksum="f4b71b6c45ac9b506f59788456b3f0c4",
    ),
    "baby": download_utils.RemoteFileMetadata(
        filename="babyslakh_16k.tar.gz",
        url="https://zenodo.org/record/4603870/files/babyslakh_16k.tar.gz?download=1",
        checksum="311096dc2bde7d61c97e930edbfc7f78",
    ),
}

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""

SPLITS = ["train", "validation", "test", "omitted"]

#: Mixing group to program number mapping
MIXING_GROUPS = {
    "piano": [0, 1, 2, 3, 4, 5, 6, 7],
    "guitar": [24, 25, 26, 27, 28, 29, 30, 31],
    "bass": [32, 33, 34, 35, 36, 37, 38, 39],
    "drums": [128],
}


class Track(core.Track):
    """Slakh Track class, for individual stems.

    Attributes:
        audio_path (str or None): path to the track's audio file. For some unusual tracks,
            such as sound effects, there is no audio and this attribute is None.
        split (str or None): one of 'train', 'validation', 'test', or 'omitted'.
            'omitted' tracks are part of slakh2100-redux which were found to be
            duplicates in the original slakh2011.
            In baby slakh there are no splits, so this attribute is None.
        data_split (str or None): equivalent to split (deprecated in 0.3.6)
        metadata_path (str): path to the multitrack's metadata file
        midi_path (str or None): path to the track's midi file. For some unusual tracks,
            such as sound effects, there is no midi and this attribute is None.
        mtrack_id (str): the track's multitrack id
        track_id (str): track id
        instrument (str): MIDI instrument class, see link for details:
            https://en.wikipedia.org/wiki/General_MIDI#Program_change_events
        integrated_loudness (float): integrated loudness (dB) of this track
            as calculated by the ITU-R BS.1770-4 spec
        is_drum (bool): whether the "drum" flag is true for this MIDI track
        midi_program_name (str): MIDI instrument program name
        plugin_name (str): patch/plugin name that rendered the audio file
        mixing_group (str): which mixing group the track belongs to.
            One of MIXING_GROUPS.
        program_number (int): MIDI instrument program number

    Cached Properties:
        midi (PrettyMIDI): midi data used to generate the audio
        notes (NoteData or None): note representation of the midi data.
            If there are no notes in the midi file, returns None.
        multif0 (MultiF0Data or None): multif0 representaation of the midi data.
            If there are no notes in the midi file, returns None.
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.mtrack_id = self.track_id.split("-")[0]
        self.audio_path = self.get_path("audio")
        self.midi_path = self.get_path("midi")
        self.metadata_path = self.get_path("metadata")

        # split (train/validation/test/omitted) is part of the relative filepath in the index
        self.split = None  # for baby_slakh, there are no data splits - set to None
        if index["version"] == "2100-redux":
            self.split = os.path.normpath(self._track_paths["metadata"][0]).split(os.sep)[1]
            assert self.split in SPLITS, "{} not a valid split - should be one of {}.".format(
                self.split, SPLITS
            )

        self.data_split = self.split  # deprecated in 0.3.6

    @core.cached_property
    def _track_metadata(self) -> dict:
        try:
            with open(self.metadata_path, "r") as fhandle:
                metadata = yaml.safe_load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"track metadata for {self.track_id} not found. Did you run .download()?"
            )
        return metadata["stems"][self.track_id.split("-")[1]]

    @property
    def instrument(self) -> Optional[str]:
        return self._track_metadata.get("inst_class")

    @property
    def integrated_loudness(self) -> Optional[float]:
        return self._track_metadata.get("integrated_loudness")

    @property
    def is_drum(self) -> Optional[bool]:
        return self._track_metadata.get("is_drum")

    @property
    def midi_program_name(self) -> Optional[str]:
        return self._track_metadata.get("midi_program_name")

    @property
    def plugin_name(self) -> Optional[str]:
        return self._track_metadata.get("plugin_name")

    @property
    def program_number(self) -> Optional[int]:
        return self._track_metadata.get("program_num")

    @property
    def mixing_group(self) -> Optional[str]:
        group = [k for k, v in MIXING_GROUPS.items() if self.program_number in v]
        if len(group) == 0:
            return None
        return group[0]

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return io.load_midi(self.midi_path)

    @core.cached_property
    def notes(self) -> Optional[annotations.NoteData]:
        return io.load_notes_from_midi(self.midi_path, self.midi, skip_drums=True)

    @core.cached_property
    def multif0(self) -> Optional[annotations.MultiF0Data]:
        return io.load_multif0_from_midi(
            self.midi_path, self.midi, skip_drums=True, pitch_bend=False
        )

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path, note_data=[(self.notes, "Notes")]
        )


class MultiTrack(core.MultiTrack):
    """Slakh multitrack class, containing information about the mix and the set
    of associated stems.

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_property (str): the name of the attribute of Track which
            returns the audio to be mixed
        mix_path (str): path to the multitrack mix audio
        midi_path (str): path to the full midi data used to generate the mixture
        metadata_path (str): path to the multitrack metadata file
        split (str or None): one of 'train', 'validation', 'test', or 'omitted'.
            'omitted' tracks are part of slakh2100-redux which were found to be
            duplicates in the original slakh2011.
        data_split (str or None): equivalent to split (deprecated in 0.3.6)
        uuid (str): File name of the original MIDI file from Lakh, sans extension
        lakh_midi_dir (str): Path to the original MIDI file from a fresh download of Lakh
        normalized (bool): whether the mix and stems were normalized according to the ITU-R BS.1770-4 spec
        overall_gain (float): gain applied to every stem to make sure mixture does not clip when stems are summed

    Cached Properties:
        midi (PrettyMIDI): midi data used to generate the mixture audio
        notes (NoteData): note representation of the midi data
        multif0 (MultiF0Data): multif0 representation of the midi data
    """

    def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
        super().__init__(
            mtrack_id=mtrack_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            track_class=track_class,
            metadata=metadata,
        )
        self.mix_path = self.get_path("mix")
        self.midi_path = self.get_path("midi")
        self.metadata_path = self.get_path("metadata")

        # split (train/validation/test) is determined by the relative filepath in the index
        self.split = None  # for baby_slakh, there are no data splits - set to None
        if index["version"] == "2100-redux":
            self.split = os.path.normpath(self._multitrack_paths["mix"][0]).split(os.sep)[1]
            assert self.split in SPLITS, "{} not in SPLITS".format(self.split)

        self.data_split = self.split  # deprecated in 0.3.6

    @property
    def track_audio_property(self) -> str:
        return "audio"

    @core.cached_property
    def _multitrack_metadata(self) -> dict:
        try:
            with open(self.metadata_path, "r") as fhandle:
                metadata = yaml.safe_load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
        return metadata

    @property
    def uuid(self) -> Optional[str]:
        return self._multitrack_metadata.get("UUID")

    @property
    def lakh_midi_dir(self) -> Optional[str]:
        return self._multitrack_metadata.get("lmd_midi_dir")

    @property
    def normalized(self) -> Optional[bool]:
        return self._multitrack_metadata.get("normalized")

    @property
    def overall_gain(self) -> Optional[float]:
        return self._multitrack_metadata.get("overall_gain")

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return io.load_midi(self.midi_path)

    @core.cached_property
    def notes(self) -> Optional[annotations.NoteData]:
        return io.load_notes_from_midi(self.midi_path, self.midi)

    @core.cached_property
    def multif0(self) -> Optional[annotations.MultiF0Data]:
        # TODO: setting pitch_bend to False by default, but there are some
        # patches that render pitch bend in the audio.
        return io.load_multif0_from_midi(
            self.midi_path, self.midi, skip_drums=True, pitch_bend=False
        )

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.mix_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.mix_path, note_data=[(self.notes, "Notes")]
        )

    def get_submix_by_group(self, target_groups):
        """Create submixes grouped by instrument type. Creates one submix per
        target group, plus one additional "other" group for any remaining
        sources. Only tracks with available audio are mixed.

        Args:
            target_groups (list): List of target groups. Elements should be one of
                MIXING_GROUPS, e.g. ["bass", "guitar"]

        Returns:
            * submixes (dict): {group: audio_signal} of submixes
            * groups (dict): {group: list of track ids} of submixes
        """
        groups = {}
        submixes = {}
        tracks_with_audio = [track for track in self.tracks.values() if track.audio_path]
        in_group = []
        for group in target_groups:
            groups[group] = [
                track.track_id for track in tracks_with_audio if track.mixing_group == group
            ]
            in_group.extend(groups[group])

            submixes[group] = None if len(groups[group]) == 0 else self.get_target(groups[group])

        groups["other"] = [
            track.track_id for track in tracks_with_audio if track.track_id not in in_group
        ]
        submixes["other"] = None if len(groups["other"]) == 0 else self.get_target(groups["other"])
        return submixes, groups


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a slakh audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=False)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The slakh dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="slakh",
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.slakh.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.slakh.load_midi", version="0.3.4")
    def load_midi(self, *args, **kwargs):
        return io.load_midi(*args, **kwargs)

    @deprecated(reason="Use mirdata.io.load_notes_from_midi", version="0.3.4")
    def load_notes_from_midi(self, *args, **kwargs):
        return io.load_notes_from_midi(*args, **kwargs)

    @deprecated(reason="Use mirdata.io.load_multif0_from_midi", version="0.3.4")
    def load_multif0_from_midi(self, *args, **kwargs):
        return io.load_multif0_from_midi(*args, **kwargs)
