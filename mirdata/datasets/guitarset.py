"""GuitarSet Loader.

.. admonition:: Dataset Info
    :class: dropdown

    GuitarSet provides audio recordings of a variety of musical excerpts
    played on an acoustic guitar, along with time-aligned annotations
    including pitch contours, string and fret positions, chords, beats,
    downbeats, and keys.

    GuitarSet contains 360 excerpts that are close to 30 seconds in length.
    The 360 excerpts are the result of the following combinations:

    - 6 players
    - 2 versions: comping (harmonic accompaniment) and soloing (melodic improvisation)
    - 5 styles: Rock, Singer-Songwriter, Bossa Nova, Jazz, and Funk
    - 3 Progressions: 12 Bar Blues, Autumn Leaves, and Pachelbel Canon.
    - 2 Tempi: slow and fast.

    The tonality (key) of each excerpt is sampled uniformly at random.

    GuitarSet was recorded with the help of a hexaphonic pickup, which outputs
    signals for each string separately, allowing automated note-level annotation.
    Excerpts are recorded with both the hexaphonic pickup and a Neumann U-87
    condenser microphone as reference.
    3 audio recordings are provided with each excerpt with the following suffix:

    - hex: original 6 channel wave file from hexaphonic pickup
    - hex_cln: hex wave files with interference removal applied
    - mic: monophonic recording from reference microphone
    - mix: monophonic mixture of original 6 channel file

    Each of the 360 excerpts has an accompanying JAMS file which stores 16 annotations.
    Pitch:

    - 6 pitch_contour annotations (1 per string)
    - 6 midi_note annotations (1 per string)

    Beat and Tempo:

    - 1 beat_position annotation
    - 1 tempo annotation

    Chords:

    - 2 chord annotations: instructed and performed. The instructed chord annotation
      is a digital version of the lead sheet that's provided to the player, and the
      performed chord annotations are inferred from note annotations, using
      segmentation and root from the digital lead sheet annotation.

    For more details, please visit: http://github.com/marl/guitarset/
"""
import logging
from typing import BinaryIO, Dict, List, Optional, TextIO, Tuple

import jams
import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io

BIBTEX = """@inproceedings{xi2018guitarset,
title={GuitarSet: A Dataset for Guitar Transcription},
author={Xi, Qingyang and Bittner, Rachel M and Ye, Xuzhou and Pauwels, Johan and Bello, Juan P},
booktitle={International Society of Music Information Retrieval (ISMIR)},
year={2018}
}"""

INDEXES = {
    "default": "1.1.0",
    "test": "1.1.0",
    "1.1.0": core.Index(filename="guitarset_index_1.1.0.json"),
}

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="annotation.zip",
        url="https://zenodo.org/record/3371780/files/annotation.zip?download=1",
        checksum="b39b78e63d3446f2e54ddb7a54df9b10",
        destination_dir="annotation",
    ),
    "audio_hex_debleeded": download_utils.RemoteFileMetadata(
        filename="audio_hex-pickup_debleeded.zip",
        url="https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1",
        checksum="c31d97279464c9a67e640cb9061fb0c6",
        destination_dir="audio_hex-pickup_debleeded",
    ),
    "audio_hex_original": download_utils.RemoteFileMetadata(
        filename="audio_hex-pickup_original.zip",
        url="https://zenodo.org/record/3371780/files/audio_hex-pickup_original.zip?download=1",
        checksum="f9911bf217cb40e9e68edf3726ef86cc",
        destination_dir="audio_hex-pickup_original",
    ),
    "audio_mic": download_utils.RemoteFileMetadata(
        filename="audio_mono-mic.zip",
        url="https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1",
        checksum="275966d6610ac34999b58426beb119c3",
        destination_dir="audio_mono-mic",
    ),
    "audio_mix": download_utils.RemoteFileMetadata(
        filename="audio_mono-pickup_mix.zip",
        url="https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip?download=1",
        checksum="aecce79f425a44e2055e46f680e10f6a",
        destination_dir="audio_mono-pickup_mix",
    ),
}
_STYLE_DICT = {
    "Jazz": "Jazz",
    "BN": "Bossa Nova",
    "Rock": "Rock",
    "SS": "Singer-Songwriter",
    "Funk": "Funk",
}
_GUITAR_STRINGS = ["E", "A", "D", "G", "B", "e"]
CONTOUR_HOP = 256.0 / 44100

LICENSE_INFO = "MIT License."


class Track(core.Track):
    """Guitarset Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_hex_cln_path (str): path to the debleeded hex wave file
        audio_hex_path (str): path to the original hex wave file
        audio_mic_path (str): path to the mono wave via microphone
        audio_mix_path (str): path to the mono wave via downmixing hex pickup
        jams_path (str): path to the jams file
        mode (str): one of ['solo', 'comp']
            For each excerpt, players are asked to first play in 'comp' mode
            and later play a 'solo' version on top of the already recorded comp.
        player_id (str): ID of the different players.
            one of ['00', '01', ... , '05']
        style (str): one of ['Jazz', 'Bossa Nova', 'Rock', 'Singer-Songwriter', 'Funk']
        tempo (float): BPM of the track
        track_id (str): track id

    Cached Properties:
        beats (BeatData): beat positions
        leadsheet_chords (ChordData): chords as written in the leadsheet
        inferred_chords (ChordData): chords inferred from played transcription
        key_mode (KeyData): key and mode
        pitch_contours (dict):
            Pitch contours per string
            - 'E': F0Data(...)
            - 'A': F0Data(...)
            - 'D': F0Data(...)
            - 'G': F0Data(...)
            - 'B': F0Data(...)
            - 'e': F0Data(...)
        multif0 (MultiF0Data): all pitch contour data as one multif0 annotation
        notes (dict):
            Notes per string
            - 'E': NoteData(...)
            - 'A': NoteData(...)
            - 'D': NoteData(...)
            - 'G': NoteData(...)
            - 'B': NoteData(...)
            - 'e': NoteData(...)
        notes_all (NoteData): all note data as one note annotation
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_hex_cln_path = self.get_path("audio_hex_cln")
        self.audio_hex_path = self.get_path("audio_hex")
        self.audio_mic_path = self.get_path("audio_mic")
        self.audio_mix_path = self.get_path("audio_mix")
        self.jams_path = self.get_path("jams")

        title_list = track_id.split("_")  # [PID, S-T-K, mode, rec_mode]
        style, tempo, _ = title_list[1].split("-")  # [style, tempo, key]
        self.player_id = title_list[0]
        self.mode = title_list[2]
        self.tempo = float(tempo)
        self.style = _STYLE_DICT[style[:-1]]

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.jams_path)

    @core.cached_property
    def leadsheet_chords(self):
        if self.mode == "solo":
            logging.info("Chord annotations for solo excerpts are the same with the comp excerpt.")
        return load_chords(self.jams_path, True)

    @core.cached_property
    def inferred_chords(self):
        if self.mode == "solo":
            logging.info("Chord annotations for solo excerpts are the same as the comp excerpt.")
        return load_chords(self.jams_path, False)

    @core.cached_property
    def key_mode(self) -> Optional[annotations.KeyData]:
        return load_key_mode(self.jams_path)

    @core.cached_property
    def pitch_contours(self) -> Dict[str, annotations.F0Data]:
        contours = {}
        # iterate over 6 strings
        for i in range(6):
            contours[_GUITAR_STRINGS[i]] = load_pitch_contour(self.jams_path, i)
        return contours

    @core.cached_property
    def multif0(self) -> annotations.MultiF0Data:
        contours: List[annotations.F0Data] = list(self.pitch_contours.values())
        max_times = np.argmax(
            [0 if contour_data is None else len(contour_data.times) for contour_data in contours]
        )  # type: ignore
        times = contours[max_times].times  # type: ignore
        frequency_list: List[list] = [[] for _ in times]
        for contour in contours:
            if contour is None:
                continue

            for i, f in enumerate(contour.frequencies):
                if f > 0:
                    frequency_list[i].append(f)
        return annotations.MultiF0Data(times, "s", frequency_list, "hz")

    @core.cached_property
    def notes(self) -> Dict[str, annotations.NoteData]:
        notes = {}
        # iterate over 6 strings
        for i in range(6):
            notes[_GUITAR_STRINGS[i]] = load_notes(self.jams_path, i)
        return notes

    @core.cached_property
    def notes_all(self) -> Optional[annotations.NoteData]:
        all_note_data = None
        for note_data in self.notes.values():
            if all_note_data is None:
                all_note_data = note_data
            else:
                all_note_data += note_data
        return all_note_data

    @property
    def audio_mic(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_mic_path)

    @property
    def audio_mix(self) -> Optional[Tuple[np.ndarray, float]]:
        """Mixture audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_mix_path)

    @property
    def audio_hex(self) -> Optional[Tuple[np.ndarray, float]]:
        """Hexaphonic audio (6-channels) with one channel per string.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_multitrack_audio(self.audio_hex_path)

    @property
    def audio_hex_cln(self) -> Optional[Tuple[np.ndarray, float]]:
        """Hexaphonic audio (6-channels) with one channel per string after
        bleed removal.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_multitrack_audio(self.audio_hex_cln_path)

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams.load(self.jams_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Guitarset audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_bytes_io
def load_multitrack_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Guitarset multitrack audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=False)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load a Guitarset beats annotation.

    Args:
        fhandle (str or file-like): File-like object or path
            of the jams annotation file

    Returns:
        BeatData: Beat data
    """
    jam = jams.load(fhandle)
    anno = jam.search(namespace="beat_position")[0]
    times, values = anno.to_event_values()
    positions = [int(v["position"]) for v in values]
    return annotations.BeatData(times, "s", np.array(positions), "bar_index")


# no decorator because of https://github.com/mir-dataset-loaders/mirdata/issues/503
def load_chords(jams_path, leadsheet_version):
    """Load a guitarset chord annotation.

    Args:
        jams_path (str): path to the jams annotation file
        leadsheet_version (Bool):
            Whether or not to load the leadsheet version of the chord annotation
            If False, load the infered version.

    Returns:
        ChordData: Chord data
    """
    try:
        with open(jams_path, "r") as fhandle:
            jam = jams.load(fhandle)
    except FileNotFoundError:
        raise FileNotFoundError("jams_path {} does not exist".format(jams_path))

    if leadsheet_version:
        anno = jam.search(namespace="chord")[0]
    else:
        anno = jam.search(namespace="chord")[1]
    intervals, values = anno.to_interval_values()
    return annotations.ChordData(intervals, "s", values, "jams")


@io.coerce_to_string_io
def load_key_mode(fhandle: TextIO) -> annotations.KeyData:
    """Load a Guitarset key-mode annotation.

    Args:
        fhandle (str or file-like): File-like object or path of the jams annotation file

    Returns:
        KeyData: Key data
    """
    jam = jams.load(fhandle)
    anno = jam.search(namespace="key_mode")[0]
    intervals, values = anno.to_interval_values()
    return annotations.KeyData(intervals, "s", values, "key_mode")


def _fill_pitch_contour(times, freqs, voicing, max_time, contour_hop, duration=None):
    """Fill a pitch contour with missing time stamps (during unpitched frames)

    Args:
        times (np.array): array of time stamps in seconds
        freqs (np.array): array of pitch values in Hz
        voicing (np.array): array of voicings
        max_time (float): maximum time stamp
        contour_hop (float): hop size in seconds
        duration (float, optional): Total duration. Defaults to None.

    Returns:
        tuple: filled_times, filled_frequencies, filled_voicing
    """
    if duration is not None and max_time > duration:
        max_time = duration
    n_stamps = int(np.floor((max_time / contour_hop)))
    filled_times = np.arange(n_stamps) * contour_hop
    filled_freqs = np.zeros((len(filled_times),))
    filled_voicing = np.zeros((len(filled_times),))

    for time, freq, voc in zip(times, freqs, voicing):
        t_idx = int(np.round(time / contour_hop))
        if time > max_time or t_idx >= n_stamps:
            continue
        filled_freqs[t_idx] = freq
        filled_voicing[t_idx] = voc

    return filled_times, filled_freqs, filled_voicing


# no decorator because of https://github.com/mir-dataset-loaders/mirdata/issues/503
def load_pitch_contour(jams_path, string_num):
    """Load a guitarset pitch contour annotation for a given string.

    Args:
        jams_path (str): path to the jams annotation file
        string_num (int), in range(6): Which string to load.
            0 is the Low E string, 5 is the high e string.

    Returns:
        F0Data: Pitch contour data for the given string
    """
    try:
        with open(jams_path, "r") as fhandle:
            jam = jams.load(fhandle)
    except FileNotFoundError:
        raise FileNotFoundError("jams_path {} does not exist".format(jams_path))

    anno_arr = jam.search(namespace="pitch_contour")
    anno = anno_arr.search(data_source=str(string_num))[0]
    times, values = anno.to_event_values()
    if len(times) == 0:
        return None
    frequencies = np.array([v["frequency"] for v in values])
    voicing = np.array([float(v["voiced"]) for v in values])
    voicing[frequencies == 0] = 0

    filled_times, filled_freqs, filled_voicing = _fill_pitch_contour(
        times, frequencies, voicing, np.max(times), CONTOUR_HOP
    )

    return annotations.F0Data(filled_times, "s", filled_freqs, "hz", filled_voicing, "binary")


# no decorator because of https://github.com/mir-dataset-loaders/mirdata/issues/503
def load_notes(jams_path, string_num):
    """Load a guitarset note annotation for a given string.

    Args:
        jams_path (str): path to the jams annotation file
        string_num (int), in range(6): Which string to load.
            0 is the Low E string, 5 is the high e string.

    Returns:
        NoteData: Note data for the given string
    """
    try:
        with open(jams_path) as fhandle:
            jam = jams.load(fhandle)
    except FileNotFoundError:
        raise FileNotFoundError("jams_path {} does not exist".format(jams_path))

    anno_arr = jam.search(namespace="note_midi")
    anno = anno_arr.search(data_source=str(string_num))[0]
    intervals, values = anno.to_interval_values()
    if len(values) == 0:
        return None
    return annotations.NoteData(intervals, "s", np.array(values), "midi")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The guitarset dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="guitarset",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.guitarset.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.guitarset.load_multitrack_audio", version="0.3.4")
    def load_multitrack_audio(self, *args, **kwargs):
        return load_multitrack_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.guitarset.load_beats", version="0.3.4")
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.guitarset.load_chords", version="0.3.4")
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.guitarset.load_key_mode", version="0.3.4")
    def load_key_mode(self, *args, **kwargs):
        return load_key_mode(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.guitarset.load_pitch_contour", version="0.3.4")
    def load_pitch_contour(self, *args, **kwargs):
        return load_pitch_contour(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.guitarset.load_notes", version="0.3.4")
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)
