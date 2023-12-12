"""Dagstuhl ChoirSet Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Dagstuhl ChoirSet (DCS) is a multitrack dataset of a cappella choral music.
    The dataset includes recordings of an amateur vocal ensemble performing two
    choir pieces in full choir and quartet settings (total duration 55min 30sec).
    The audio data was recorded during an MIR seminar at Schloss Dagstuhl using
    different close-up microphones to capture the individual singers’ voices:

    * Larynx microphone (LRX): contact microphone attached to the singer's throat.
    * Dynamic microphone (DYN): handheld dynamic microphone.
    * Headset microphone (HSM): microphone close to the singer's mouth.

    LRX, DYN and HSM recordings are provided on the Track level.
    All tracks in the dataset have a LRX recording, while only a subset has DYN and HSM recordings.

    In addition to the close-up microphone tracks, the dataset also provides the following recordings:

    * Room microphone mixdown (STM): mixdown of the stereo room microphone.
    * Room microphone left (STL): left channel of the stereo microphone.
    * Room microphone right (STR): right channel of the stereo microphone.
    * Room microphone mixdown with reverb (StereoReverb_STM): STM signal with artificial reverb.
    * Piano left (SPL): left channel of the piano accompaniment.
    * Piano right (SPR): right channel of the piano accompaniment.

    All room microphone and piano recordings are provided on the Multitrack level.
    All multitracks have room microphone signals, while only a subset has piano recordings.

    For more details, we refer to:
    Sebastian Rosenzweig (1), Helena Cuesta (2), Christof Weiß (1),
    Frank Scherbaum (3), Emilia Gómez (2,4), and Meinard Müller (1):
    Dagstuhl ChoirSet: A Multitrack Dataset for MIR Research on Choral Singing.
    Transactions of the International Society for Music Information Retrieval,
    3(1), pp. 98–110, 2020.
    DOI: https://doi.org/10.5334/tismir.48

    (1) International Audio Laboratories Erlangen, DE
    (2) Music Technology Group, Universitat Pompeu Fabra, Barcelona, ES
    (3) University of Potsdam, DE
    (4) Joint Research Centre, European Commission, Seville, ES
"""
import csv
from typing import BinaryIO, List, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@article{RosenzweigCWSGM20_DCS_TISMIR,
    author    = {Sebastian Rosenzweig and Helena Cuesta and Christof Wei{\ss} and Frank Scherbaum and Emilia G{\'o}mez and Meinard M{\"u}ller},
    title     = {{D}agstuhl {ChoirSet}: {A} Multitrack Dataset for {MIR} Research on Choral Singing},
    journal   = {Transactions of the International Society for Music Information Retrieval ({TISMIR})},
    volume    = {3},
    number    = {1},
    year      = {2020},
    pages     = {98--110},
    publisher = {Ubiquity Press},
    doi       = {10.5334/tismir.48},
    url       = {http://doi.org/10.5334/tismir.48},
    url-demo  = {https://www.audiolabs-erlangen.de/resources/MIR/2020-DagstuhlChoirSet}
}
"""

INDEXES = {
    "default": "1.2.3",
    "test": "1.2.3",
    "1.2.3": core.Index(filename="dagstuhl_choirset_index_1.2.3.json"),
}

REMOTES = {
    "full_dataset": download_utils.RemoteFileMetadata(
        filename="DagstuhlChoirSet_V1.2.3.zip",
        url="https://zenodo.org/record/4618287/files/DagstuhlChoirSet_V1.2.3.zip?download=1",
        checksum="82b95faa634d0c9fc05c81e0868f0217",
        unpack_directories=["DagstuhlChoirSet_V1.2.3"],
    )
}

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Track(core.Track):
    """Dagstuhl ChoirSet Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_dyn_path (str): dynamic microphone audio path
        audio_hsm_path (str): headset microphone audio path
        audio_lrx_path (str): larynx microphone audio path
        f0_crepe_dyn_path (str): crepe f0 annotation for dynamic microphone path
        f0_crepe_hsm_path (str): crepe f0 annotation for headset microphone path
        f0_crepe_lrx_path (str): crepe f0 annotation for larynx microphone path
        f0_pyin_dyn_path (str): pyin f0 annotation for dynamic microphone path
        f0_pyin_hsm_path (str): pyin f0 annotation for headset microphone path
        f0_pyin_lrx_path (str): pyin f0 annotation for larynx microphone path
        f0_manual_lrx_path (str): manual f0 annotation for larynx microphone path
        score_path (str): score annotation path

    Cached Properties:
        f0_crepe_dyn (F0Data): algorithm-labeled (crepe) f0 annotations for dynamic microphone
        f0_crepe_hsn (F0Data): algorithm-labeled (crepe) f0 annotations for headset microphone
        f0_crepe_lrx (F0Data): algorithm-labeled (crepe) f0 annotations for larynx microphone
        f0_pyin_dyn (F0Data): algorithm-labeled (pyin) f0 annotations for dynamic microphone
        f0_pyin_hsn (F0Data): algorithm-labeled (pyin) f0 annotations for headset microphone
        f0_pyin_lrx (F0Data): algorithm-labeled (pyin) f0 annotations for larynx microphone
        f0_manual_lrx (F0Data): manually labeled f0 annotations for larynx microphone
        score (NoteData): time-aligned score representation
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id=track_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_dyn_path = self.get_path("audio_dyn")
        self.audio_hsm_path = self.get_path("audio_hsm")
        self.audio_lrx_path = self.get_path("audio_lrx")

        self.f0_crepe_dyn_path = self.get_path("f0_crepe_dyn")
        self.f0_crepe_hsm_path = self.get_path("f0_crepe_hsm")
        self.f0_crepe_lrx_path = self.get_path("f0_crepe_lrx")

        self.f0_pyin_dyn_path = self.get_path("f0_pyin_dyn")
        self.f0_pyin_hsm_path = self.get_path("f0_pyin_hsm")
        self.f0_pyin_lrx_path = self.get_path("f0_pyin_lrx")

        self.f0_manual_lrx_path = self.get_path("f0_manual_lrx")

        self.score_path = self.get_path("score")

    @core.cached_property
    def f0_crepe_dyn(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_crepe_dyn_path)

    @core.cached_property
    def f0_crepe_hsm(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_crepe_hsm_path)

    @core.cached_property
    def f0_crepe_lrx(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_crepe_lrx_path)

    @core.cached_property
    def f0_pyin_dyn(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_pyin_dyn_path)

    @core.cached_property
    def f0_pyin_hsm(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_pyin_hsm_path)

    @core.cached_property
    def f0_pyin_lrx(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_pyin_lrx_path)

    @core.cached_property
    def f0_manual_lrx(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_manual_lrx_path)

    @core.cached_property
    def score(self) -> Optional[annotations.NoteData]:
        return load_score(self.score_path)

    @property
    def audio_dyn(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the track's dynamic microphone (if available)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_dyn_path)

    @property
    def audio_hsm(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the track's headset microphone (if available)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_hsm_path)

    @property
    def audio_lrx(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the track's larynx microphone (if available)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_lrx_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""

        f0_data = [
            (self.f0_crepe_dyn, "crepe - DYN"),
            (self.f0_crepe_hsm, "crepe - HSM"),
            (self.f0_crepe_lrx, "crepe - LRX"),
            (self.f0_pyin_dyn, "pyin - DYN"),
            (self.f0_pyin_hsm, "pyin - HSM"),
            (self.f0_pyin_lrx, "pyin - LRX"),
            (self.f0_manual_lrx, "manual - LRX"),
        ]
        # remove missing annotations from the list
        f0_data = [tup for tup in f0_data if tup[1]]
        score_data = [(self.score, "score")] if self.score else None

        if self.audio_hsm_path:
            audio_path = self.audio_hsm_path
        elif self.audio_dyn_path:
            audio_path = self.audio_dyn_path
        else:
            audio_path = self.audio_lrx_path

        return jams_utils.jams_converter(
            audio_path=audio_path, f0_data=f0_data, note_data=score_data
        )


class MultiTrack(core.MultiTrack):
    """Dagstuhl ChoirSet multitrack class.

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/dagstuhl_choirset`

    Attributes:
        audio_stm_path (str): path to room mic (mono mixdown) audio file
        audio_str_path (str): path to room mic (right channel) audio file
        audio_stl_path (str): path to room mic (left channel) audio file
        audio_rev_path (str): path to room mic with artifical reverb (mono mixdown) audio file
        audio_spl_path (str): path to piano accompaniment (left channel) audio file
        audio_spr_path (str): path to piano accompaniement (right channel) audio file
        beat_path (str): path to beat annotation file

    Cached Properties:
        beat (annotations.BeatData): Beat annotation
        notes (annotations.NoteData): Note annotation
        multif0 (annotations.MultiF0Data): Aggregate of f0 annotations for tracks
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

        self.audio_stm_path = self.get_path("audio_stm")
        self.audio_str_path = self.get_path("audio_str")
        self.audio_stl_path = self.get_path("audio_stl")
        self.audio_rev_path = self.get_path("audio_rev")
        self.audio_spl_path = self.get_path("audio_spl")
        self.audio_spr_path = self.get_path("audio_spr")
        self.beat_path = self.get_path("beat")

    @property
    def track_audio_property(self):
        return "audio_dyn"

    @core.cached_property
    def beat(self) -> Optional[annotations.BeatData]:
        return load_beat(self.beat_path)

    @core.cached_property
    def notes(self) -> Optional[annotations.NoteData]:
        tracks_with_notes = [t for t in self.tracks.values() if t.score is not None]
        if len(tracks_with_notes) == 0:
            return None

        notes = tracks_with_notes[0].score
        if len(tracks_with_notes) > 1:
            for track in tracks_with_notes[1:]:
                notes += track.score
        return notes

    @core.cached_property
    def multif0(self) -> Optional[annotations.MultiF0Data]:
        f0_priority = [
            "f0_manual_lrx",
            "f0_crepe_lrx",
            "f0_pyin_lrx",
            "f0_crepe_hsm",
            "f0_pyin_hsm",
            "f0_crepe_dyn",
            "f0_pyin_dyn",
        ]
        multif0 = None
        for track in self.tracks.values():
            f0_data: Optional[annotations.F0Data] = None
            # get the best f0 annotation we can for this track
            for f0_attr in f0_priority:
                if getattr(track, f0_attr) is not None:
                    f0_data = getattr(track, f0_attr)
                    break

            if multif0 is None:
                multif0 = f0_data.to_multif0()  # type: ignore
            else:
                multif0 += f0_data

        return multif0

    @property
    def audio_stm(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the room mic (mono mixdown)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_stm_path)

    @property
    def audio_str(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the room mic (right channel)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_str_path)

    @property
    def audio_stl(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the room mic (left channel)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_stl_path)

    @property
    def audio_rev(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the room mic with artifical reverb (mono mixdown)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_rev_path)

    @property
    def audio_spl(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the piano accompaniment DI (left channel)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_spl_path)

    @property
    def audio_spr(self) -> Optional[Tuple[np.ndarray, float]]:
        """The audio for the piano accompaniment DI (right channel)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_spr_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""

        beat_data = [(self.beat, "beat")] if self.beat else None

        return jams_utils.jams_converter(audio_path=self.audio_stm_path, beat_data=beat_data)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Dagstuhl ChoirSet audio file.

    Args:
        audio_path (str): path pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=22050, mono=True)


@io.coerce_to_string_io
def load_f0(fhandle: TextIO) -> annotations.F0Data:
    """Load a Dagstuhl ChoirSet F0-trajectory.

    Args:
        fhandle (str or file-like): File-like object or path to F0 file

    Returns:
        F0Data Object - the F0-trajectory
    """
    times = []
    freqs = []
    voicings = []
    confs: List[Optional[float]]
    conf_array: Optional[np.ndarray]
    confs = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        times.append(float(line[0]))
        freq_val = float(line[1])
        voicings.append(float(freq_val > 0))
        freqs.append(np.abs(freq_val))
        if len(line) == 3:
            confs.append(float(line[2]))
        else:
            confs.append(None)

    if all([not c for c in confs]):
        conf_array = None
        conf_unit = None
    else:
        conf_array = np.array(confs)
        conf_unit = "likelihood"

    return annotations.F0Data(
        np.array(times),
        "s",
        np.array(freqs),
        "hz",
        np.array(voicings),
        "binary",
        conf_array,
        conf_unit,
    )


@io.coerce_to_string_io
def load_score(fhandle: TextIO) -> annotations.NoteData:
    """Load a Dagstuhl ChoirSet time-aligned score representation.

    Args:
        fhandle (str or file-like): File-like object or path to score representation file

    Returns:
        NoteData Object - the time-aligned score representation
    """
    intervals = []
    notes = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        intervals.append([float(line[0]), float(line[1])])
        notes.append(float(line[2]))

    return annotations.NoteData(np.array(intervals), "s", librosa.midi_to_hz(notes), "hz")


@io.coerce_to_string_io
def load_beat(fhandle: TextIO) -> annotations.BeatData:
    """Load a Dagstuhl ChoirSet beat annotation.

    Args:
        fhandle (str or file-like): File-like object or path to beat annotation file

    Returns:
        BeatData Object - the beat annotation
    """
    times = []
    positions = []
    position = 0
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        times.append(float(line[0]))
        raw_position = float(line[1])
        if np.floor(raw_position) == raw_position:
            position = 1
        else:
            position += 1
        positions.append(position)

    return annotations.BeatData(np.array(times), "s", np.array(positions), "bar_index")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Dagstuhl ChoirSet dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="dagstuhl_choirset",
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.dagstuhl_choirset.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.dagstuhl_choirset.load_f0", version="0.3.4")
    def load_f0(self, *args, **kwargs):
        return load_f0(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.dagstuhl_choirset.load_score", version="0.3.4")
    def load_score(self, *args, **kwargs):
        return load_score(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.dagstuhl_choirset.load_beat", version="0.3.4")
    def load_beat(self, *args, **kwargs):
        return load_beat(*args, **kwargs)
