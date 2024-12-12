"""MIR-1K Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    MIR-1K is a dataset of 110 amateur karaoke recordings of Chinese pop songs, performed by a total of 8 female and 11 male singers. The total duration of the dataset is 133 minutes.

    The recordings are split into 1000 stereo clips ranging from 4 to 13 seconds in length. Each clip contains the accompaniment on the left channel and the isolated singing voice on the right channel. The clips manually annotated with pitch contours, voicing labels and lyrics. Additionally, unvoiced frames are labeled as belonging to one of 5 categories:

    1. unvoiced stop
    2. unvoiced fricative and affricate
    3. /h/
    4. inhaling sound
    5. other (including voiced frames and frames without any vocal activity)

    The dataset was collected by the MIR Lab at National Taiwan University and first published in [1].

    The MIR Lab offers the dataset for download on their website [2] without specifying a license.

    [1] Hsu, Chao-Ling and Jang, Jyh-Shing Roger. "On the Improvement of Singing Voice Separation for Monaural Recordings Using the MIR-1K Dataset." IEEE Transactions on Audio, Speech, and Language Processing (2010)

    [2] http://mirlab.org/dataset/public
"""

from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, jams_utils, io

BIBTEX = """
@article{hsu2010improvement,
  author={Hsu, Chao-Ling and Jang, Jyh-Shing Roger},
  journal={IEEE Transactions on Audio, Speech, and Language Processing}, 
  title={On the Improvement of Singing Voice Separation for Monaural Recordings Using the MIR-1K Dataset}, 
  year={2010},
  volume={18},
  number={2},
  pages={310-319},
  doi={10.1109/TASL.2009.2026503}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="mir_1k_index_1.0.json",
        url="https://zenodo.org/records/14428193/files/mir_1k_index_1.0.json?download=1",
        checksum="c93b0e9145d590f2cff86d02aa2ee855",
    ),
    "sample": core.Index(filename="mir_1k_index_1.0_sample.json"),
}

REMOTES = {
    "mirlab": download_utils.RemoteFileMetadata(
        filename="MIR-1K.zip",
        url="http://mirlab.org/dataset/public/MIR-1K.zip",
        checksum="3a4e5acd740110ae481a836460dabb6a",
        unpack_directories=["MIR-1K"],
    )
}

LICENSE_INFO = "Unknown"

UNVOICED_LABELS = {
    "1": "unvoiced stop",
    "2": "unvoiced fricative and affricate",
    "3": "/h/",
    "4": "inhaling sound",
    "5": "other",
}


class Track(core.Track):
    """mir_1k Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        f0_path (str): path to the track's f0 annotation file
        lyrics_path (str): path to the track's lyrics file
        track_id (str): track id
        unvoiced_labels_path (str): path to the track's unvoiced labels
        vocal_activity_path (str): path to the track's vocal activity labels

    Cached Properties:
        f0 (F0Data): the track's f0 annotation
        lyrics (LyricData): the track's lyrics
        unvoiced_labels (EventData): the track's unvoiced labels, as defined in mir_1k.UNVOICED_LABELS
        vocal_activity (EventData): the track's vocal activity labels: "0" for non-vocal, "1" for vocal

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.f0_path = self.get_path("f0")
        self.lyrics_path = self.get_path("lyrics")
        self.vocal_activity_path = self.get_path("vocal-flag")
        self.unvoiced_labels_path = self.get_path("unvoiced-category")
        self.audio_path = self.get_path("audio")

    @core.cached_property
    def f0(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_path)

    @core.cached_property
    def lyrics(self) -> Optional[annotations.LyricData]:
        return load_lyrics(self.lyrics_path)

    @core.cached_property
    def vocal_activity(self) -> Optional[annotations.EventData]:
        return load_event_labels(self.vocal_activity_path)

    @core.cached_property
    def unvoiced_labels(self) -> Optional[annotations.EventData]:
        return load_event_labels(self.unvoiced_labels_path)

    @property
    def instrumental_audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """accompaniment audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_instrumental_audio(self.audio_path)

    @property
    def vocal_audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """solo vocal audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_vocal_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        jams = jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.f0, None)],
            event_data=[
                (self.vocal_activity, "Vocal Activity Labels"),
                (self.unvoiced_labels, "Unvoiced Category Labels"),
            ],
            lyrics_data=[(self.lyrics, None)],
            metadata={
                "track_id": self.track_id,
            },
        )

        # manual midi to hz conversion
        # jams format expects Hz pitches, but jams_utils.f0s_to_jams does not take frequency_unit into account
        for f0 in jams.search(namespace="pitch_contour")[0]["data"]:
            if f0.value["frequency"] > 0:
                f0.value["frequency"] = librosa.midi_to_hz(f0.value["frequency"])

        return jams


def frame_timestamps(n_frames: int, hop_size_s=0.02) -> np.ndarray:
    """Generate timestamps for MIR-1K annotation frames."""

    return hop_size_s * np.arange(1, n_frames + 1)


@io.coerce_to_bytes_io
def load_vocal_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load MIR-1K vocal audio

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate

    """
    audio, sr = librosa.load(fhandle, sr=None, mono=False)
    return audio[1, :], sr


@io.coerce_to_bytes_io
def load_instrumental_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load MIR-1K instrumental audio

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate

    """
    audio, sr = librosa.load(fhandle, sr=None, mono=False)
    return audio[0, :], sr


@io.coerce_to_string_io
def load_f0(fhandle: TextIO) -> annotations.F0Data:
    """Load a MIR-1K f0 annotation

    Args:
        fhandle (str or file-like): File-like object or path to f0 annotation file

    Raises:
        IOError: If f0_path does not exist

    Returns:
        F0Data: the f0 annotation data

    """
    f0 = np.genfromtxt(fhandle)
    assert f0.ndim == 1

    times = frame_timestamps(len(f0))

    return annotations.F0Data(
        times=times,
        time_unit="s",
        frequencies=f0,
        frequency_unit="midi",
        voicing=(f0 > 0).astype(np.float64),
        voicing_unit="binary",
    )


# not using @io.coerce_to_string_io here because it has hardcoded utf-8 encoding
def load_lyrics(path: str) -> Optional[annotations.LyricData]:
    """Read lyrics from a file. MIR-1K lyrics do not include timestamps and are written in traditional Chinese characters stored in Big5-hkscs encoding.

    Args:
        path (str): path to lyric annotation file

    Raises:
        IOError: if lyrics_path does not exist

    Returns:
        str: the lyrics

    """

    return annotations.LyricData(
        intervals=np.array([[0.0, 0.0]]),
        interval_unit="s",
        lyrics=[open(path, encoding="big5-hkscs").read().strip()],
        lyric_unit="words",
    )


def load_event_labels(fhandle: TextIO) -> Optional[annotations.EventData]:
    """Read a MIR-1K annotation and generate a EventData object

    Args:
        fhandle (str or file-like): File-like object or path to annotation file

    Raises:
        IOError: if fhandle does not exist

    Returns:
        EventData: the event annotation data

    """
    labels = np.genfromtxt(fhandle, dtype=str)
    times = frame_timestamps(len(labels))

    # convert to intervals
    intervals = []
    events = []

    current_label = None
    start_time = times[0]
    for t, label in zip(times, labels):
        if label != current_label:
            if current_label is not None:
                intervals.append((start_time, t))
                events.append(current_label)
            start_time = t
            current_label = label

    # closing final interval
    intervals.append((start_time, times[-1]))
    events.append(current_label)

    return annotations.EventData(
        np.array(intervals),
        "s",
        events,
        "open",
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The MIR-1K dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="mir_1k",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )
