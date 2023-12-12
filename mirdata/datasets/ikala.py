"""IKala Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The iKala dataset is comprised of 252 30-second excerpts sampled from 206 iKala
    songs (plus 100 hidden excerpts reserved for MIREX).
    The music accompaniment and the singing voice are recorded at the left and right
    channels respectively and can be found under the Wavfile directory.
    In addition, the human-labeled pitch contours and timestamped lyrics can be
    found under PitchLabel and Lyrics respectively.

    For more details, please visit: http://mac.citi.sinica.edu.tw/ikala/
"""
import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{chan2015vocal,
    title={Vocal activity informed singing voice separation with the iKala dataset},
    author={Chan, Tak-Shing and Yeh, Tzu-Chun and Fan, Zhe-Cheng and Chen, Hung-Wei and Su, Li and Yang, Yi-Hsuan and
    Jang, Roger},
    booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={718--722},
    year={2015},
    organization={IEEE}
}"""

INDEXES = {
    "default": "2.0",
    "test": "2.0",
    "1.0": core.Index(filename="ikala_index_1.0.json", partial_download=["metadata"]),
    "2.0": core.Index(filename="ikala_index_2.0.json"),
}

TIME_STEP = 0.032  # seconds
REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="id_mapping.txt",
        url="http://mac.citi.sinica.edu.tw/ikala/id_mapping.txt",
        checksum="81097b587804ce93e56c7a331ba06abc",
    ),
    "notes_pyin": download_utils.RemoteFileMetadata(
        filename="ikala-pyin-notes.zip",
        url="https://zenodo.org/record/4728756/files/ikala-pyin-notes.zip?download=1",
        checksum="8b9464a48b11bf26762de9c18a3a51ea",
    ),
}

DOWNLOAD_INFO = """
    Unfortunately most of the iKala dataset is not available for download.
    If you have the iKala dataset, place the contents into a folder called
    iKala with the following structure:
        > iKala/
            > Lyrics/
            > PitchLabel/
            > Wavfile/
    and copy the iKala folder to {}
"""

LICENSE_INFO = """
When it was distributed, Ikala used to have a custom license.
Visit http://mac.citi.sinica.edu.tw/ikala/ for more details.
"""


class Track(core.Track):
    """Ikala Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        f0_path (str): path to the track's f0 annotation file
        notes_pyin_path (str): path to the note annotation file
        lyrics_path (str): path to the track's lyric annotation file
        section (str): section. Either 'verse' or 'chorus'
        singer_id (str): singer id
        song_id (str): song id of the track
        track_id (str): track id

    Cached Properties:
        f0 (F0Data): human-annotated singing voice pitch
        notes_pyin (NoteData): notes estimated by the pyin algorithm
        lyrics (LyricsData): human-annotated lyrics
        pronunciations (LyricsData): human-annotation lyric pronunciations
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.f0_path = self.get_path("pitch")
        self.lyrics_path = self.get_path("lyrics")
        self.notes_pyin_path = self.get_path("notes_pyin")

        self.audio_path = self.get_path("audio")

        self.song_id = track_id.split("_")[0]
        self.section = track_id.split("_")[1]

    @property
    def singer_id(self):
        return self._track_metadata.get(self.song_id)

    @core.cached_property
    def f0(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_path)

    @core.cached_property
    def notes_pyin(self) -> Optional[annotations.NoteData]:
        return load_notes(self.notes_pyin_path)

    @core.cached_property
    def lyrics(self) -> Optional[annotations.LyricData]:
        return load_lyrics(self.lyrics_path)

    @core.cached_property
    def pronunciations(self) -> Optional[annotations.LyricData]:
        return load_pronunciations(self.lyrics_path)

    @property
    def vocal_audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """Solo vocal audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_vocal_audio(self.audio_path)

    @property
    def instrumental_audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """Instrumental audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_instrumental_audio(self.audio_path)

    @property
    def mix_audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """Mixture audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_mix_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.f0, None)],
            note_data=[(self.notes_pyin, "pyin estimated notes")],
            lyrics_data=[(self.lyrics, None)],
            metadata={
                "section": self.section,
                "singer_id": self.singer_id,
                "track_id": self.track_id,
                "song_id": self.song_id,
            },
        )


@io.coerce_to_bytes_io
def load_vocal_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load ikala vocal audio.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate
    """
    audio, sr = librosa.load(fhandle, sr=None, mono=False)
    vocal_channel = audio[1, :]
    return vocal_channel, sr


@io.coerce_to_bytes_io
def load_instrumental_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load ikala instrumental audio.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate
    """
    audio, sr = librosa.load(fhandle, sr=None, mono=False)
    instrumental_channel = audio[0, :]
    return instrumental_channel, sr


@io.coerce_to_bytes_io
def load_mix_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load an ikala mix.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate
    """
    mixed_audio, sr = librosa.load(fhandle, sr=None, mono=True)
    # multipy by 2 because librosa averages the left and right channel.
    return 2.0 * mixed_audio, sr


@io.coerce_to_string_io
def load_f0(fhandle: TextIO) -> annotations.F0Data:
    """Load an ikala f0 annotation.

    Args:
        fhandle (str or file-like): File-like object or path to f0 annotation file

    Raises:
        IOError: If f0_path does not exist

    Returns:
        F0Data: the f0 annotation data
    """
    lines = fhandle.readlines()
    f0_midi = np.array([float(line) for line in lines])
    f0_hz = librosa.midi_to_hz(f0_midi) * (f0_midi > 0)
    voicing = (f0_hz > 0).astype(float)
    f0_hz = np.abs(f0_hz)
    times = (np.arange(len(f0_midi)) * TIME_STEP) + (TIME_STEP / 2.0)
    f0_data = annotations.F0Data(times, "s", f0_hz, "hz", voicing, "binary")
    return f0_data


@io.coerce_to_string_io
def load_notes(fhandle: TextIO) -> Optional[annotations.NoteData]:
    """Load a note annotation file.

    Args:
        fhandle (str or file-like): str or file-like to note annotation file

    Raises:
        IOError: if file doesn't exist

    Returns:
        NoteData: note annotation
    """
    intervals = []
    freqs = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        start_time = float(line[0])
        intervals.append([start_time, start_time + float(line[1])])
        freqs.append(float(line[2]))

    return annotations.NoteData(np.array(intervals), "s", np.array(freqs), "hz")


@io.coerce_to_string_io
def load_lyrics(fhandle: TextIO) -> annotations.LyricData:
    """Load an ikala lyrics annotation.

    Args:
        fhandle (str or file-like): File-like object or path to lyric annotation file

    Raises:
        IOError: if lyrics_path does not exist

    Returns:
        LyricData: lyric annotation data
    """
    # input: start time (ms), end time (ms), lyric, [pronunciation]
    reader = csv.reader(fhandle, delimiter=" ")
    start_times = []
    end_times = []
    lyrics = []
    pronunciations = []
    for line in reader:
        start_times.append(float(line[0]) / 1000.0)
        end_times.append(float(line[1]) / 1000.0)
        lyrics.append(line[2])
        if len(line) > 2:
            pronunciation = " ".join(line[3:])
            pronunciations.append(pronunciation)
        else:
            pronunciations.append("")

    lyrics_data = annotations.LyricData(np.array([start_times, end_times]).T, "s", lyrics, "words")
    return lyrics_data


@io.coerce_to_string_io
def load_pronunciations(fhandle: TextIO) -> annotations.LyricData:
    """Load an ikala pronunciation annotation.

    Args:
        fhandle (str or file-like): File-like object or path to lyric annotation file

    Raises:
        IOError: if lyrics_path does not exist

    Returns:
        LyricData: pronunciation annotation data
    """
    # input: start time (ms), end time (ms), lyric, [pronunciation]
    reader = csv.reader(fhandle, delimiter=" ")
    start_times = []
    end_times = []
    pronunciations = []
    for line in reader:
        start_times.append(float(line[0]) / 1000.0)
        end_times.append(float(line[1]) / 1000.0)
        if len(line) > 2:
            pronunciation = " ".join(line[3:])
            pronunciations.append(pronunciation)
        else:
            pronunciations.append("")

    lyrics_data = annotations.LyricData(
        np.array([start_times, end_times]).T, "s", pronunciations, "pronunciations_open"
    )
    return lyrics_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The ikala dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="ikala",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        id_map_path = os.path.join(self.data_home, "id_mapping.txt")
        try:
            with open(id_map_path, "r") as fhandle:
                reader = csv.reader(fhandle, delimiter="\t")
                singer_map = {}
                for line in reader:
                    if line[0] == "singer":
                        continue
                    singer_map[line[1]] = line[0]
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return singer_map

    @deprecated(reason="Use mirdata.datasets.ikala.load_vocal_audio", version="0.3.4")
    def load_vocal_audio(self, *args, **kwargs):
        return load_vocal_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.ikala.load_instrumental_audio", version="0.3.4")
    def load_instrumental_audio(self, *args, **kwargs):
        return load_instrumental_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.ikala.load_mix_audio", version="0.3.4")
    def load_mix_audio(self, *args, **kwargs):
        return load_mix_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.ikala.load_f0", version="0.3.4")
    def load_f0(self, *args, **kwargs):
        return load_f0(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.ikala.load_notes", version="0.3.4")
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.ikala.load_lyrics", version="0.3.4")
    def load_lyrics(self, *args, **kwargs):
        return load_lyrics(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.ikala.load_pronunciations", version="0.3.4")
    def load_pronunciations(self, *args, **kwargs):
        return load_pronunciations(*args, **kwargs)
