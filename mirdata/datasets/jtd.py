"""Jazz Trio Database (JTD) Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Jazz Trio Database (JTD) is a dataset comprising 1,294 multitrack jazz performances (about 45 hours total)
    annotated by an automated signal processing pipeline. All performances are commercial recordings of jazz piano
    trios, comprising acoustic piano, upright bass, and drum kit, and are broadly in the "straight-ahead" jazz style.

    Its purpose is to serve as a reference database for the design, evaluation, and implementation of various music
    information retrieval systems related to jazz and improvised music, including (but not limited to) onset detection,
    beat tracking, automatic music transcription, and performer identification.

    For every performance, the following audio files are included:

    1) the "raw" audio from the piano solo, typically including piano, bass, and drums (stereo, 44.1 kHz)
        - for some performances, individual audio files for the left and right stereo channels are also included
    2) unmixed piano audio obtained by applying a music source separation model to the "raw" audio
    3) unmixed bass audio
    4) unmixed drums audio

    For the "raw" audio, there are the following annotations:

    1) Beat timestamps for the start of each quarter note
    2) Downbeat annotations for the start of each bar

    For the three "unmixed" audio files, there are the following annotations:

    1) MIDI transcription (frame-level, currently piano only)
    2) Onset timestamps
    3) Beat-matched onsets

    To "match" onsets in the unmixed audio and beats in the "raw" audio, a window of -32nd/+16th note is applied to
    every beat timestamp, and the nearest onset from every unmixed audio file is taken as the "match". In cases where
    no onsets are contained inside the window, the beat is set to "missing" in the data, such that the number of
    beat-matched onsets is always the same as the number of beats.

    Finally, there are the following piece-level annotations:

    1) Tempo, in quarter-note beats-per-minute
    2) Time signature (either three or four quarter-note beats)
    3) Timestamps for the duration of the piano solo within the performance
    4) Metadata (e.g., recording year, performer names, album title)

    The JTD was created by researchers at the Centre for Music & Science, University of Cambridge, as part of Huw
    Cheston's PhD research, during the period 2023-2024.

    The audio data is not publicly available and access must be requested on Zenodo. The annotations and metadata are
    freely available. The database is made available for research and educational purposes under the MIT license
    (https://github.com/HuwCheston/Jazz-Trio-Database/blob/main/LICENSE).

    For more details, please visit our GitHub repository (https://github.com/HuwCheston/Jazz-Trio-Database/) or
    our TISMIR publication (https://doi.org/10.5334/tismir.186).

"""

import csv
import json
from typing import BinaryIO, Optional, TextIO, Tuple
from smart_open import open

import librosa
import numpy as np

from mirdata import download_utils, jams_utils, core, annotations, io


BIBTEX = """
@article{jazz-trio-database
    title = {Jazz Trio Database: Automated Annotation of Jazz Piano Trio Recordings Processed Using Audio Source Separation},
    url = {https://doi.org/10.5334/tismir.186},
    doi = {10.5334/tismir.186},
    publisher = {Transactions of the International Society for Music Information Retrieval},
    author = {Cheston, Huw and Schlichting, Joshua L and Cross, Ian and Harrison, Peter M C},
    year = {2024},
}
"""

INDEXES = {
    "default": "2",
    "test": "sample",
    "2": core.Index(
        filename="jtd_index_2.0.json",
        url="https://raw.githubusercontent.com/HuwCheston/Jazz-Trio-Database/refs/heads/main/references/jtd_index_2.0.json",
        checksum="cdf32f3751fd9d0351c45b6b94f91a0f",
    ),
    "sample": core.Index(filename="jtd_index_2.0_sample.json"),
}

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="annotation.zip",
        url="https://github.com/HuwCheston/Jazz-Trio-Database/releases/download/v02-zenodo/jazz-trio-database-v02.zip",
        checksum="43f543fb286c6222ae1f52bcf7561f37",
        destination_dir="annotations",
        unpack_directories=[
            "jazz-trio-database-v02"
        ],  # removes a redundant extra subdirectory
    )
}

DOWNLOAD_INFO = """
To download the audio for files for JTD, visit: https://zenodo.org/records/13828030 and request access.

After you've been granted access, press the "Download all" button on the Zenodo record.

This will create a new file named files-archive (with no extension). Rename the file to files-archive.zip and extract 
using any unzipping tool (7zip, WinRAR, the unarchiver) or the command line. This will give you a list of multi-part 
zip files in the form [processed.zip.001, processed.zip.002, ...] and [raw.zip.001, raw.zip.002, ...]. 

To extract these, use 7zip from the command line:

```
7z x processed.zip.001
7z x raw.zip.001
```

Note that the default `unzip` command on Linux can't handle these files, so you'll need to use 7zip. You may also be 
able to use a GUI tool like WinRAR, which was used to create the archive in the first place. 

These commands will extract the audio to the current folder. You'll then need to move the results to {0}/processed and 
{0}/raw, respectively, creating these folders if they don't already exist.

Combined with the annotation files (which can be obtained by calling `.download()` on the `mirdata.Dataset` instance 
you've just initialized), the end result should be a file structure that looks like:

```
{0}
├─ raw
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067.wav    # one to three audio files per performance
│  ├─ ...
├─ processed
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano.wav     # always three audio files per performance
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067_bass.wav
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067_drums.wav
│  ├─ ...
├─ annotations
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067    # one folder per performance
│  │  ├─ bass_onsets.csv
│  │  ├─ beats.csv
│  │  ├─ ...
│  ├─ barronk-beautifullove-mrazgrileyb-2009-c87abfa6
│  ├─ ...
```

"""

LICENSE_INFO = """

The MIT License (MIT)
Copyright (c) 2023, Huw Cheston

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


class Track(core.Track):
    """JTD track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to audio file
        onsets_path (str): path to onsets file
        midi_path (str): path to MIDI file
        beats_path (str): path to beats file
        instrument (str): name of the instrument for this track, either "piano", "bass", or "drums"

    Properties:
        audio(tuple): audio signal and sample rate for the isolated instrument track of this performance
        musician (str): name of the musician playing the `instrument` on this track

    Cached Properties:
        beats (BeatData): beat times for this instrument
        onsets (EventData): onset and offset times
        midi (NoteData): midi pitches, onset, offset times, and velocities

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=lambda: json.load(open(self.get_path("metadata"), "r")),
        )

        self.audio_path = self.get_path("audio")
        self.onsets_path = self.get_path("onsets")
        self.midi_path = self.get_path("midi")
        self.beats_path = self.get_path("beats")
        self.instrument = self.track_id.split("_")[1]

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The source-separated audio for this instrument

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        """The times of onsets by this musician matched to the nearest quarter-note beat timestamp

        Returns:
            * annotations.BeatData - timestamp, beat number (1-indexed to bar)

        """
        # This maps instrument names onto columns of the CSV file
        column_mapping = {"piano": 1, "bass": 2, "drums": 3}
        instrument = self.instrument
        return load_beats(self.beats_path, column_mapping[instrument])

    @core.cached_property
    def midi(self) -> Optional[annotations.NoteData]:
        """The MIDI for this instrument

        Returns:
            * annotations.NoteData

        """
        if self.midi_path is None:
            return None
        return io.load_notes_from_midi(self.midi_path)  # returns None if no MIDI

    @property
    def musician(self) -> str:
        """The name of the musician playing on this track

        Returns:
            * str - name of musician

        """
        # The `musicians` dictionary has a different mapping to the `instruments` one
        instruments_and_roles = {
            "piano": "pianist",
            "bass": "bassist",
            "drums": "drummer",
        }
        # This maps e.g. "piano" -> "pianist", "bass" -> "bassist
        current_role = instruments_and_roles[self.instrument]
        return self._track_metadata["musicians"][current_role]

    @core.cached_property
    def onsets(self) -> Optional[annotations.EventData]:
        """The onsets for this instrument

        Returns:
            * annotations.EventData

        """
        return load_onsets(self.onsets_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.beats, "beats")],
            note_data=[(self.midi, "midi")],
            event_data=[(self.onsets, "onsets")],
            metadata=dict(instrument=self.instrument, musician=self.musician),
        )


class MultiTrack(core.MultiTrack):
    """JTD multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/jtd`

    Attributes:
        album (str): The name of the album that this performance was taken from.
        audio (Tuple[np.ndarray, float]): The track's audio, center channel.
        audio_lchan (Tuple[np.ndarray, float]): The track's audio, left channel (if available).
        audio_rchan (Tuple[np.ndarray, float]): The track's audio, right channel (if available).
        bandleader (str): The full name of the bandleader who led the recording session.
        bass (Track): The associated bass track for this recording.
        drums (Track): The associated drums track for this recording.
        duration (float): The duration of the piano solo in seconds.
        jtd_300 (bool): Whether the track is contained in the smaller JTD-300 subset of 300 recordings.
        mtrack_id (str): track id
        musicbrainz_id (str): The MusicBrainz ID for the recording.
        name (str): The track's name.
        piano (Track): The associated piano track for this recording.
        start (int): The start of the piano solo relative to the full recording (in seconds).
        stop (int): The end of the piano solo relative to the full recording (in seconds).
        tempo (float): The average tempo of the track in beats per minute.
        time_signature (int): The time signature of the recording (3 or 4 quarter-note beats).
        tracks (dict): Dictionary of track IDs and `Track` instances
        year (int): The year the recording was made.

    Cached Properties:
        beats (annotations.BeatData): The times of quarter-note beats for the recording.

    """

    def __init__(
        self, mtrack_id, data_home, dataset_name, index, track_class, metadata
    ):
        super().__init__(
            mtrack_id=mtrack_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            track_class=track_class,
            metadata=lambda: json.load(open(self.get_path("metadata"), "r")),
        )

        self.audio_path = self.get_path("audio")
        self.audio_lchan_path = self.get_path("audio-lchan")
        self.audio_rchan_path = self.get_path("audio-rchan")
        self.beats_path = self.get_path("beats")

    @property
    def album(self) -> str:
        """The name of the album that this performance was taken from

        Returns:
            * str - name of the album

        """
        return self._multitrack_metadata["album_name"]

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio, center channel

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def audio_lchan(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio, left channel (not present for all tracks)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_lchan_path)

    @property
    def audio_rchan(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio, right channel (not present for all tracks)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_rchan_path)

    @property
    def bandleader(self) -> str:
        """The full name of the bandleader who led the recording session

        Returns:
            * str - name of the bandleader

        """
        return self._multitrack_metadata["bandleader"]

    @property
    def bass(self) -> Track:
        """The associated bass track for this recording

        Returns:
            * Track

        """
        return self.tracks[self.mtrack_id + "_bass"]

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        """The times of quarter-note beats for the recording

        Returns:
            * annotations.BeatData - timestamp, beat number (1-indexed to bar)

        """
        return load_beats(self.beats_path, 0)

    @property
    def drums(self) -> Track:
        """The associated drums track for this recording

        Returns:
            * Track

        """
        return self.tracks[self.mtrack_id + "_drums"]

    @property
    def duration(self) -> int:
        """The duration of the piano solo

        Returns:
            * float - solo duration (in seconds)

        """
        start = self._multitrack_metadata["timestamps"]["start"]
        stop = self._multitrack_metadata["timestamps"]["end"]
        return timestamp_to_seconds(stop) - timestamp_to_seconds(start)

    @property
    def jtd_300(self) -> bool:
        """Whether the track is contained in the smaller JTD-300 subset of 300 recordings

        Returns:
            * bool - True if contained in JTD-300, otherwise false

        """
        return self._multitrack_metadata["in_30_corpus"]

    @property
    def musicbrainz_id(self) -> str:
        """The MusicBrainz ID for the recording

        Returns:
            * str - musicbrainz ID

        """
        return self._multitrack_metadata["mbz_id"]

    @property
    def name(self) -> str:
        """The track's name

        Returns:
            * str - track name

        """
        return self._multitrack_metadata["track_name"]

    @property
    def piano(self) -> Track:
        """The associated piano track for this recording

        Returns:
            * Track

        """
        return self.tracks[self.mtrack_id + "_piano"]

    @property
    def start(self) -> int:
        """The start of the piano solo relative to the full recording

        Returns:
            * int - start of performance, in seconds

        """
        return timestamp_to_seconds(self._multitrack_metadata["timestamps"]["start"])

    @property
    def stop(self) -> int:
        """The end of the piano solo relative to the full recording

        Returns:
            * int - end of performance, in seconds

        """
        return timestamp_to_seconds(self._multitrack_metadata["timestamps"]["end"])

    @property
    def tempo(self) -> float:
        """The average tempo of the track

        Returns:
            * float - the tempo, in beats-per-minute

        """
        return float(self._multitrack_metadata["tempo"])

    @property
    def time_signature(self) -> int:
        """The time signature of the recording, either 3 or 4 quarter-note beats

        Returns:
            * int - time signature

        """
        return int(self._multitrack_metadata["time_signature"])

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.beats, "beats")],
            tempo_data=[(self.tempo, "tempo")],
            metadata=dict(
                album=self.album,
                bandleader=self.bandleader,
                duration=self.duration,
                jtd_300=self.jtd_300,
                musicbrainz_id=self.musicbrainz_id,
                name=self.name,
                start=self.start,
                stop=self.stop,
                time_signature=self.time_signature,
                year=self.year,
            ),
        )

    @property
    def track_audio_property(self):
        return "audio"

    @property
    def year(self) -> Optional[int]:
        """The year the recording was made

        Returns:
            * int - recording year

        """
        return int(self._multitrack_metadata["recording_year"])


def timestamp_to_seconds(ts: str) -> int:
    """Coerces timestamp in form `%M:%S` or `%H:%M:S` to an integer"""
    # Timestamp is in format hours-minutes-seconds
    if ts.count(":") == 2:
        hours, minutes, seconds = map(int, ts.split(":"))
        return int((hours * 60 * 60) + (minutes * 60) + seconds)
    # Timestamp is in format minutes-seconds
    elif ts.count(":") == 1:
        minutes, seconds = map(int, ts.split(":"))
        return int((minutes * 60) + seconds)
    # Timestamp is in incorrect format
    else:
        raise ValueError("Timestamp must be in format %H:%M:%S or %M:%S")


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a JTD audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=44100, mono=True)


@io.coerce_to_string_io
def load_onsets(fhandle: TextIO) -> Optional[annotations.EventData]:
    """Load a JTD onset file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an onset csv file

    Returns:
        * annotations.EventData - the onset data

    """
    reader: list = list(csv.reader(fhandle))
    # Flatten list and evaluate items as floats
    reader = [float(x) for xs in reader for x in xs]
    intervals = []
    annotation = []
    # Iterate over successive onset times
    for line_num, (line1, line2) in enumerate(zip(reader, reader[1:])):
        # Creates a list of (onset1, onset2), (onset2, onset3), ... (i.e., onset and offset times)
        intervals.append([line1, line2])
        # This is just the count of onsets, 0-indexed
        annotation.append(str(line_num))
    # Needs to be an array to pass validation
    intervals_arr = np.array(intervals)
    return annotations.EventData(intervals_arr, "s", annotation, "open")


# no decorator because of https://github.com/mir-dataset-loaders/mirdata/issues/503
#  see also https://github.com/mir-dataset-loaders/mirdata/pull/652#discussion_r1927327838
def load_beats(fhandle: TextIO, col_idx: int) -> Optional[annotations.BeatData]:
    """Load a JTD beat file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a beat csv file
        col_idx (int, optional): index of the column to use (0=overall, 1=piano, 2=bass, 3=drums), defaults to 0

    Returns:
        * annotations.BeatData - the beat data

    """
    try:
        with open(fhandle, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            reader.__next__()  # The first line of the CSV is always a header so we can just skip it
            timestamps, positions = [], []
            # Iterating over each line of the CSV file (i.e., each 'beat')
            for beat_number, beat, piano, bass, drums, metre in reader:
                # Get the required data from the row
                desired_data = [beat, piano, bass, drums][col_idx]
                # Coerce empty strings to NaN values
                if desired_data == "":
                    desired_data_fmt = np.nan
                else:
                    desired_data_fmt = float(desired_data)
                # Append everything to the list with the required datatypes
                timestamps.append(desired_data_fmt)
                positions.append(
                    int(float(metre))
                )  # coerce string to float and then to int
    except FileNotFoundError:
        raise FileNotFoundError("beats path {} does not exist".format(fhandle))
    else:
        return annotations.BeatData(
            np.array(timestamps), "s", np.array(positions), "bar_index"
        )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Jazz Trio Database.
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="jtd",
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )
        # Note: this needs to be added to pass `test_full_dataset.test_load_mtracks`
        #  currently this test fails on many other datasets, e.g., `dagstuhl_choirset`
        #  as the `mm` attribute is never set when calling `super().__init__()`
        self.mm = list(self._index["multitracks"].keys())
