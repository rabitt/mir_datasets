"""RWC Classical Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Classical Music Database consists of 50 pieces

    * Symphonies: 4 pieces
    * Concerti: 2 pieces
    * Orchestral music: 4 pieces
    * Chamber music: 10 pieces
    * Solo performances: 24 pieces
    * Vocal performances: 6 pieces

    **A note about the Beat annotations:**

    - 48 corresponds to the duration of a quarter note (crotchet)
    - 24 corresponds to the duration of an eighth note (quaver)
    - 384 corresponds to the position of a downbeat

    In 4/4 time signature, they correspond as follows:

    .. code-block:: latex

        384: 1st beat in a measure (i.e., downbeat position)
        48: 2nd beat
        96: 3rd beat
        144 4th beat

    In 3/4 time signature, they correspond as follows:

    .. code-block:: latex

        384: 1st beat in a measure (i.e., downbeat position)
        48: 2nd beat
        96: 3rd beat

    In 6/8 time signature, they correspond as follows:

    .. code-block:: latex

        384: 1st beat in a measure (i.e., downbeat position)
        24: 2nd beat
        48: 3rd beat
        72: 4th beat
        96: 5th beat
        120: 6th beat

    For more details, please visit: https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-c.html
"""
import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="rwc_classical_index_1.0.json"),
}

REMOTES = {
    "annotations_beat": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-C-2001.BEAT.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.BEAT.zip",
        checksum="e8ee05854833cbf5eb7280663f71c29b",
        destination_dir="annotations",
    ),
    "annotations_sections": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-C-2001.CHORUS.zip",
        url=(
            "https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.CHORUS.zip"
        ),
        checksum="f77bd527510376f59f5a2eed8fd7feb3",
        destination_dir="annotations",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="master.zip",
        url="https://github.com/magdalenafuentes/metadata/archive/master.zip",
        checksum="7dbe87fedbaaa1f348625a2af1d78030",
    ),
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the RWC-Classical dataset are not available
    for download. If you have the RWC-Classical dataset, place the contents into a
    folder called RWC-Classical with the following structure:
        > RWC-Classical/
            > annotations/
            > audio/rwc-c-m0i with i in [1 .. 6]
            > metadata-master/
    and copy the RWC-Classical folder to {}
"""

LICENSE_INFO = """
From the dataset's owner webpage:

'Users who have submitted the Pledge and received authorization may freely use the database for research purposes
without facing the usual copyright restrictions, but all of the copyrights and neighboring rights connected with
this database belong to the National Institute of Advanced Industrial Science and Technology and are managed by the
RWC Music Database Administrator. Persons or organizations that have not submitted a Pledge and that have not
received authorization may not use the database.'

See https://staff.aist.go.jp/m.goto/RWC-MDB/ for more details.
"""


class Track(core.Track):
    """rwc_classical Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): the track's artist
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        category (str): One of 'Symphony', 'Concerto', 'Orchestral',
            'Solo', 'Chamber', 'Vocal', or blank.
        composer (str): Composer of this Track.
        duration (float): Duration of the track in seconds
        piece_number (str): Piece number of this Track, [1-50]
        sections_path (str): path of the section annotation file
        suffix (str): string within M01-M06
        title (str): Title of The track.
        track_id (str): track id
        track_number (str): CD track number of this Track

    Cached Properties:
        sections (SectionData): human-labeled section annotations
        beats (BeatData): human-labeled beat annotations
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.sections_path = self.get_path("sections")
        self.beats_path = self.get_path("beats")

        self.audio_path = self.get_path("audio")

    @property
    def piece_number(self):
        return self._track_metadata.get("piece_number")

    @property
    def suffix(self):
        return self._track_metadata.get("suffix")

    @property
    def track_number(self):
        return self._track_metadata.get("track_number")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def composer(self):
        return self._track_metadata.get("composer")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def duration(self):
        return self._track_metadata.get("duration")

    @property
    def category(self):
        return self._track_metadata.get("category")

    @core.cached_property
    def sections(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_path)

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

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
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.beats, None)],
            section_data=[(self.sections, None)],
            metadata=self._track_metadata,
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a RWC audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_sections(fhandle: TextIO) -> Optional[annotations.SectionData]:
    """Load rwc section data from a file.

    Args:
        fhandle (str or file-like): File-like object or path to sections annotation file

    Returns:
        SectionData: section data
    """
    begs = []  # timestamps of section beginnings
    ends = []  # timestamps of section endings
    secs = []  # section labels

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        begs.append(float(line[0]) / 100.0)
        ends.append(float(line[1]) / 100.0)
        secs.append(line[2])

    if not begs:  # some files are empty
        return None

    return annotations.SectionData(np.array([begs, ends]).T, "s", secs, "open")


def _position_in_bar(beat_positions, beat_times):
    """Map raw rwc eat data to beat position in bar (e.g. 1, 2, 3, 4).

    Args:
        beat_positions (np.ndarray): raw rwc beat positions
        beat_times (np.ndarray): raw rwc time stamps

    Returns:
        * np.ndarray - normalized beat positions
        * np.ndarray - normalized time stamps
    """
    # Remove -1
    _beat_positions = np.delete(beat_positions, np.where(beat_positions == -1))
    beat_times_corrected = np.delete(beat_times, np.where(beat_positions == -1))

    # Create corrected array with downbeat positions
    beat_positions_corrected = np.zeros((len(_beat_positions),))
    downbeat_positions = np.where(_beat_positions == 384)[0]
    _beat_positions[downbeat_positions] = 1
    beat_positions_corrected[downbeat_positions] = 1

    # Propagate positions
    for b in range(0, len(_beat_positions)):
        if _beat_positions[b] > _beat_positions[b - 1]:
            beat_positions_corrected[b] = beat_positions_corrected[b - 1] + 1

    if not downbeat_positions[0] == 0:
        timesig_next_bar = beat_positions_corrected[downbeat_positions[1] - 1]
        for b in range(1, downbeat_positions[0] + 1):
            beat_positions_corrected[downbeat_positions[0] - b] = timesig_next_bar - b + 1

    return beat_positions_corrected, beat_times_corrected


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load rwc beat data from a file.

    Args:
        fhandle (str or file-like): File-like object or path to beats annotation file

    Returns:
        BeatData: beat data
    """

    beat_times = []  # timestamps of beat interval beginnings
    beat_positions = []  # beat position inside the bar

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        beat_times.append(float(line[0]) / 100.0)
        beat_positions.append(int(line[2]))
    beat_positions_in_bar, beat_times = _position_in_bar(
        np.array(beat_positions), np.array(beat_times)
    )

    return annotations.BeatData(beat_times, "s", beat_positions_in_bar.astype(int), "bar_index")


def _duration_to_sec(duration):
    """Convert min:sec duration values to seconds.

    Args:
        duration (str): duration in form min:sec

    Returns:
        float: duration in seconds
    """
    if type(duration) == str:
        if ":" in duration:
            if len(duration.split(":")) <= 2:
                minutes, secs = duration.split(":")
            else:
                minutes, secs, _ = duration.split(":")  # mistake in annotation in RM-J044
            total_secs = float(minutes) * 60 + float(secs)
            return total_secs
    else:
        raise ValueError("Expected duration to have type str, got {}".format(type(duration)))


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The rwc_classical dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rwc_classical",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata-master", "rwc-c.csv")

        try:
            with open(metadata_path, "r") as fhandle:
                dialect = csv.Sniffer().sniff(fhandle.read(1024))
                fhandle.seek(0)
                reader = csv.reader(fhandle, dialect)
                raw_data = []
                for line in reader:
                    if line[0] != "Piece No.":
                        raw_data.append(line)
        except:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata_index = {}
        for line in raw_data:
            if line[0] == "Piece No.":
                continue
            p = "00" + line[0].split(".")[1][1:]
            track_id = "RM-C{}".format(p[len(p) - 3 :])

            metadata_index[track_id] = {
                "piece_number": line[0],
                "suffix": line[1],
                "track_number": line[2],
                "title": line[3],
                "composer": line[4],
                "artist": line[5],
                "duration": _duration_to_sec(line[6]),
                "category": line[7],
            }

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.rwc_classical.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_classical.load_sections", version="0.3.4")
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_classical.load_beats", version="0.3.4")
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)
