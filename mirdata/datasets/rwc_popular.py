"""RWC Popular Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Popular Music Database consists of 100 songs — 20 songs with English lyrics
    performed in the style of popular music typical of songs on the American hit
    charts in the 1980s, and 80 songs with Japanese lyrics performed in the style of
    modern Japanese popular music typical of songs on the Japanese hit charts in
    the 1990s.

    For more details, please visit: https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-p.html
"""
import csv
import os
from typing import Optional, TextIO, Tuple

import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

# these functions are identical for all rwc datasets
from mirdata.datasets.rwc_classical import (
    LICENSE_INFO,
    _duration_to_sec,
    load_audio,
    load_beats,
    load_sections,
)

BIBTEX = """@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
  note={Cite this if using audio, beat or section annotations},
}
@inproceedings{cho2011feature,
  title={A feature smoothing method for chord recognition using recurrence plots},
  author={Cho, Taemin and Bello, Juan P},
  booktitle={12th International Society for Music Information Retrieval Conference},
  year={2011},
  series={ISMIR},
  note={Cite this if using chord annotations},
}
@inproceedings{mauch2011timbre,
  title={Timbre and Melody Features for the Recognition of Vocal Activity and Instrumental Solos in Polyphonic Music.},
  author={Mauch, Matthias and Fujihara, Hiromasa and Yoshii, Kazuyoshi and Goto, Masataka},
  booktitle={ISMIR},
  year={2011},
  series={ISMIR},
  note={Cite this if using vocal-instrumental activity annotations},
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="rwc_popular_index_1.0.json"),
}

REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="master.zip",
        url="https://github.com/magdalenafuentes/metadata/archive/master.zip",
        checksum="7dbe87fedbaaa1f348625a2af1d78030",
    ),
    "annotations_beat": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.BEAT.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.BEAT.zip",
        checksum="3858aa989535bd7196b3cd07b512b5b6",
        destination_dir="annotations",
    ),
    "annotations_sections": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.CHORUS.zip",
        url=(
            "https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.CHORUS.zip"
        ),
        checksum="f76b3a32701fbd9bf78baa608f692a77",
        destination_dir="annotations",
    ),
    "annotations_chords": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.CHORD.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.CHORD.zip",
        checksum="68379c88bc8ec3f1907b32a3579197c5",
        destination_dir="annotations",
    ),
    "annotations_vocal_act": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.VOCA_INST.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.VOCA_INST.zip",
        checksum="47ded648a496407ef49dba9c8bf80e87",
        destination_dir="annotations",
    ),
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the RWC-Popular dataset are not available
    for download. If you have the RWC-Popular dataset, place the contents into a
    folder called RWC-Popular with the following structure:
        > RWC-Popular/
            > annotations/
            > audio/rwc-p-m0i with i in [1 .. 7]
            > metadata-master/
    and copy the RWC-Popular folder to {}
"""


class Track(core.Track):
    """rwc_popular Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): artist
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        chords_path (str): path of the chord annotation file
        drum_information (str): If the drum is 'Drum sequences', 'Live drums',
            or 'Drum loops'
        duration (float): Duration of the track in seconds
        instruments (str): List of used instruments
        piece_number (str): Piece number, [1-50]
        sections_path (str): path of the section annotation file
        singer_information (str): could be male, female or vocal group
        suffix (str): M01-M04
        tempo (str): Tempo of the track in BPM
        title (str): title
        track_id (str): track id
        track_number (str): CD track number
        voca_inst_path (str): path of the vocal/instrumental annotation file

    Cached Properties:
        sections (SectionData): human-labeled section annotation
        beats (BeatData): human-labeled beat annotation
        chords (ChordData): human-labeled chord annotation
        vocal_instrument_activity (EventData): human-labeled vocal/instrument activity
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.sections_path = self.get_path("sections")
        self.beats_path = self.get_path("beats")
        self.chords_path = self.get_path("chords")
        self.voca_inst_path = self.get_path("voca_inst")

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
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def singer_information(self):
        return self._track_metadata.get("singer_information")

    @property
    def duration(self):
        return self._track_metadata.get("duration")

    @property
    def tempo(self):
        return self._track_metadata.get("tempo")

    @property
    def instruments(self):
        return self._track_metadata.get("instruments")

    @property
    def drum_information(self):
        return self._track_metadata.get("drum_information")

    @core.cached_property
    def sections(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_path)

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def chords(self) -> Optional[annotations.ChordData]:
        return load_chords(self.chords_path)

    @core.cached_property
    def vocal_instrument_activity(self) -> Optional[annotations.EventData]:
        return load_vocal_activity(self.voca_inst_path)

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
            chord_data=[(self.chords, None)],
            metadata=self._track_metadata,
        )


@io.coerce_to_string_io
def load_chords(fhandle: TextIO) -> annotations.ChordData:
    """Load rwc chord data from a file.

    Args:
        fhandle (str or file-like): File-like object or path to chord annotation file

    Returns:
        ChordData: chord data
    """
    begs = []  # timestamps of chord beginnings
    ends = []  # timestamps of chord endings
    chords = []  # chord labels

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        begs.append(float(line[0]))
        ends.append(float(line[1]))
        chords.append(line[2])

    return annotations.ChordData(np.array([begs, ends]).T, "s", chords, "harte")


@io.coerce_to_string_io
def load_vocal_activity(fhandle: TextIO) -> annotations.EventData:
    """Load rwc vocal activity data from a file.

    Args:
        fhandle (str or file-like): File-like object or path to vocal activity annotation file

    Returns:
        EventData: vocal activity data
    """
    begs = []  # timestamps of vocal-instrument activity beginnings
    ends = []  # timestamps of vocal-instrument activity endings
    events = []  # vocal-instrument activity labels

    reader = csv.reader(fhandle, delimiter="\t")
    raw_data = []
    for line in reader:
        if line[0] != "Piece No.":
            raw_data.append(line)

    for i in range(len(raw_data)):
        # Parsing vocal-instrument activity as intervals (beg, end, event)
        if raw_data[i] != raw_data[-1]:
            begs.append(float(raw_data[i][0]))
            ends.append(float(raw_data[i + 1][0]))
            events.append(raw_data[i][1])

    return annotations.EventData(np.array([begs, ends]).T, "s", events, "open")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The rwc_popular dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rwc_popular",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata-master", "rwc-p.csv")

        try:
            with open(metadata_path, "r") as fhandle:
                dialect = csv.Sniffer().sniff(fhandle.read(1024))
                fhandle.seek(0)
                reader = csv.reader(fhandle, dialect)
                raw_data = []
                for line in reader:
                    if line[0] != "Piece No.":
                        raw_data.append(line)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata_index = {}
        for line in raw_data:
            if line[0] == "Piece No.":
                continue
            p = "00" + line[0].split(".")[1][1:]
            track_id = "RM-P{}".format(p[len(p) - 3 :])

            metadata_index[track_id] = {
                "piece_number": line[0],
                "suffix": line[1],
                "track_number": line[2],
                "title": line[3],
                "artist": line[4],
                "singer_information": line[5],
                "duration": _duration_to_sec(line[6]),
                "tempo": line[7],
                "instruments": line[8],
                "drum_information": line[9],
            }

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.rwc_popular.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_popular.load_sections", version="0.3.4")
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_popular.load_beats", version="0.3.4")
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_popular.load_chords", version="0.3.4")
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_popular.load_vocal_activity", version="0.3.4")
    def load_vocal_activity(self, *args, **kwargs):
        return load_vocal_activity(*args, **kwargs)
