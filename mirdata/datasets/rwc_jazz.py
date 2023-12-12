"""RWC Jazz Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Jazz Music Database consists of 50 pieces:

    - **Instrumentation variations:** 35 pieces (5 pieces × 7 instrumentations).

        The instrumentation-variation pieces were recorded to obtain different versions
        of the same piece; i.e., different arrangements performed by different player
        instrumentations. Five standard-style jazz pieces were originally composed
        and then performed in modern-jazz style using the following seven instrumentations:

        1. Piano solo
        2. Guitar solo
        3. Duo: Vibraphone + Piano, Flute + Piano, and Piano + Bass
        4. Piano trio: Piano + Bass + Drums
        5. Piano trio + Trumpet or Tenor saxophone
        6. Octet: Piano trio + Guitar + Alto saxophone + Baritone saxophone + Tenor saxophone × 2
        7. Piano trio + Vibraphone or Flute

    - **Style variations:** 9 pieces

        The style-variation pieces were recorded to represent various styles of jazz.
        They include four well-known public-domain pieces and consist of

        1. Vocal jazz: 2 pieces (including "Aura Lee")
        2. Big band jazz: 2 pieces (including "The Entertainer")
        3. Modal jazz: 2 pieces
        4. Funky jazz: 2 pieces (including "Silent Night")
        5. Free jazz: 1 piece (including "Joyful, Joyful, We Adore Thee")

    - **Fusion (crossover):** 6 pieces

        The fusion pieces were recorded to obtain music that combines elements of jazz
        with other styles such as popular, rock, and latin. They include music with an
        eighth-note feel, music with a sixteenth-note feel, and Latin jazz music.

    For more details, please visit: https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-j.html
"""
import csv
import os
from typing import Optional, Tuple

import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, jams_utils

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
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="rwc_jazz_index_1.0.json"),
}

REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="master.zip",
        url="https://github.com/magdalenafuentes/metadata/archive/master.zip",
        checksum="7dbe87fedbaaa1f348625a2af1d78030",
    ),
    "annotations_beat": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-J-2001.BEAT.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-J-2001.BEAT.zip",
        checksum="b483853da05d0fff3992879f7729bcb4",
        destination_dir="annotations",
    ),
    "annotations_sections": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-J-2001.CHORUS.zip",
        url=(
            "https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-J-2001.CHORUS.zip"
        ),
        checksum="44afcf7f193d7e48a7d99e7a6f3ed39d",
        destination_dir="annotations",
    ),
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the RWC-Jazz dataset are not available
    for download. If you have the RWC-Jazz dataset, place the contents into a
    folder called RWC-Jazz with the following structure:
        > RWC-Jazz/
            > annotations/
            > audio/rwc-j-m0i with i in [1 .. 4]
            > metadata-master/
    and copy the RWC-Jazz folder to {}
"""


class Track(core.Track):
    """rwc_jazz Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): Artist name
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        duration (float): Duration of the track in seconds
        instruments (str): list of used instruments.
        piece_number (str): Piece number of this Track, [1-50]
        sections_path (str): path of the section annotation file
        suffix (str): M01-M04
        title (str): Title of The track.
        track_id (str): track id
        track_number (str): CD track number of this Track
        variation (str):  style variations

    Cached Properties:
        sections (SectionData): human-labeled section data
        beats (BeatData): human-labeled beat data
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
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def duration(self):
        return self._track_metadata.get("duration")

    @property
    def variation(self):
        return self._track_metadata.get("variation")

    @property
    def instruments(self):
        return self._track_metadata.get("instruments")

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


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The rwc_jazz dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rwc_jazz",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata-master", "rwc-j.csv")

        try:
            with open(metadata_path, "r", encoding="utf-8") as fhandle:
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
            track_id = "RM-J{}".format(p[len(p) - 3 :])

            metadata_index[track_id] = {
                "piece_number": line[0],
                "suffix": line[1],
                "track_number": line[2],
                "title": line[3],
                "artist": line[4],
                "duration": _duration_to_sec(line[5]),
                "variation": line[6],
                "instruments": line[7],
            }

        return metadata_index

    @deprecated(reason="Use mirdata.datasets.rwc_jazz.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_jazz.load_sections", version="0.3.4")
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.rwc_jazz.load_beats", version="0.3.4")
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)
