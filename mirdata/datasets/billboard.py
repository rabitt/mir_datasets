"""McGill Billboard Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The McGill Billboard dataset includes annotations and audio features corresponding to 890 slots from a random sample of Billboard chart slots.
    It also includes metadata like Billboard chart date, peak rank, artist name, etc.
    Details can be found at https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)
"""

import csv
import os
import re
from typing import BinaryIO, Dict, List, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@inproceedings{burgoyne_billboard,
author = {Burgoyne, John Ashley and Wild, Jonathan and Fujinaga, Ichiro},
year = {2011},
title = {An {Expert} {Ground} {Truth} {Set} for {Audio} {Chord} {Recognition} and {Music} {Analysis}},
booktitle={Proceedings of the 12th International Society for Music Information Retrieval Conference, ISMIR}
}

@phdthesis{phdthesis,
  author       = {Burgoyne, John Ashley}, 
  title        = {Stochastic {Processes} and {Database}-{Driven} {Musicology}},
  school       = {McGill University, Montréal, Québec},
  year         = 2012,
}
"""

INDEXES = {
    "default": "2.0",
    "test": "2.0",
    "2.0": core.Index(filename="billboard_index_2.0.json"),
}

REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="billboard-2.0-index.csv",
        url="https://www.dropbox.com/s/o0olz0uwl9z9stb/billboard-2.0-index.csv?dl=1",
        checksum="c47d304c212725998839cf9bb1a417aa",
    ),
    "annotation_salami": download_utils.RemoteFileMetadata(
        filename="billboard-2.0-salami_chords.tar.gz",
        url="https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1",
        checksum="6954a6fad962a111e69c9c80cb87d3a5",
    ),
    "annotation_lab": download_utils.RemoteFileMetadata(
        filename="billboard-2.0.1-lab.tar.gz",
        url="https://www.dropbox.com/s/t390alzrkx0c9yt/billboard-2.0.1-lab.tar.gz?dl=1",
        checksum="a7b1fa6a7e454bf73ced7c29207aa597",
    ),
    "annotation_mirex13": download_utils.RemoteFileMetadata(
        filename="billboard-2.0.1-mirex.tar.gz",
        url="https://www.dropbox.com/s/fg8lvy79o7etiyc/billboard-2.0.1-mirex.tar.gz?dl=1",
        checksum="97e5754699f3b45aa5cc70d8a7611c54",
    ),
    "annotation_chordino": download_utils.RemoteFileMetadata(
        filename="billboard-2.0-chordino.tar.gz",
        url="https://www.dropbox.com/s/e9dm23vbawg9dsw/billboard-2.0-chordino.tar.gz?dl=1",
        checksum="530218e8d7077bbd4b08b45f447f5e8f",
    ),
}

LICENSE_INFO = """
This data is released under a Creative Commons 0 license, effectively dedicating it to
the public domain. More information about this dedication and your rights, please see the
details here: http://creativecommons.org/publicdomain/zero/1.0/ and
http://creativecommons.org/publicdomain/zero/1.0/legalcode.
"""


class Track(core.Track):
    """McGill Billboard Dataset Track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        track_id (str): the index for the sample entry
        audio_path (str): audio path of the track
        chart date (str): the date of the chart for the entry
        target rank (int): the desired rank on that chart
        actual rank (int): the rank of the song actually annotated, which may be up to 2 ranks higher or lower than the target rank
        title (str): the title of the song annotated
        artist (str): the name of the artist performing the song annotated
        peak rank (int): the highest rank the song annotated ever achieved on the Billboard Hot 100
        weeks on chart (int): the number of weeks the song annotated spent on the Billboard Hot 100 chart in total

    Cached Properties:
        chords_full (ChordData): HTK-style LAB files for the chord annotations (full)
        chords_majmin7 (ChordData): HTK-style LAB files for the chord annotations (majmin7)
        chords_majmin7inv (ChordData): HTK-style LAB files for the chord annotations (majmin7inv)
        chords_majmin (ChordData): HTK-style LAB files for the chord annotations (majmin)
        chords_majmininv (ChordData): HTK-style LAB files for the chord annotations(majmininv)
        chroma (np.array): Array containing the non-negative-least-squares chroma vectors
        tuning (list): List containing the tuning estimates
        sections (SectionData): Letter-annotated section data (A,B,A')
        named_sections (SectionData): Name-annotated section data (intro, verse, chorus)
        salami_metadata (dict): Metadata of the Salami LAB file
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.salami_path = self.get_path("salami")
        self.lab_full_path = self.get_path("lab_full")
        self.lab_majmin7_path = self.get_path("lab_majmin7")
        self.lab_majmin7inv_path = self.get_path("lab_majmin7inv")
        self.lab_majmin_path = self.get_path("lab_majmin")
        self.lab_majmininv_path = self.get_path("lab_majmininv")
        self.bothchroma_path = self.get_path("bothchroma")
        self.tuning_path = self.get_path("tuning")

    @property
    def chart_date(self):
        return self._track_metadata.get("chart_date")

    @property
    def target_rank(self):
        return self._track_metadata.get("target_rank")

    @property
    def actual_rank(self):
        return self._track_metadata.get("actual_rank")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def peak_rank(self):
        return self._track_metadata.get("peak_rank")

    @property
    def weeks_on_chart(self):
        return self._track_metadata.get("weeks_on_chart")

    @core.cached_property
    def chords_full(self):
        return load_chords(self.lab_full_path)

    @core.cached_property
    def chords_majmin7(self):
        return load_chords(self.lab_majmin7_path)

    @core.cached_property
    def chords_majmin7inv(self):
        return load_chords(self.lab_majmin7inv_path)

    @core.cached_property
    def chords_majmin(self):
        return load_chords(self.lab_majmin_path)

    @core.cached_property
    def chords_majmininv(self):
        return load_chords(self.lab_majmininv_path)

    @core.cached_property
    def chroma(self):
        """Non-negative-least-squares (NNLS) chroma vectors from the Chordino
        Vamp plug-in.

        Returns:
            np.ndarray - NNLS chroma vector
        """
        # removed the first column since it contains metadata.
        with open(self.bothchroma_path, "r") as f:
            return np.array([l for l in csv.reader(f)])[:, 1:].astype(np.float32)

    @core.cached_property
    def tuning(self):
        """Tuning estimates from the Chordino Vamp plug-in.

        Returns:
            list - list of of tuning estimates []
        """
        with open(self.tuning_path, "r") as f:
            return next(csv.reader(f))[1:]

    @core.cached_property
    def sections(self):
        return load_sections(os.path.join(self._data_home, self._track_paths["salami"][0]))

    @core.cached_property
    def named_sections(self):
        return load_named_sections(os.path.join(self._data_home, self._track_paths["salami"][0]))

    @core.cached_property
    def salami_metadata(self):
        return _parse_salami_metadata(os.path.join(self._data_home, self._track_paths["salami"][0]))

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
            chord_data=[
                (self.chords_full, "Full chords"),
                (self.chords_majmin, "Major/minor chords"),
                (self.chords_majmininv, "Major/minor chords with inversions"),
                (self.chords_majmin7, "Major/minor chords with 7th"),
                (self.chords_majmin7inv, "Major/minor chords with 7th and inversions"),
            ],
            section_data=[
                (self.sections, "Sections annotated using section letters"),
                (self.named_sections, "Sections annotated using section names"),
            ],
            metadata=self._track_metadata,
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Billboard audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_chords(fhandle: TextIO):
    """Load chords from a Salami LAB file.

    Args:
        fhandle (str or file-like): path to audio file

    Returns:
        ChordData: chord data
    """
    start_times = []
    end_times = []
    chords = []

    reader = csv.reader(fhandle, delimiter="\t")
    for l in reader:
        if len(l) > 0:
            start_times.append(float(l[0]))
            end_times.append(float(l[1]))
            chords.append(l[2])

    chord_data = annotations.ChordData(np.array([start_times, end_times]).T, "s", chords, "jams")
    return chord_data


def load_sections(fpath: str):
    """Load letter-annotated sections from a Salami LAB file.

    Args:
        fpath (str): path to sections file

    Returns:
        SectionData: section data
    """
    return _load_sections(fpath, "letter")


def load_named_sections(fpath: str):
    """Load name-annotated sections from a Salami LAB file.

    Args:
        fpath (str): path to sections file

    Returns:
        SectionData: section data
    """
    return _load_sections(fpath, "name")


def _load_sections(fpath: str, section_type: str):
    timed_sections = _parse_timed_sections(fpath)
    assert timed_sections is not None

    # Clean sections
    timed_sections_clean = [ts for ts in timed_sections if ts["section"] is not None]

    start_times = []
    end_times = []
    sections = []

    if section_type == "letter":
        section_label_idx = 0
    elif section_type == "name":
        section_label_idx = 1
    else:
        raise ValueError("This section type is not available.")

    for idx, ts in enumerate(timed_sections_clean):
        if idx < len(timed_sections_clean) - 1:
            start_times.append(timed_sections_clean[idx]["time"])
            end_times.append(timed_sections_clean[idx + 1]["time"])
            sections.append(timed_sections_clean[idx]["section"][section_label_idx])
        else:
            start_times.append(timed_sections_clean[idx]["time"])
            end_times.append(timed_sections[-1]["time"])  # end of song
            sections.append(timed_sections_clean[idx]["section"][section_label_idx])

    section_data = annotations.SectionData(
        np.array([start_times, end_times]).T, "s", sections, "open"
    )
    return section_data


@io.coerce_to_string_io
def _parse_salami_metadata(fhandle: TextIO):
    s = fhandle.read().split("\n")
    o = {}
    for x in s:
        if x.startswith("#"):
            if x[2:].startswith("title:"):
                o["title"] = x[9:]
            if x[2:].startswith("artist:"):
                o["artist"] = x[10:]
            if x[2:].startswith("metre:"):
                o["meter"] = x[9:]
            if x[2:].startswith("tonic:"):
                o["tonic"] = x[9:]
        else:
            break
    return o


@io.coerce_to_string_io
def _parse_timed_sections(fhandle: TextIO) -> List:
    lines = fhandle.read().split("\n")
    salami = _parse_salami(lines)
    assert salami is not None
    timed_sections = _timed_sections(salami)
    return timed_sections


def _parse_salami(s: List) -> Dict:
    """
    Author:
        Brian Whitman
        brian@echonest.com
        https://gist.github.com/bwhitman/11453443
    Parse a salami_chords.txt file and return a dict with all the stuff in it
    """

    def parse(s):
        o = {}
        o["events"] = []
        for x in s:
            if x.startswith("#"):
                if x[2:].startswith("title:"):
                    o["title"] = x[9:]
                if x[2:].startswith("artist:"):
                    o["artist"] = x[10:]
                if x[2:].startswith("metre:"):
                    o["meter"] = x[9:]
                if x[2:].startswith("tonic:"):
                    o["tonic"] = x[9:]
            elif len(x) > 1:
                spot = x.find("\t")
                if spot > 0:
                    time = float(x[0:spot])
                    event = {}
                    event["time"] = time
                    event["notes"] = []
                    rest = x[spot + 1 :]
                    items = rest.split(", ")
                    for i in items:
                        chords = re.findall(r"(?=\| (.*?) \|)", i)
                        section = i.split("|")
                        if len(section) == 1 and not ("(" in section or ")" in section):
                            event["section"] = section[0]
                        if len(chords):
                            event["chords"] = chords
                        else:
                            event["notes"].append(i)
                    o["events"].append(event)
        return o

    o = parse(s)
    return o


def _timed_sections(parsed: Dict) -> List:
    """
    Author:
        Brian Whitman
        brian@echonest.com
        https://gist.github.com/bwhitman/11453443
    Given a salami parse return a list of parsed chords with timestamps & deltas
    """
    timed_sections = []
    tic = 0
    for i, e in enumerate(parsed["events"]):
        sections = []
        try:
            dt = parsed["events"][i + 1]["time"] - e["time"]
        except IndexError:
            dt = 0

        section = None
        if e.get("notes"):
            if len(e.get("notes")) > 1:
                section = (e.get("notes")[0], e.get("notes")[1])
            sections.append(section)

        tic = e["time"]
        if len(sections):
            seconds_per_chord = dt / float(len(sections))
            for c in sections:
                timed_sections.append({"time": tic, "section": c, "length": seconds_per_chord})
                tic = tic + seconds_per_chord
    return timed_sections


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The McGill Billboard dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="billboard",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "billboard-2.0-index.csv")

        try:
            with open(metadata_path, "r") as fhandle:
                reader = csv.reader(fhandle, delimiter=",")
                next(reader, None)
                raw_data = [line for line in reader if line != []]
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata_index = {}
        for line in raw_data:
            track_id = line[0]
            metadata_index[track_id] = {
                "chart_date": line[1],
                "target_rank": int(line[2]) if line[2] else None,
                "actual_rank": int(line[3]) if line[3] else None,
                "title": line[4],
                "artist": line[5],
                "peak_rank": int(line[6]) if line[6] else None,
                "weeks_on_chart": int(line[7]) if line[7] else None,
            }
        return metadata_index

    @deprecated(reason="Use mirdata.datasets.billboard.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.billboard.load_sections", version="0.3.4")
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.billboard.load_named_sections", version="0.3.4")
    def load_named_sections(self, *args, **kwargs):
        return load_named_sections(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.billboard.load_chords", version="0.3.4")
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)
