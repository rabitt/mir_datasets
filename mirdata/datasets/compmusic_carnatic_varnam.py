# -*- coding: utf-8 -*-
"""CompMusic Carnatic Varnam Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Carnatic varnam dataset is a collection of 28 solo vocal recordings, recorded for our research on intonation
    analysis of Carnatic raagas. The collection has the audio recordings, taala cycle annotations and notations in a
    machine readable format.

    **Audio music content**
    They feature 7 varnams in 7 rāgas sung by 5 young professional singers who received training for more than 15 years.
    They are all set to Adi taala. Measuring the intonation variations require absolutely clean pitch contours. For
    this, all the varṇaṁs are recorded without accompanying instruments, except the drone.

    **Taala annotations**
    The recordings are annotated with taala cycles, each annotation marking the starting of a cycle. We have later
    automatically divided each cycle into 8 equal parts. The annotations are made available as sonic visualizer
    annotation layers. Each annotation is of the format m.n where m is the cycle number and n is the division within
    the cycle. All m.1 annotations are manually done, whereas m.[2-8] are automatically labelled.

    **Notations**
    The notations for 7 varnams are procured from an archive curated by Shivkumar, in word document format. They are
    manually converted to a machine readable format (yaml). Each file is essentially a dictionary with section names
    of the composition as keys. Each section is represented as a list of cycles. Each cycle in turn has a list of
    divisions.

    **Sections**
    The notation is given a single time per section, however, to align the svaras with the tala annotations, structure
    information is given. The structure is given in yaml format, specifying the order of the sections, and how many svaras
    are sung per each tala tick. Broadly, there are just two only cases, 2 svaras per tick, and 4 svaras per tick.
    The structure information has been added in the 1.1 version of the dataset.

    **Possible uses of the dataset**
    The distinct advantage of this dataset is the free availability of the audio content. Along with the annotations,
    it can be used for melodic analyses: characterizing intonation, motif discovery and tonic identification. The
    availability of a machine readable notation files allows the dataset to be used for audio-score alignment.
"""

import csv
import glob
import os
from xml.dom import minidom

import librosa
import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@dataset{koduri_g_k_2014_1257118,
  author       = {Koduri, G. K. and
                  Ishwar, V. and
                  Serrà, J. and
                  Serra, X.},
  title        = {Carnatic Varnam Dataset},
  month        = feb,
  year         = 2014,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1257118},
  url          = {https://doi.org/10.5281/zenodo.1257118}
}
"""

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="carnatic_varnam_1.1.zip",
        url="https://zenodo.org/record/7726167/files/carnatic_varnam_1.1.zip?download=1",
        checksum="87afaf907e1fbfa5928ef4e93ead1fba",
    )
}

INDEXES = {
    "default": "1.1",
    "test": "1.1",
    "1.1": core.Index(filename="compmusic_carnatic_varnam_index_1.1.json"),
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial No Derivatives 4.0 International"


class Track(core.Track):
    """CompMusic Carnatic Varnam Track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        artist (str): string identifying the performing artist in the track
        raaga (str): string identifying the raaga present in the track

    Cached Properties:
        tonic (float): float identifying the absolute tonic of the track
        taala (BeatData): taala annotations
        notation (EventData): note notations in IAM solfège symbols representation
        sections (SectionData): track section annotations
        mbid (str): musicbrainz id of the composition
        arohanam (list, str): arohanam annotation of the related raaga
        avarohanam (list, str): avarohanam annotation of the related raaga
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        # Audio path
        self.audio_path = self.get_path("audio")

        # Annotation paths
        self.taala_path = self.get_path("taala")
        self.notation_path = self.get_path("notation")
        self.structure_path = self.get_path("structure")
        self.artist = self.track_id.split("_")[0]
        self.raaga = self.track_id.split("_")[1]

    @core.cached_property
    def taala(self):
        return load_taala(self.taala_path)

    @core.cached_property
    def notation(self):
        return load_notation(self.notation_path, self.taala_path, self.structure_path)[0]

    @core.cached_property
    def sections(self):
        return load_notation(self.notation_path, self.taala_path, self.structure_path)[1]

    @core.cached_property
    def mbid(self):
        return load_mbid(self.notation_path)

    @core.cached_property
    def arohanam(self):
        return load_moorchanas(self.notation_path)[0]

    @core.cached_property
    def avarohanam(self):
        return load_moorchanas(self.notation_path)[1]

    @core.cached_property
    def tonic(self):
        return self._track_metadata

    @property
    def audio(self):
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
            beat_data=[(self.taala, "taala")],
            section_data=[(self.sections, "sections")],
            event_data=[(self.notation, "notation")],
            metadata={
                "performer": self.artist,
                "raaga": self.raaga,
                "tonic": self.tonic,
                "arohanam": self.arohanam,
                "avarohanam": self.avarohanam,
            },
        )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(audio_path):
    """Load a Carnatic Varnam audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    if audio_path is None:
        return None
    return librosa.load(audio_path, sr=44100, mono=False)


@io.coerce_to_string_io
def load_taala(fhandle):
    """Load taala annotation.

    Args:
        taala_path (str): Local path where the taala annotation is stored.

    Returns:
        BeatData: taala annotation for track
    """
    # Load svl file
    dom = minidom.parse(fhandle)

    # Load data
    data = dom.getElementsByTagName("data")[0]

    # Store points and calculate total length
    points = data.getElementsByTagName("dataset")[0].getElementsByTagName("point")
    num_points = len(points)

    # Parse sampling frequency
    fs = float(data.getElementsByTagName("model")[0].getAttribute("sampleRate"))

    beat_times = []
    beat_positions = []
    for beat in range(num_points):
        beat_times.append(float(points[beat].getAttribute("frame")) / fs)
        beat_positions.append(0)

    return annotations.BeatData(np.array(beat_times), "s", np.array(beat_positions), "global_index")


# no decorator here because we need three paths
def load_notation(note_path, taala_path, structure_path):
    """Load notation and structure.

    Args:
        note_path (str): Local path where the note annotation is stored.
            If `None`, returns None.
        taala_path (str): Local path where the taala annotation is stored.
            If `None`, returns None.
        structure_path: (str): Local path where the structure annotation is stored.
            If `None`, returns None.

    Returns:
        EventData: melodic notation for track
    """
    try:
        note_file = open(note_path, "r")
        note_reader = csv.reader(note_file, delimiter="-")
    except FileNotFoundError:
        raise FileNotFoundError(
            "note_path {} does not exist, have you run .download()?".format(note_path)
        )

    try:
        taala_file = open(taala_path, "r")
        taala_reader = minidom.parse(taala_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            "taala_path {} does not exist, have you run .download()?".format(taala_path)
        )

    try:
        structure_file = open(structure_path, "r")
        structure_reader = csv.reader(structure_file, delimiter=":")
    except FileNotFoundError:
        raise FileNotFoundError(
            "structure_path {} does not exist, have you run .download()?".format(structure_path)
        )

    start_times = []
    end_times = []
    events = []

    data = taala_reader.getElementsByTagName("data")[0]
    points = data.getElementsByTagName("dataset")[0].getElementsByTagName("point")
    num_points = len(points)
    fs = float(data.getElementsByTagName("model")[0].getAttribute("sampleRate"))

    prev_timestamp = float(points[0].getAttribute("frame")) / fs
    for beat in range(1, num_points):
        start_times.append(prev_timestamp)
        end_times.append(float(points[beat].getAttribute("frame")) / fs)
        prev_timestamp = float(points[beat].getAttribute("frame")) / fs
    start_times.append(prev_timestamp)
    end_times.append(prev_timestamp + (end_times[-1] - start_times[-2]))

    # Getting structure
    structure = []
    for row in structure_reader:
        if len(row) > 1:
            ky = row[0].replace("'", "").replace(" ", "").replace(":", "")
            vl = row[1].replace("'", "").replace(" ", "").replace(":", "")
            if vl and len(vl) < 2:
                structure.append((ky, int(vl)))

    # Getting notation
    for row in note_reader:
        events.append(row[-1].replace("'", "").replace(" ", "").replace(":", ""))

    notation_dict = {}
    section_dict = {events.index(x): x for x in list(np.unique([x[0] for x in structure]))}
    start_idxs = sorted(section_dict.keys())
    end_idxs = sorted(section_dict.keys())[1:] + [len(events)]
    for start, end in zip(start_idxs, end_idxs):
        notation_dict[section_dict[start]] = events[start + 1 : end]

    # Putting all together
    events = []
    intervals = []
    section_labels = []
    for section in structure:
        not_per_sec = notation_dict[section[0]]
        section_start = start_times[len(events)]
        if section[1] == 2:
            for x in np.arange(len(not_per_sec), step=2):
                # notes = [not_per_sec[x], not_per_sec[x+1]]
                notes = not_per_sec[x] + not_per_sec[x + 1]
                events.append(notes)
        if section[1] == 4:
            for x in np.arange(len(not_per_sec), step=4):
                # notes = [not_per_sec[x], not_per_sec[x+1], not_per_sec[x+2], not_per_sec[x+3]]
                notes = (
                    not_per_sec[x] + not_per_sec[x + 1] + not_per_sec[x + 2] + not_per_sec[x + 3]
                )
                events.append(notes)
        section_end = end_times[len(events) - 1]
        intervals.append([section_start, section_end])
        section_labels.append(section[0])

    notes = annotations.EventData(np.array([start_times, end_times]).T, "s", events, "open")
    sections = annotations.SectionData(np.array(intervals), "s", section_labels, "open")
    return notes, sections


@io.coerce_to_string_io
def load_mbid(fhandle):
    """Load musicbrainz id.

    Args:
        fhandle (str or file-like): Local path where the annotation is stored.
            If `None`, returns None.

    Returns:
        string: musicbrainz id for the composition
    """
    reader = csv.reader(fhandle, delimiter=":")
    for row in reader:
        if row[0] == "mbid":
            return row[-1].replace("'", "").replace(" ", "")


@io.coerce_to_string_io
def load_moorchanas(fhandle):
    """Load arohanam and avarohanam annotations.

    Args:
        fhandle (str or file-like): Local path where moorchana annotation is stored.
            If `None`, returns None.

    Returns:
        (list, string): section annotation for track
    """
    notes = []
    reader = csv.reader(fhandle, delimiter="-")
    for row in reader:
        if row[0] == "pallavi:":
            break
        notes.append(str(row[-1].replace(" ", "")))

    arohanam_ind = notes.index("arohana:") + 1  # Get left boundary of arohanam notations
    avarohanam_ind = notes.index("avarohana:") + 1  # Get left boundary of avarohanam notations

    arohanam = notes[arohanam_ind : avarohanam_ind - 1]  # Get arohanam
    avarohanam = notes[avarohanam_ind:]  # Get avarohanam

    return [arohanam, avarohanam]


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The compmusic_carnatic_varnam dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="compmusic_carnatic_varnam",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            indexes=INDEXES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        """Load tonic.

        Args:
            fhandle (str or file-like): Local path where tonic annotations are stored.
                If `None`, returns None.
            track_id (str): Track ID to get the artist name

        Returns:
            (float): tonic
        """
        data_folder = self.remotes["all"].filename.replace(".zip", "")
        tonics_dict = {}
        tonics_path = os.path.join(
            self.data_home,
            data_folder,
            "Notations_Annotations",
            "annotations",
            "tonics.yaml",
        )
        try:
            f = open(tonics_path, "r")
            reader = csv.reader(f, delimiter=":")
        except FileNotFoundError:
            raise FileNotFoundError(
                "tonics_path {} does not exist, have you run .download()?".format(tonics_path)
            )
        for line in reader:
            tonics_dict[line[0]] = float(line[1])

        taalas_path = os.path.join(
            self.data_home,
            data_folder,
            "Notations_Annotations",
            "annotations",
            "taalas",
        )
        out_tonic = {}
        for taala_path in glob.glob(os.path.join(taalas_path, "*/")):
            taala = taala_path.split("/")[-2]
            for track in glob.glob(os.path.join(taala_path, "*.svl")):
                artist = track.split("/")[-1].replace(".svl", "")
                idx = artist + "_" + taala
                out_tonic[idx] = tonics_dict[artist]
        return out_tonic
