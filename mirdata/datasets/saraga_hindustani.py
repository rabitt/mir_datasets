"""Saraga Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    This dataset contains time aligned melody, rhythm and structural annotations of Hindustani Music tracks, extracted
    from the large open Indian Art Music corpora of CompMusic.

    The dataset contains the following manual annotations referring to audio files:

    - Section and tempo annotations stored as start and end timestamps together with the name of the section and
      tempo during the section (in a separate file)
    - Sama annotations referring to rhythmic cycle boundaries stored
      as timestamps
    - Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
      ({S, r, R, g, G, m, M, P, d, D, n, N})
    - Audio features automatically extracted and stored: pitch and tonic.
    - The annotations are stored in text files, named as the audio filename but with the respective extension at the
      end, for instance: "Bhuvini Dasudane.tempo-manual.txt".

    The dataset contains a total of 108 tracks.

    The files of this dataset are shared with the following license:
    Creative Commons Attribution Non Commercial Share Alike 4.0 International

    Dataset compiled by: Bozkurt, B.; Srinivasamurthy, A.; Gulati, S. and Serra, X.

    For more information about the dataset as well as IAM and annotations, please refer to:
    https://mtg.github.io/saraga/, where a really detailed explanation of the data and annotations is published.
"""
import csv
import json
import os

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@dataset{bozkurt_b_2018_4301737,
  author       = {Bozkurt, B. and
                  Srinivasamurthy, A. and
                  Gulati, S. and
                  Serra, X.},
  title        = {Saraga: research datasets of Indian Art Music},
  month        = may,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {1.5},
  doi          = {10.5281/zenodo.4301737},
  url          = {https://doi.org/10.5281/zenodo.4301737}
}
"""

INDEXES = {
    "default": "1.5",
    "test": "1.5",
    "1.5": core.Index(filename="saraga_hindustani_index_1.5.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="saraga1.5_hindustani.zip",
        url="https://zenodo.org/record/4301737/files/saraga1.5_hindustani.zip?download=1",
        checksum="ea9ed2885ea37a1b10e42f60cf299702",
    )
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial Share Alike 4.0 International."


class Track(core.Track):
    """Saraga Hindustani Track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        ctonic_path (str): path to ctonic annotation file
        pitch_path (str): path to pitch annotation file
        tempo_path (str): path to tempo annotation file
        sama_path (str): path to sama annotation file
        sections_path (str): path to sections annotation file
        phrases_path (str): path to phrases annotation file
        metadata_path (str): path to metadata annotation file

    Cached Properties:
        tonic (float): tonic annotation
        pitch (F0Data): pitch annotation
        tempo (dict): tempo annotations
        sama (BeatData): Sama section annotations
        sections (SectionData): track section annotations
        phrases (EventData): phrase annotations
        metadata (dict): track metadata with the following fields

            - title (str): Title of the piece in the track
            - mbid (str): MusicBrainz ID of the track
            - album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
            - artists (list, dicts): list of dicts containing information of the featuring artists in the track
            - raags (list, dict): list of dicts containing information about the raags present in the track
            - forms (list, dict): list of dicts containing information about the forms present in the track
            - release (list, dicts): list of dicts containing information of the release where the track is found
            - works (list, dicts): list of dicts containing the work present in the piece, and its mbid
            - taals (list, dicts): list of dicts containing the taals present in the track and its uuid
            - layas (list, dicts): list of dicts containing the layas present in the track and its uuid
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        # Audio path
        self.audio_path = self.get_path("audio")

        # Annotation paths
        self.ctonic_path = self.get_path("ctonic")
        self.pitch_path = self.get_path("pitch")
        self.tempo_path = self.get_path("tempo")
        self.sama_path = self.get_path("sama")
        self.sections_path = self.get_path("sections")
        self.phrases_path = self.get_path("phrases")
        self.metadata_path = self.get_path("metadata")

    @core.cached_property
    def tonic(self):
        return load_tonic(self.ctonic_path)

    @core.cached_property
    def pitch(self):
        return load_pitch(self.pitch_path)

    @core.cached_property
    def tempo(self):
        return load_tempo(self.tempo_path)

    @core.cached_property
    def sama(self):
        return load_sama(self.sama_path)

    @core.cached_property
    def sections(self):
        return load_sections(self.sections_path)

    @core.cached_property
    def phrases(self):
        return load_phrases(self.phrases_path)

    @core.cached_property
    def metadata(self):
        return load_metadata(self.metadata_path)

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
            beat_data=[(self.sama, "sama")],
            event_data=[(self.phrases, "phrases")],
            f0_data=[(self.pitch, "pitch")],
            section_data=[(self.sections, "sections")],
            metadata={
                "tempo": self.tempo,
                "tonic": self.tonic,
                "metadata": self.metadata,
            },
        )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(audio_path):
    """Load a Saraga Hindustani audio file.

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
def load_tonic(fhandle):
    """Load track absolute tonic.

    Args:
        fhandle (str or file-like): Local path where the tonic path is stored.
            If `None`, returns None.

    Returns:
        int: Tonic annotation in Hz
    """
    reader = csv.reader(fhandle, delimiter="\t")
    tonic = float(next(reader)[0])
    return tonic


@io.coerce_to_string_io
def load_pitch(fhandle):
    """Load automatic extracted pitch or melody.

    Args:
        fhandle (str or file-like): Local path where the pitch annotation is stored.
            If `None`, returns None.

    Returns:
        F0Data: pitch annotation
    """
    times = []
    freqs = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        times.append(float(line[0]))
        freqs.append(float(line[1]))

    if not times:
        return None

    times = np.array(times)
    freqs = np.array(freqs)
    voicing = (freqs > 0).astype(float)
    return annotations.F0Data(times, "s", freqs, "hz", voicing, "binary")


@io.coerce_to_string_io
def load_tempo(fhandle):
    """Load tempo from hindustani collection.

    Args:
        fhandle (str or file-like): Local path where the tempo annotation is stored.

    Returns:
        dict:
            Dictionary of tempo information with the following keys:

            - tempo: median tempo for the section in mātrās per minute (MPM)
            - matra_interval: tempo expressed as the duration of the mātra (essentially
              dividing 60 by tempo, expressed in seconds)
            - sama_interval: median duration of one tāl cycle in the section
            - matras_per_cycle: indicator of the structure of the tāl, showing the number
              of mātrā in a cycle of the tāl of the recording
            - start_time: start time of the section
            - duration: duration of the section
    """
    tempo_annotation = {}
    head, tail = os.path.split(fhandle.name)
    sections_path = tail.split(".")[0] + ".sections-manual-p.txt"
    sections_abs_path = os.path.join(head, sections_path)

    sections = []
    try:
        with open(sections_abs_path, "r", encoding="utf-8") as fhandle2:
            reader = csv.reader(fhandle2, delimiter=",")
            for line in reader:
                if line != "\n":
                    sections.append(line[3])
    except FileNotFoundError:
        raise FileNotFoundError(f"File {sections_abs_path} not found.")

    section_count = 0

    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        if "NaN" in line or " NaN" in line or "NaN " in line:
            return None

        # Store partial tempo information
        tempo = line[0]
        matra = line[1]
        sama_interval = line[2]
        matras_per_cycle = line[3]
        start_time = line[4]
        duration = line[5]

        tempo_annotation[sections[section_count]] = {
            "tempo": float(tempo) if "." in tempo else int(tempo),
            "matra_interval": float(matra) if "." in matra else int(matra),
            "sama_interval": float(sama_interval) if "." in sama_interval else int(sama_interval),
            "matras_per_cycle": float(matras_per_cycle)
            if "." in matras_per_cycle
            else int(matras_per_cycle),
            "start_time": float(start_time) if "." in start_time else int(start_time),
            "duration": float(duration) if "." in duration else int(duration),
        }

        section_count += 1  # Go to next section

    return tempo_annotation


@io.coerce_to_string_io
def load_sama(fhandle):
    """Load sama.

    Args:
        fhandle (str or file-like): Local path where the sama annotation is stored.
            If `None`, returns None.

    Returns:
        SectionData: sama annotations
    """
    beat_times = []
    beat_positions = []
    idx = 1

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        beat_times.append(float(line[0]))
        beat_positions.append(idx)
        idx += 1

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(np.array(beat_times), "s", np.array(beat_positions), "global_index")


@io.coerce_to_string_io
def load_sections(fhandle):
    """Load tracks sections.

    Args:
        fhandle (str or file-like): Local path where the section annotation is stored.

    Returns:
        SectionData: section annotations for track
    """
    intervals = []
    section_labels = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        if line:
            intervals.append([float(line[0]), float(line[0]) + float(line[2])])
            section_labels.append(str(line[3]) + "-" + str(line[1]))

    # Return None if sections file is empty
    if not intervals:
        return None

    return annotations.SectionData(np.array(intervals), "s", section_labels, "open")


@io.coerce_to_string_io
def load_phrases(fhandle):
    """Load phrases.

    Args:
        fhandle (str or file-like): Local path where the phrase annotation is stored.
            If `None`, returns None.

    Returns:
        EventData: phrases annotation for track
    """
    start_times = []
    end_times = []
    events = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[0]) + float(line[2]))
        if len(line) == 4:
            events.append(str(line[3].split("\n")[0]))
        else:
            events.append("")

    if not start_times:
        return None

    return annotations.EventData(np.array([start_times, end_times]).T, "s", events, "open")


@io.coerce_to_string_io
def load_metadata(fhandle):
    """Load a Saraga Hindustani metadata file.

    Args:
        fhandle (str or file-like): path to metadata json file

    Returns:
        dict: metadata with the following fields

            - title (str): Title of the piece in the track
            - mbid (str): MusicBrainz ID of the track
            - album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
            - artists (list, dicts): list of dicts containing information of the featuring artists in the track
            - raags (list, dict): list of dicts containing information about the raags present in the track
            - forms (list, dict): list of dicts containing information about the forms present in the track
            - release (list, dicts): list of dicts containing information of the release where the track is found
            - works (list, dicts): list of dicts containing the work present in the piece, and its mbid
            - taals (list, dicts): list of dicts containing the taals present in the track and its uuid
            - layas (list, dicts): list of dicts containing the layas present in the track and its uuid
    """
    return json.load(fhandle)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The saraga_hindustani dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="saraga_hindustani",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.saraga_hindustani.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.saraga_hindustani.load_tonic", version="0.3.4")
    def load_tonic(self, *args, **kwargs):
        return load_tonic(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.saraga_hindustani.load_pitch", version="0.3.4")
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.saraga_hindustani.load_tempo", version="0.3.4")
    def load_tempo(self, *args, **kwargs):
        return load_tempo(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.saraga_hindustani.load_sama", version="0.3.4")
    def load_sama(self, *args, **kwargs):
        return load_sama(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.saraga_hindustani.load_sections", version="0.3.4")
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.saraga_hindustani.load_phrases", version="0.3.4")
    def load_phrases(self, *args, **kwargs):
        return load_phrases(*args, **kwargs)
