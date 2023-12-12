"""Filosax Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Filosax dataset was conceived, curated and compiled by Dave Foster (a PhD student on the AIM programme at QMUL) and his supervisor Simon Dixon (C4DM @ QMUL).
    The dataset is a collection of 48 multitrack jazz recordings, where each piece has 8 corresponding audio files:

    1) The original Aebersold backing track (stereo)
    2) Bass_Drums, a mono file of a mix of bass and drums
    3) Piano_Drums, a mono file of a mix of piano and drums
    4) Participant 1 Sax, a mono file of solo saxophone
    5) Participant 2 Sax, a mono file of solo saxophone
    6) Participant 3 Sax, a mono file of solo saxophone
    7) Participant 4 Sax, a mono file of solo saxophone
    8) Participant 5 Sax, a mono file of solo saxophone

    Each piece is ~6mins long, so each of the 8 stems contains ~5hours of audio

    For each piece, there is a corresponding .jams file containing piece-level annotations:

    1) Beat annotation for the start of each bar and any mid-bar chord change
    2) Chord annotation for each bar, and mid-bar chord change
    3) Section annotation for when the solo changes between the 3 categories:
        a) head (melody)
        b) written solo (interpretation of transcribed solo)
        c) improvised solo

    For each Sax recording (5 per piece), there is a corresponding .json file containing note annotations (see Note object).

    The Participant folders also contain MIDI files of the transcriptions (frame level and score level) as well as a PDF and MusicXML of the typeset solo.

    The dataset comes in 2 flavours: full (all 48 tracks and 5 sax players) and lite (5 tracks and 2 sax players).
    Both flavours can be used with or without the backing tracks (which need to be purchased online).
    Hence, when opening the dataset, use one of 4 versions: 'full', 'full_sax', 'lite', 'lite_sax'.
"""
import json
from typing import BinaryIO, List, Optional, TextIO, Tuple

import jams
import librosa
import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@inproceedings{
  foster_filosax_2021,
  title={Filosax: A Dataset of Annotated Jazz Saxophone Recordings},
  author={Foster, Dave and Dixon, Simon},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2021}
}
"""

INDEXES = {
    "default": "full_1.0",
    "full": "full_1.0",
    "full_sax": "full_sax_1.0",
    "lite": "lite_1.0",
    "lite_sax": "lite_sax_1.0",
    "full_1.0": core.Index(filename="filosax_index_full_1.0.json"),
    "full_sax_1.0": core.Index(filename="filosax_index_full_sax_1.0.json"),
    "lite_1.0": core.Index(filename="filosax_index_lite_1.0.json"),
    "lite_sax_1.0": core.Index(filename="filosax_index_lite_sax_1.0.json"),
    "test": core.Index(filename="filosax_index_lite_1.0.json"),
}

DOWNLOAD_INFO = """
To download the dataset, first go to the Zenodo pages below to request access:

(Full - 14.5GB)
https://zenodo.org/record/5643843#.YYL7aS2l3UI

(Lite - 558MB)
https://zenodo.org/record/5643734#.YYLQ-i2l3UI

Unzip the downloaded file to the folder /Users/<username>/mir_datasets/ (or wherever data_home has been assigned on initialization), and remove the version number from the folder:

(Full)
/Users/<username>/mir_datasets/Filosax

(Lite)
/Users/<username>/mir_datasets/Filosax_Lite

This data is sufficient to use the dataset in the "_sax" (sax only) mode. To download the backing data, go to the Aebersold sites:

(Full)
https://www.jazzbooks.com/mm5/merchant.mvc?&Screen=WISH&Store_Code=JAJAZZ&WishList_ID=1679

(Lite)
https://www.jazzbooks.com/mm5/merchant.mvc?&Screen=WISH&Store_Code=JAJAZZ&WishList_ID=1678

Put the files downloaded into the "/Aebersold" folder, and then run the appropriate script from inside the home folder:

(Full)
python Scripts/Compile_Backing.py -version full

(Lite)
python Scripts/Compile_Backing.py -version lite

which populates the "/Backing" folder with edited files, which match the versions that were used in the recordings.

"""

LICENSE_INFO = """
The Filosax dataset contains copyright material and is shared with researchers under the following conditions:
1. Filosax may only be used by the individual signing below and by members of the research group or organisation of this individual. This permission is not transferable.
2. Filosax may be used only for non-commercial research purposes.
3. Filosax (or data enabling the its reproduction) may not be sold, leased, published or distributed to any third party without written permission from the Filosax administrator.
4. When research results obtained using Filosax are publicly released (in the form of reports, publications, or derivative software), clear indication of the use of Filosax shall be given, usually in the form of a citation of the following paper:
    D. Foster and S. Dixon (2021),  Filosax: A Dataset of Annotated Jazz Saxophone Recordings.
    22nd International Society for Music Information Retrieval Conference (ISMIR).
5. Queen Mary University of London shall not be held liable for any errors in the content of Filosax nor damage arising from the use of Filosax.
6. The Filosax administrator may update these conditions of use at any time. 
"""


class Note:
    """Filosax Note class - dictionary wrapper to give dot properties

    Args:
        input_dict (dict): dictionary of attributes

    Attributes:
        a_start_time (float): the time stamp of the note start, in seconds
        a_end_time (float): the time stamp of the note end, in seconds
        a_duration (float): the duration of the note, in seconds
        a_onset_time (float): the onset time (compared to a_start_time) (filosax_full only, 0.0 otherwise)
        midi_pitch (int): the quantised midi pitch
        crochet_num (int): the number of sub-divisions which define a crochet (always 24)
        musician (int): the participant ID
        bar_num (int): the bar number of the start of the note
        s_start_time (float): the time stamp of the score note start, in seconds
        s_duration (float): the duration of the score note, in seconds
        s_end_time (float): the time stamp of the score note end, in seconds
        s_rhythmic_duration (int): the duration of the score note (compared to crochet_num)
        s_rhythmic_position (int): the position in the bar of the score note start (compared to crochet_num)
        tempo (float): the tempo at the start of the note, in beats per minute
        bar_type (int): the section annotation where 0 = head, 1 = written solo, 2 = improvised solo
        is_grace (bool): is the note a grace note, associated with the following note
        chord_changes (dict): the chords, where the key is the rhythmic position of the chord (using crochet_num, relative to s_rhythmic_position) and the value a JAMS chord annotation  (An additional chord is added in the case of a quaver at the end of the bar, followed by a rest on the downbeat)
        num_chord_changes (int): the number of chords which accompany the note (usually 1, sometimes >1 for long notes)
        main_chord_num (int): usually 0, sometimes 1 in the quaver case described above
        scale_changes (list, int): the degree of the chromatic scale when midi_pitch is compared to chord_root
        loudness_max_val (float): the value (db) of the maximum loudness
        loudness_max_time (float): the time (seconds) of the maximum loudness (compared to a_start_time)
        loudness_curve (list, float): the inter-note loudness values, 1 per millisecond
        pitch_average_val (float): the value (midi) of the average pitch and
        pitch_average_time (float): the time (seconds) of the average pitch (compared to a_start_time)
        pitch_curve (list, float): the inter-note pitch values, 1 per millisecond
        pitch_vib_freq (float): the vibrato frequency (Hz), 0.0 if no vibrato detected
        pitch_vib_ext (float): the vibrato extent (midi), 0.0 if no vibrato detected
        spec_cent (float): the spectral centroid value at the time of the maximum loudness
        spec_flux (float): the spectral flux value at the time of the maximum loudness
        spec_cent_curve (list, float): the inter-note spectral centroid values, 1 per millisecond
        spec_flux_curve (list, float): the inter-note spectral flux values, 1 per millisecond
        seq_len (int): the length of the phrase in which the note falls (filosax_full only, -1 otherwise)
        seq_num (int): the note position in the phrase (filosax_full only, -1 otherwise)

    """

    def __init__(self, input_dict):
        self.a_start_time = input_dict["a_start_time"] if "a_start_time" in input_dict else 0.0
        self.a_end_time = input_dict["a_end_time"] if "a_end_time" in input_dict else 0.0
        self.a_duration = input_dict["a_duration"] if "a_duration" in input_dict else 0.0
        self.a_onset_time = input_dict["a_onset_time"] if "a_onset_time" in input_dict else 0.0
        self.midi_pitch = input_dict["midi_pitch"] if "midi_pitch" in input_dict else 0
        self.crochet_num = input_dict["crochet_num"] if "crochet_num" in input_dict else 24
        self.musician = input_dict["musician"] if "musician" in input_dict else 1
        self.bar_num = input_dict["bar_num"] if "bar_num" in input_dict else 1
        self.s_start_time = input_dict["s_start_time"] if "s_start_time" in input_dict else 0.0
        self.s_duration = input_dict["s_duration"] if "s_duration" in input_dict else 0.0
        self.s_end_time = (
            (self.s_start_time + self.s_duration) if "s_start_time" in input_dict else 0.0
        )
        self.s_rhythmic_duration = (
            input_dict["s_rhythmic_duration"] if "s_rhythmic_duration" in input_dict else 0.0
        )
        self.s_rhythmic_position = (
            input_dict["s_rhythmic_position"] if "s_rhythmic_position" in input_dict else 0.0
        )
        self.tempo = input_dict["tempo"] if "tempo" in input_dict else 0.0
        self.bar_type = input_dict["bar_type"] if "bar_type" in input_dict else 1
        self.is_grace = input_dict["is_grace"] if "is_grace" in input_dict else 0
        self.chord_changes = input_dict["chord_changes"] if "chord_changes" in input_dict else [0]
        self.num_chord_changes = (
            input_dict["num_chord_changes"] if "num_chord_changes" in input_dict else 0
        )
        self.main_chord_num = input_dict["main_chord_num"] if "main_chord_num" in input_dict else 0
        self.scale_changes = input_dict["scale_changes"] if "scale_changes" in input_dict else [0]
        self.loudness_max_val = (
            input_dict["loudness_max_val"] if "loudness_max_val" in input_dict else 0.0
        )
        self.loudness_max_time = (
            input_dict["loudness_max_time"] if "loudness_max_time" in input_dict else 0.0
        )
        self.loudness_curve = (
            input_dict["loudness_curve"] if "loudness_curve" in input_dict else [0.0]
        )
        self.pitch_average_val = (
            input_dict["pitch_average_val"] if "pitch_average_val" in input_dict else 0.0
        )
        self.pitch_average_time = (
            input_dict["pitch_average_time"] if "pitch_average_time" in input_dict else 0.0
        )
        self.pitch_curve = input_dict["pitch_curve"] if "pitch_curve" in input_dict else [0.0]
        self.pitch_vib_freq = (
            input_dict["pitch_vib_freq"] if "pitch_vib_freq" in input_dict else 0.0
        )
        self.pitch_vib_ext = input_dict["pitch_vib_ext"] if "pitch_vib_ext" in input_dict else 0.0
        self.spec_cent = input_dict["spec_cent"] if "spec_cent" in input_dict else 0.0
        self.spec_flux = input_dict["spec_flux"] if "spec_flux" in input_dict else 0.0
        self.spec_cent_curve = (
            input_dict["spec_cent_curve"] if "spec_cent_curve" in input_dict else [0.0]
        )
        self.spec_flux_curve = (
            input_dict["spec_flux_curve"] if "spec_flux_curve" in input_dict else [0.0]
        )
        self.seq_len = input_dict["seq_len"] if "seq_len" in input_dict else -1
        self.seq_num = input_dict["seq_num"] if "seq_len" in input_dict else -1


class Track(core.Track):
    """Filosax track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to audio file
        annotation_path (str): path to annotation file
        midi_path (str): path to MIDI file
        musicXML_path (str): path to musicXML file
        pdf_path (str): path to PDF file

    Cached Properties:
        notes (list, Note): an ordered list of Note objects
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_path = self.get_path("audio")
        self.annotation_path = self.get_path("annotation")
        self.midi_path = self.get_path("midi")
        self.musicXML_path = self.get_path("musicXML")
        self.pdf_path = self.get_path("pdf")

    @core.cached_property
    def notes(self) -> Optional[List[Note]]:
        """The track's note list - only for Sax files

        Returns:
            * [Note] - ordered list of Note objects (empty if Backing file)

        """
        if not self.annotation_path:
            return [Note({})]
        else:
            return load_annotation(self.annotation_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    def to_jams(self):
        return jams_utils.jams_converter(audio_path=self.audio_path)


class MultiTrack(core.MultiTrack):
    """Filosax multitrack class.

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Filosax`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_property (str): the name of the attribute of Track which
            returns the audio to be mixed
        name (str): the name of the tune
        duration (float): the duration, in seconds
        beats (list, Observation): the time and beat numbers of bars and chord changes
        chords (list, Observation): the time of chord changes
        segments (list, Observation): the time of segment changes
        bass_drums (Track): the associated bass/drums track
        piano_drums (Track): the associated piano/drums track
        sax (list, Track): a list of associated sax tracks

    Cached Properties:
        annotation (jams.JAMS): a .jams file containing the annotations
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
        self.annotation_path = self.get_path("annotations")

    @property
    def track_audio_property(self):
        return "audio"

    @core.cached_property
    def annotation(self) -> Optional[annotations.EventData]:
        """output type: .jams file"""
        return jams.load(self.annotation_path)

    @property
    def name(self):
        """The track's name.

        Returns:
            * str - track name
        """
        return self.annotation["file_metadata"]["title"]

    @property
    def duration(self):
        """The track's duration.

        Returns:
            * float - track duration (in seconds)
        """
        return self.annotation["file_metadata"]["duration"]

    @property
    def beats(self):
        """The times of downbeats and chord changes.

        Returns:
            * (SortedKeyList, Observation) - timestamp, duration (seconds), beat
        """
        return self.annotation.search(namespace="beat")[0]["data"]

    @property
    def chords(self):
        """The times and values of chord changes.

        Returns:
            * (SortedKeyList, Observation) - timestamp, duration (seconds), chord symbol
        """
        return self.annotation.search(namespace="chord")[0]["data"]

    @property
    def segments(self):
        """The times of segment changes (values are 'head', 'written solo',
        'improvised solo')

        Returns:
            * (SortedKeyList, Observation) - timestamp, duration (seconds), beat
        """
        return self.annotation.search(namespace="segment_open")[0]["data"]

    @property
    def bass_drums(self):
        """The associated bass/drums track.

        Returns:
            * Track
        """
        return self.tracks[self.mtrack_id + "_bass_drums"]

    @property
    def piano_drums(self):
        """The associated piano/drums track.

        Returns:
            * Track
        """
        return self.tracks[self.mtrack_id + "_piano_drums"]

    @property
    def sax(self):
        """The associated sax tracks (1-5)

        Returns:
            * (list, Track)
        """
        return [self.tracks["%s_sax_%d" % (self.mtrack_id, n)] for n in [1, 2, 3, 4, 5]]

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return self.annotation


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Filosax audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_annotation(fhandle: TextIO) -> List[Note]:
    """Load a Filosax annotation file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * (list, Note): an ordered list of Note objects
    """
    note_dict = json.load(fhandle)["notes"]
    return [Note(n) for n in note_dict]


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Filosax dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="filosax",
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            indexes=INDEXES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )
