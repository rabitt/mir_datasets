"""Jingju A Cappella Singing Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Description:
        This dataset is a collection of boundary annotations of a cappella singing performed by
        Beijing Opera (Jingju, 京剧) professional and amateur singers.

    Contents:
        1. wav.zip: audio files in .wav format, mono or stereo.
        2. pycode.zip: util code for parsing the .textgrid annotation
        3. catalogue*.csv: recording metadata, source separation recordings are not included.
        4. annotation_txt.zip: phrase, syllable and phoneme time boundaries (second) and labels in .txt format

    The annotation_txt.zip folder annotations are represented as follows:
        1. phrase_char: phrase-level time boundaries, labeled in Mandarin characters
        2. phrase: phrase-level time boundaries, labeled in Mandarin pinyin
        3. syllable: syllable-level time boundaries, labeled in Mandarin pinyin
        4. phoneme: phoneme-level time boundaries, labeled in X-SAMPA

    The boundaries (onset and offset) have been annotated hierarchically:
        1. phrase (line)
        2. syllable
        3. phoneme

    Annotation details:
        Singing units in pinyin and X-SAMPA have been annotated to a jingju a cappella singing audio dataset.

    Audio details:
        The corresponding audio files are the a cappella singing arias recordings, which are stereo or mono,
        sampled at 44.1 kHz, and stored as .wav files. The .wav files are recorded by two institutes: those file
        names ending with ‘qm’ are recorded by C4DM, Queen Mary University of London; others file names ending with
        ‘upf’ or ‘lon’ are recorded by MTG-UPF. Additionally, another collection of 15 clean singing recordings is
        included in this dataset. They are extracted from the commercial recordings which originally contains karaoke
        accompaniment and mixed versions.

    Additional details:
        Annotation format, units, parsing code and other information please refer to:
        https://github.com/MTG/jingjuPhonemeAnnotation

    License information:
        Textgrid annotations are licensed under Creative Commons Attribution-NonCommercial 4.0 International License.
        Wav audio ending with ‘upf’ or ‘lon’ is licensed under Creative Commons Attribution-NonCommercial 4.0 International.
        For the license of .wav audio ending with ‘qm’ from C4DM Queen Mary University of London, please refer to
        this page http://isophonics.org/SingingVoiceDataset
"""

import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@dataset{rong_gong_2018_1323561,
  author       = {Rong Gong and
                  Rafael Caro Repetto and
                  Yile Yang and
                  Xavier Serra},
  title        = {Jingju a cappella singing dataset part1},
  month        = jul,
  year         = 2018,
  publisher    = {Zenodo},
  version      = 7,
  doi          = {10.5281/zenodo.1323561},
  url          = {https://doi.org/10.5281/zenodo.1323561}
}
@article{black2014automatic,
  title={Automatic identification of emotional cues in Chinese opera singing},
  author={Black, Dawn AA and Li, Ma and Tian, Mi},
  journal={ICMPC, Seoul, South Korea},
  year={2014}
}
"""

INDEXES = {
    "default": "7.0",
    "test": "7.0",
    "7.0": core.Index(filename="compmusic_jingju_acappella_index_7.0.json"),
}

REMOTES = {
    "annotation_txt": download_utils.RemoteFileMetadata(
        filename="annotation_txt.zip",
        url="https://zenodo.org/record/1323561/files/annotation_txt.zip?download=1",
        checksum="851c9c3fe195fd20bec42d32ddd9deb7",
        destination_dir=".",
    ),
    "catalogue_dan": download_utils.RemoteFileMetadata(
        filename="catalogue - dan.csv",
        url="https://zenodo.org/record/1323561/files/catalogue%20-%20dan.csv?download=1",
        checksum="82ce90bd8508b1ae12c6a1fe489618a4",
        destination_dir=".",
    ),
    "catalogue_laosheng": download_utils.RemoteFileMetadata(
        filename="catalogue - laosheng.csv",
        url="https://zenodo.org/record/1323561/files/catalogue%20-%20laosheng.csv?download=1",
        checksum="768fa00ce1f8880ae5480fae103ecc06",
        destination_dir=".",
    ),
    "wav": download_utils.RemoteFileMetadata(
        filename="wav.zip",
        url="https://zenodo.org/record/1323561/files/wav.zip?download=1",
        checksum="4722abda831c20b169a62b2754b15bea",
        destination_dir=".",
    ),
}

LICENSE_INFO = (
    "audio files ending with upf or lon: Creative Commons Attribution Non-Commercial 4.0"
    " International, " + "audio files ending with qm: http://isophonics.org/SingingVoiceDataset"
)


class Track(core.Track):
    """Jingju A Cappella Singing Track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): local path where the audio is stored
        phoneme_path (str): local path where the phoneme annotation is stored
        phrase_char_path (str): local path where the lyric phrase annotation in chinese is stored
        phrase_path (str): local path where the lyric phrase annotation in western characters is stored
        syllable_path (str): local path where the syllable annotation is stored
        work (str): string referring to the work where the track belongs
        details (float): string referring to additional details about the track

    Cached Properties:
        phoneme (EventData): phoneme annotation
        phrase_char (LyricsData): lyric phrase annotation in chinese
        phrase (LyricsData): lyric phrase annotation in western characters
        syllable (EventData): syllable annotation
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

        self.phoneme_path = self.get_path("phoneme")
        self.phrase_char_path = self.get_path("phrase_char")
        self.phrase_path = self.get_path("phrase")
        self.syllable_path = self.get_path("syllable")

    @core.cached_property
    def phoneme(self):
        return load_phonemes(self.phoneme_path)

    @core.cached_property
    def phrase(self):
        return load_phrases(self.phrase_path)

    @core.cached_property
    def phrase_char(self):
        return load_phrases(self.phrase_char_path)

    @core.cached_property
    def syllable(self):
        return load_syllable(self.syllable_path)

    @property
    def work(self):
        return self._track_metadata.get("work")

    @property
    def details(self):
        return self._track_metadata.get("details")

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
            lyrics_data=[
                (self.phrase, "phrases"),
                (self.phrase_char, "phrases_char"),
                (self.phoneme, "phoneme"),
                (self.syllable, "syllable"),
            ],
            metadata={"work": self.work, "details": self.details},
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load Jingju A Cappella Singing audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=44100, mono=True)


@io.coerce_to_string_io
def load_phonemes(fhandle: TextIO) -> annotations.LyricData:
    """Load phonemes.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a phoneme annotation file

    Returns:
        LyricData: phoneme annotation
    """

    start_times = []
    end_times = []
    events = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        events.append(str(line[2] if line[2] != "sil" else ""))

    return annotations.LyricData(
        np.array([start_times, end_times]).T, "s", events, "pronunciations_open"
    )


@io.coerce_to_string_io
def load_phrases(fhandle: TextIO) -> annotations.LyricData:
    """Load lyric phrases annotation.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a lyric annotation file

    Returns:
        LyricData: lyric phrase annotation
    """
    start_times = []
    end_times = []
    lyrics = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        lyrics.append(line[2] if line[2] != "sil" else "")

    return annotations.LyricData(np.array([start_times, end_times]).T, "s", lyrics, "words")


@io.coerce_to_string_io
def load_syllable(fhandle: TextIO) -> annotations.LyricData:
    """Load syllable.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a syllable annotation file

    Returns:
        LyricData: syllable annotation
    """

    start_times = []
    end_times = []
    events = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        events.append(line[2] if line[2] != "sil" else "")

    return annotations.LyricData(np.array([start_times, end_times]).T, "s", events, "syllable_open")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The compmusic_jingju_acappella dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="compmusic_jingju_acappella",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path_laosheng = os.path.join(self.data_home, "catalogue - laosheng.csv")
        # metadata_path_dan = os.path.join(
        #     self.data_home,
        #     "catalogue - dan.csv",
        # )

        metadata = {}
        try:
            with open(metadata_path_laosheng, "r", encoding="utf-8") as fhandle:
                reader = csv.reader(fhandle, delimiter=",")
                next(reader)
                for line in reader:
                    work = line[1] if line[1] else None
                    details = line[3] if line[3] else None
                    metadata[line[0]] = {"work": work, "details": details}

                data_home = os.path.dirname(metadata_path_laosheng)
                metadata["data_home"] = data_home
        except FileNotFoundError:
            raise FileNotFoundError("laosheng metadata not found. Did you run .download()?")

        return metadata

    @deprecated(reason="Use mirdata.datasets.jingju_acapella.load_phonemes", version="0.3.4")
    def load_phonemes(self, *args, **kwargs):
        return load_phonemes(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.jingju_acapella.load_phrases", version="0.3.4")
    def load_phrases(self, *args, **kwargs):
        return load_phrases(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.jingju_acapella.load_syllable", version="0.3.4")
    def load_syllable(self, *args, **kwargs):
        return load_syllable(*args, **kwargs)
