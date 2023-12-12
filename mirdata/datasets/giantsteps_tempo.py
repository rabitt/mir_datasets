"""giantsteps_tempo Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    GiantSteps tempo + genre is a collection of annotations for 664 2min(1) audio previews from
    www.beatport.com, created by Richard Vogl <richard.vogl@tuwien.ac.at> and
    Peter Knees <peter.knees@tuwien.ac.at>

    references:

    .. [giantsteps_tempo_cit_1] Peter Knees, Ángel Faraldo, Perfecto Herrera, Richard Vogl,
        Sebastian Böck, Florian Hörschläger, Mickael Le Goff: "Two data
        sets for tempo estimation and key detection in electronic dance
        music annotated from user corrections", Proc. of the 16th
        Conference of the International Society for Music Information
        Retrieval (ISMIR'15), Oct. 2015, Malaga, Spain.

    .. [giantsteps_tempo_cit_2] Hendrik Schreiber, Meinard Müller: "A Crowdsourced Experiment
        for Tempo Estimation of Electronic Dance Music", Proc. of the
        19th Conference of the International Society for Music
        Information Retrieval (ISMIR'18), Sept. 2018, Paris, France.

    The audio files (664 files, size ~1gb) can be downloaded from http://www.beatport.com/
    using the bash script:

    https://github.com/GiantSteps/giantsteps-tempo-dataset/blob/master/audio_dl.sh

    To download the files manually use links of the following form:
    http://geo-samples.beatport.com/lofi/<name of mp3 file>
    e.g.:
    http://geo-samples.beatport.com/lofi/5377710.LOFI.mp3

    To convert the audio files to .wav use the script found at
    https://github.com/GiantSteps/giantsteps-tempo-dataset/blob/master/convert_audio.sh and run:

    .. code-block:: bash

        ./convert_audio.sh

    To retrieve the genre information, the JSON contained within the website was parsed.
    The tempo annotation was extracted from forum entries of people correcting the bpm values (i.e. manual annotation of tempo).
    For more information please refer to the publication [giantsteps_tempo_cit_1]_.

    [giantsteps_tempo_cit_2]_ found some files without tempo. There are:

    .. code-block:: bash

        3041381.LOFI.mp3
        3041383.LOFI.mp3
        1327052.LOFI.mp3

    Their v2 tempo is denoted as 0.0 in tempo and mirex and has no annotation in the JAMS format.

    Most of the audio files are 120 seconds long. Exceptions are:

    .. code-block:: bash

        name              length (sec)
        906760.LOFI.mp3   62
        1327052.LOFI.mp3  70
        4416506.LOFI.mp3  80
        1855660.LOFI.mp3  119
        3419452.LOFI.mp3  119
        3577631.LOFI.mp3  119
"""
from typing import Optional, TextIO, Tuple

import jams
import librosa
import numpy as np
from deprecated.sphinx import deprecated

from mirdata import annotations, core, download_utils, io

BIBTEX = """@inproceedings{knees2015two,
  title={Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections},
  author={Knees, Peter and Faraldo P{\'e}rez, {\'A}ngel and Boyer, Herrera and Vogl, Richard and B{\"o}ck, Sebastian and H{\"o}rschl{\"a}ger, Florian and Le Goff, Mickael and others},
  booktitle={Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR); 2015 Oct 26-30; M{\'a}laga, Spain.[M{\'a}laga]: International Society for Music Information Retrieval, 2015. p. 364-70.},
  year={2015},
  organization={International Society for Music Information Retrieval (ISMIR)},
}
@inproceedings{SchreiberM18a_Tempo_ISMIR,
  author={Hendrik Schreiber and Meinard M{\"u}ller},
  title={A Crowdsourced Experiment for Tempo Estimation of Electronic Dance Music},
  booktitle={Proceedings of the International Conference on Music Information Retrieval ({ISMIR})},
  address={Paris, France},
  year={2018},
  url-pdf={http://www.tagtraum.com/download/2018_schreiber_tempo_giantsteps.pdf},
}"""

INDEXES = {
    "default": "2.0",
    "test": "2.0",
    "2.0": core.Index(filename="giantsteps_tempo_index_2.0.json"),
}

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="giantsteps-tempo-dataset-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb.zip",
        url="https://github.com/GiantSteps/giantsteps-tempo-dataset/archive/0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb.zip",
        checksum="8fdafbaf505fe3f293bd912c92b72ac8",
    )
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the Giant Steps Tempo dataset are not available
    for download. If you have the Giant Steps audio dataset, place the contents into
    a folder called GiantSteps_tempo with the following structure:
        > GiantSteps_tempo/
            > giantsteps-tempo-dataset-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/
            > audio/
    and copy the folder to {}
"""

LICENSE_INFO = "Creative Commons Attribution Share Alike 4.0 International."


class Track(core.Track):
    """giantsteps_tempo track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        title (str): title of the track
        track_id (str): track id
        annotation_v1_path (str): track annotation v1 path
        annotation_v2_path (str): track annotation v2 path

    Cached Properties:
        genre (dict): Human-labeled metadata annotation
        tempo (list): List of annotations.TempoData, ordered by confidence
        tempo_v2 (list): List of annotations.TempoData for version 2, ordered by confidence
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.annotation_v1_path = self.get_path("annotation_v1")
        self.annotation_v2_path = self.get_path("annotation_v2")

        self.audio_path = self.get_path("audio")

        self.title = self.audio_path.replace(".mp3", "").split("/")[-1].split(".")[0]

    @core.cached_property
    def genre(self) -> Optional[str]:
        return load_genre(self.annotation_v1_path)

    @core.cached_property
    def tempo(self) -> Optional[annotations.TempoData]:
        return load_tempo(self.annotation_v1_path)

    @core.cached_property
    def tempo_v2(self) -> Optional[annotations.TempoData]:
        return load_tempo(self.annotation_v2_path)

    @property
    def audio(self) -> Tuple[np.ndarray, float]:
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
        return jams.load(self.annotation_v1_path)

    def to_jams_v2(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams.load(self.annotation_v2_path)


def load_audio(fhandle: str) -> Tuple[np.ndarray, float]:
    """Load a giantsteps_tempo audio file.

    Args:
        fhandle (str or file-like): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_genre(fhandle: TextIO) -> str:
    """Load genre data from a file.

    Args:
        path (str): path to metadata annotation file

    Returns:
        str: loaded genre data
    """
    annotation = jams.load(fhandle)
    return annotation.search(namespace="tag_open")[0]["data"][0].value


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> annotations.TempoData:
    """Load giantsteps_tempo tempo data from a file ordered by confidence.

    Args:
        fhandle (str or file-like): File-like object or path to tempo annotation file

    Returns:
        annotations.TempoData: Tempo data
    """
    annotation = jams.load(fhandle)

    tempo = annotation.search(namespace="tempo")[0]["data"]

    return annotations.TempoData(
        np.array([[t.time for t in tempo], [t.time + t.duration for t in tempo]]).T,
        "s",
        np.array([t.value for t in tempo]),
        "bpm",
        np.array([t.confidence for t in tempo]),
        "likelihood",
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The giantsteps_tempo dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="giantsteps_tempo",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.giantsteps_tempo.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.giantsteps_tempo.load_genre", version="0.3.4")
    def load_genre(self, *args, **kwargs):
        return load_genre(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.giantsteps_tempo.load_tempo", version="0.3.4")
    def load_tempo(self, *args, **kwargs):
        return load_tempo(*args, **kwargs)
