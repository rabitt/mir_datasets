"""Can I play it? (CIPI) Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The "Can I Play It?" (CIPI) dataset is a specialized collection of 652 classical piano scores, provided in a
    machine-readable MusicXML format and accompanied by integer-based difficulty levels ranging from 1 to 9, as
    verified by expert pianists. Developed by the Music Technology Group in Barcelona, this dataset focuses
    exclusively on classical piano music, offering a rich resource for music researchers, educators, and students.

    The CIPI dataset facilitates various applications such as the study of musical complexity, the selection of
    appropriately leveled pieces for students, and general research in music education. The dataset, alongside
    embeddings of multiple dimensions of difficulty, has been made publicly available to encourage ongoing innovation
    and collaboration within the music education and research communities.
"""
import json
import logging
import os
import pdb
import pickle
from typing import Optional, TextIO, List

import smart_open
from smart_open import open

from deprecated.sphinx import deprecated

from mirdata import core, io, jams_utils, download_utils

try:
    import music21
except ImportError:
    logging.error(
        "In order to use cipi you must have music21 installed. "
        "Please reinstall mirdata using `pip install 'mirdata[cipi]'"
    )
    raise ImportError

BIBTEX = """
@article{Ramoneda2024,
  author    = {Pedro Ramoneda and Dasaem Jeong and Vsevolod Eremenko and Nazif Can Tamer and Marius Miron and Xavier Serra},
  title     = {Combining Piano Performance Dimensions for Score Difficulty Classification},
  journal   = {Expert Systems with Applications},
  volume    = {238},
  pages     = {121776},
  year      = {2024},
  doi       = {10.1016/j.eswa.2023.121776},
  url       = {https://doi.org/10.1016/j.eswa.2023.121776}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="cipi_index_1.0.json"),
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """
    Unfortunately the files of the CIPI dataset are available
    for download upon request. After requesting the dataset, you will receive a
    link to download the dataset. You must download scores.zip, embeddings.zip and index.json
    copy the files into the folder:
        > cipi/
            > index.json
            > embeddings.zip
            > scores.zip
    unzip embedding.zip and scores.zip and copy the CIPI folder to {}
"""


class Track(core.Track):
    """Can I play it? (CIPI) track class

    Args:
        track_id (str): track id of the track

    Attributes:
        title (str): title of the track
        book (str): book of the track
        URI (str): URI of the track
        composer (str): name of the author of the track
        track_id (str): track id
        musicxml_paths (list): path to musicxml score. If the music piece contains multiple movents the list will contain multiple paths.
        difficulty annotation (str): annotated difficulty

    Cached Properties:
        Fingering path (str): Path of fingering features from technique dimension computed with ArGNN fingering model. Return of two paths, embeddings of the right hand and the ones of the left hand. Use torch.load(...) for loading the embeddings.
        Expressiviness path (str): Path of expressiviness features from sound dimension computed with virtuosoNet model.Use torch.load(...) for loading the embeddings.
        Notes path (str): Path of note features from notation dimension. Use torch.load(...) for loading the embeddings.
        scores (list[music21.stream.Score]): music21 scores. If the work is splited in several movements the list will contain multiple scores.
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)
        self._data_home = data_home

    @property
    def title(self) -> str:
        return (
            self._track_metadata["work_name"]
            if "work_name" in self._track_metadata
            else None
        )

    @property
    def book(self) -> str:
        return self._track_metadata["book"] if "book" in self._track_metadata else None

    @property
    def URI(self) -> str:
        return self._track_metadata["URI"] if "URI" in self._track_metadata else None

    @property
    def composer(self) -> str:
        return (
            self._track_metadata["composer"]
            if "composer" in self._track_metadata
            else None
        )

    @property
    def musicxml_paths(self) -> List[str]:
        return (
            list(self._track_metadata["path"].values())
            if "path" in self._track_metadata
            else []
        )

    @property
    def difficulty_annotation(self) -> str:
        return (
            self._track_metadata["henle"] if "henle" in self._track_metadata else None
        )

    def _check_embedding(self, fpath, file_type: str) -> str:
        """
        Verifies the existence of an embedding file and returns its path.

        Args:
            fpaths (str): The path to the embedding file.
            file_type (str): The type of the embedding file.

        Returns:
            str: The path to the embedding file.

        Raises:
            FileNotFoundError: If the embedding file does not exist.
        """
        embedding_path = os.path.join(
            self._data_home,
            fpath,
            f"{self.track_id}_{file_type}.ext",
        )  # Adjust the path and extension as necessary
        try:
            with smart_open.open(embedding_path):
                return embedding_path
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{file_type} embedding {embedding_path} for track {self.track_id} not found. "
                "Did you run .download()?"
            )

    @core.cached_property
    def fingering(self) -> tuple:
        path_rh = self.get_path("rh_fingering")
        path_lh = self.get_path("lh_fingering")
        return self._check_embedding(path_rh, "Fingering"), self._check_embedding(
            path_lh, "Fingering"
        )

    @core.cached_property
    def expressiviness(self) -> str:
        path = self.get_path("expressiviness")
        return self._check_embedding(path, "Expressiviness")

    @core.cached_property
    def notes(self) -> str:
        path = self.get_path("notes")
        return self._check_embedding(path, "Expressiviness")

    @core.cached_property
    def scores(self) -> list:
        try:
            scores = [load_score(path, self._data_home) for path in self.musicxml_paths]
        except FileNotFoundError:
            raise FileNotFoundError(
                "MusicXML file {} for track {} not found. "
                "Did you run .download()?".format(self.musicxml_paths, self.track_id)
            )
        return scores

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            metadata={
                "title": self.title,
                "artist": self.composer,
                "duration": 0.0,
                "book": self.book,
                "URI": self.URI,
                "composer": self.composer,
                "track_id": self.track_id,
                "musicxml_paths": self.musicxml_paths,
                "difficulty_annotation": self.difficulty_annotation,
            }
        )


def load_score(
    fhandle: str, data_home: str = "tests/resources/mir_datasets/cipi"
) -> music21.stream.Score:
    """Load cipi score in music21 stream

    Args:
        fhandle (str): path to MusicXML score
        data_home (str): path to cipi dataset

    Returns:
        music21.stream.Score: score in music21 format
    """
    try:
        score = music21.converter.parse(os.path.join(data_home, fhandle))
    except:
        raise FileNotFoundError("File {} not found.".format(fhandle))
    return score


@deprecated(
    reason="convert_and_save_to_midi is deprecated and will be removed in a future version",
    version="0.3.4",
)
def convert_and_save_to_midi(fpath: str):
    """convert to midi file and return the midi path

    Args:
        fpath (str): path to score file

    Returns:
        str: midi file path

    """
    midi_path = os.path.splitext(fpath)[0] + ".mid"
    score = music21.converter.parse(fpath)
    score.write("midi", fp=midi_path)
    return midi_path


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Can I play it? (CIPI) dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="cipi",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            license_info=LICENSE_INFO,
            download_info=DOWNLOAD_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "index.json")
        try:
            with open(metadata_path, "r") as fhandle:
                metadata_index = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Metadata {metadata_path} not found. Did you download the files?"
            )
        return dict(metadata_index)
