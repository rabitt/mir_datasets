"""Can I play it? (CIPI) Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The "Can I Play It?" (CIPI) dataset is a specialized collection of 652 classical piano scores, provided in a
    machine-readable MusicXML format and accompanied by integer-based difficulty levels ranging from 1 to 9, as
    verified by expert pianists. Then, it provides embeddings for fingering and expresiveness of the piece. Each
    recording has multiple scores corresponding to it. This dataset focuses exclusively on classical piano music,
    offering a rich resource for music researchers, educators, and students. Developed by the Music Technology Group
    in Barcelona, by P. Ramoneda et al.

    The CIPI dataset facilitates various applications such as the study of musical complexity, the selection of
    appropriately leveled pieces for students, and general research in music education. The dataset, alongside
    embeddings of multiple dimensions of difficulty, has been made publicly available to encourage ongoing innovation
    and collaboration within the music education and research communities.

    The dataset has been published alongside a paper in Expert Systems with Applications Journal.

    The dataset is shared under a Creative Commons Attribution Non Commercial Share Alike 4.0 International License, but
    need to be requested. Please do request the dataset here: https://zenodo.org/records/8037327. The dataset can only
    be used for open research purposes.
"""
import json
import logging
import os
from typing import List, Optional

from smart_open import open

from mirdata import core, jams_utils

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

LICENSE_INFO = "Creative Commons Attribution Non Commercial Share Alike 4.0 International."

DOWNLOAD_INFO = """
    Unfortunately the files of the CIPI dataset are available
    for download upon request here: https://zenodo.org/records/8037327.
    After requesting the dataset, you will receive a link to download the 
    dataset. You must download scores.zip, embeddings.zip and index.json
    copy the files into the folder:
        > cipi/
            > index.json
            > embeddings.zip
            > scores.zip
    unzip embedding.zip and scores.zip and copy the CIPI folder to {}
"""


class Track(core.Track):
    """Can I play it? (CIPI) track class.

    Args:
        track_id (str): track id of the track

    Attributes:
        title (str): title of the track
        book (str): book of the track
        URI (str): URI of the track
        composer (str): name of the author of the track
        track_id (str): track id
        musicxml_paths (list): path to musicxml score. If the music piece contains multiple movents the list will contain multiple paths.
        difficulty_annotation (int): annotated difficulty
        fingering_path (tuple): Path of fingering features from technique dimension computed with ArGNN fingering model. Return of two paths, the right hand and the ones of the left hand. Use torch.load(...) for loading the embeddings.
        expressiveness_path (str): Path of expressiveness features from sound dimension computed with virtuosoNet model.Use torch.load(...) for loading the embeddings.
        notes_path (str): Path of note features from notation dimension. Use torch.load(...) for loading the embeddings.

    Cached Properties:
        scores (list[music21.stream.Score]): music21 scores. If the work is splited in several movements the list will contain multiple scores.
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)
        self._data_home = data_home
        self.fingering_path = (
            self.get_path("rh_fingering"),
            self.get_path("lh_fingering"),
        )
        self.expressiveness_path = self.get_path("expressiveness")
        self.notes_path = self.get_path("notes")

    @property
    def title(self) -> Optional[str]:
        return self._track_metadata["work_name"] if "work_name" in self._track_metadata else None

    @property
    def book(self) -> Optional[str]:
        return self._track_metadata["book"] if "book" in self._track_metadata else None

    @property
    def URI(self) -> Optional[str]:
        return self._track_metadata["URI"] if "URI" in self._track_metadata else None

    @property
    def composer(self) -> Optional[str]:
        return self._track_metadata["composer"] if "composer" in self._track_metadata else None

    @property
    def musicxml_paths(self) -> List[str]:
        return list(self._track_metadata["path"].values()) if "path" in self._track_metadata else []

    @property
    def difficulty_annotation(self) -> int:
        return self._track_metadata["henle"] if "henle" in self._track_metadata else None

    @core.cached_property
    def scores(self) -> list:
        try:
            scores = [load_score(path, self._data_home) for path in self.musicxml_paths]
        except FileNotFoundError:
            raise FileNotFoundError(
                "Some MusicXML files for track id {} not found. "
                "Did you request, download, and store the files as indicated?".format(self.track_id)
            )
        return scores

    def to_jams(self):
        """Get the track's data in jams format.

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
    """Load cipi score in music21 stream.

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


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Can I play it?

    (CIPI) dataset
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
