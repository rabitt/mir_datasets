"""4MuLA-small Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset is a small version extracted by 4MuLA dataset:

    .. code-block:: latex

        "4MuLA: A Multitask, Multimodal, and Multilingual Dataset of Music Lyrics and Audio Features."
        by Angelo Cesar Mendes da Silva, Diego Furtado Silva, and Ricardo Marcondes Marcacini.
        In Proceedings of the Brazilian Symposium on Multimedia and the Web (WebMedia '20). 2020.
        DOI:https://doi.org/10.1145/3428658.3431089


    The dataset consists of 9661 music recordings represented by melspectrogram and metadata. It
    conveys 51 genres and 491 artists. The audios file are not available. The metadata includes the following attributes:
    ['music_id', 'music_name', 'music_lang', 'music_lyrics', 'art_id',
    'art_name', 'art_rank', 'main_genre', 'related_genre', 'related_art',
    'related_music', 'musicnn_tags', 'melspectrogram']

"""
import numpy as np
import os
import pyarrow.parquet as pq
from mirdata.download_utils import RemoteFileMetadata
from mirdata import core, jams_utils

BIBTEX = """@inproceedings{10.1145/3428658.3431089,
  author = {da Silva, Angelo Cesar Mendes and Silva, Diego Furtado and Marcacini, Ricardo Marcondes},
  title = {4MuLA: A Multitask, Multimodal, and Multilingual Dataset of Music Lyrics and Audio Features},
  year = {2020},
  isbn = {9781450381963},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3428658.3431089},
  doi = {10.1145/3428658.3431089},    
  booktitle = {Proceedings of the Brazilian Symposium on Multimedia and the Web},
  pages = {145–148},
  numpages = {4},
  keywords = {latin musical dataset, music dataset, multimodal musical dataset},
  location = {S\~{a}o Lu\'{\i}s, Brazil},
  series = {WebMedia '20}
}"""

REMOTES = {
    "dataset": RemoteFileMetadata(
        filename="4mula_small.parquet",
        url="https://zenodo.org/record/4636802/files/4mula_small.parquet?download=1",
        checksum="30210cf6f52449c8d0670fc0942410c4",
        destination_dir=".",
    )
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International."

DOWNLOAD_INFO = """
  This loader is designed to load the melspectrogram and metadata, as it is available for download.
  Unfortunately the audio files of the 4MULA dataset are not available
  for download. You can more details and follow the updates in the following repository: 
  ==> https://github.com/4mulaDataset/4mula    
"""

DEFAULT_COLUMNS = [
    "music_id",
    "music_name",
    "music_lang",
    "music_lyrics",
    "art_id",
    "art_name",
    "art_rank",
    "main_genre",
    "related_genre",
    "related_art",
    "related_music",
    "musicnn_tags",
    "melspectrogram",
]


class Track(core.Track):
    """small_4mula track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/small_4mula`

    Attributes:
        music_id (str): track id
        music_name (str): music name
        music_lang (str): music language
        music_lyrics (str): lyrics
        art_id (str): artist id
        art_name (str): artist name
        art_rank (int): artist rank
        main_genre (str): artist main genre
        related_genre (list): artist related genre
        related_art (list): others artist related to artist
        related_music (list): others music related to track
        musicnn_tags (list): top-3 tags got by Musicnn
        melspectrogram (numpy.ndarray): melspectrogram extracted by librosa using default params

    """

    def __init__(
        self,
        track_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.annotation_path = os.path.join(
            self._data_home, self._track_paths["annotation"][0]
        )
        self.melspectrogram_path = os.path.join(
            self._data_home, self._track_paths["melspectrogram"][0]
        )

    @property
    def music_id(self):
        return self._track_metadata.get("music_id")

    @property
    def music_name(self):
        return self._track_metadata.get("music_name")

    @property
    def music_lang(self):
        return self._track_metadata.get("music_lang")

    @property
    def music_lyrics(self):
        return self._track_metadata.get("music_lyrics")

    @property
    def art_id(self):
        return self._track_metadata.get("art_id")

    @property
    def art_name(self):
        return self._track_metadata.get("art_name")

    @property
    def art_rank(self):
        return self._track_metadata.get("art_rank")

    @property
    def main_genre(self):
        return self._track_metadata.get("main_genre")

    @property
    def related_art(self):
        return eval(self._track_metadata.get("related_art"))

    @property
    def related_genre(self):
        return eval(self._track_metadata.get("related_genre"))

    @property
    def related_music(self):
        return eval(self._track_metadata.get("related_music"))

    @property
    def musicnn_tags(self):
        return eval(self._track_metadata.get("musicnn_tags"))

    @property
    def load_spectrogram(self):
        return np.load(self.melspectrogram_path, allow_pickle=True)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        metadata = self._track_metadata
        metadata.update({"duration": 30})  # approximation

        return jams_utils.jams_converter(
            spectrogram_path=self.melspectrogram_path, metadata=metadata
        )


def load_melspectrogram(fhandle: str) -> np.ndarray:
    """Load a small_4mula dataset melspectrogram.

    Args:
        fhandle (str): path pointing to the dataset file

    Returns:
        numpy.ndarray: melspectrogram

    """
    if not os.path.isfile(str(fhandle)):
        raise FileNotFoundError("Dataset not found. Did you run .download()?")
    df = pq.ParquetFile(fhandle)
    df = df.read(columns=["melspectrogram"])
    return df.to_pandas(split_blocks=True, self_destruct=True).melspectrogram.values


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The small version of 4MuLA dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="small_4mula",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "4mula_small.parquet")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
        columns = [
            "music_id",
            "music_name",
            "music_lang",
            "music_lyrics",
            "art_id",
            "art_name",
            "art_rank",
            "main_genre",
            "related_genre",
            "related_art",
            "related_music",
            "musicnn_tags",
        ]
        df = pq.ParquetFile(metadata_path)
        df = df.read(columns=columns).to_pydict()

        metadata = dict()
        for i, j in zip(df["music_id"], range(len(df["music_id"]))):
            metadata[i] = {
                "music_id": df["music_id"][j],
                "music_name": df["music_name"][j],
                "music_lang": df["music_lang"][j],
                "music_lyrics": df["music_lyrics"][j],
                "art_id": df["art_id"][j],
                "art_name": df["art_name"][j],
                "art_rank": df["art_rank"][j],
                "main_genre": df["main_genre"][j],
                "related_genre": df["related_genre"][j],
                "related_art": df["related_art"][j],
                "related_music": df["related_music"][j],
                "musicnn_tags": df["musicnn_tags"][j],
            }
        return metadata

    @core.copy_docs(load_melspectrogram)
    def load_spectrogram(self, *args, **kwargs):
        return load_melspectrogram(*args, **kwargs)
