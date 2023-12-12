"""Indian Art Music Tonic Loader.

.. admonition:: Dataset Info
    :class: dropdown

    This loader includes a combination of six different datasets for the task of Indian Art Music tonic identification.

    These datasets comprise audio excerpts and manually done annotations of the tonic pitch of the lead artist for each audio excerpt.
    Each excerpt is accompanied by its associated editorial metadata. These datasets can be used to develop and evaluate computational
    approaches for automatic tonic identification in Indian art music. These datasets have been used in several articles mentioned below.
    A majority of These datasets come from the CompMusic corpora of Indian art music, for which each recording is associated with a MBID.
    Through the MBID other information can be obtained using the Dunya API.


    These six datasets are used for for the task of tonic identification for Indian Art Music, and can be used for a comparative evaluation.
    To the best of our knowledge these are the largest datasets available for tonic identification for Indian art  music. These datases vary
    in terms of the audio quality, recording period (decade), the number of recordings for Carnatic, Hindustani, male and female singers and
    instrumental and vocal excerpts.

    All the datasets (annotations) are version controlled. The audio files corresponding to these datsets are made available on request
    for only research purposes. See DOWNLOAD_INFO of this loader.

    The tonic annotations are availabe both in tsv and json format. The loader uses the JSON formatted annotations.

    .. code-block::

        'ID': {
            'artist': <name of the lead artist if available>,
            'filepath': <relative path to the audio file>,
            'gender': <gender of the lead singer if available>,
            'mbid': <musicbrainz id when available>,
            'tonic': <tonic in Hz>,
            'tradition': <Hindustani or Carnatic>,
            'type': <vocal or instrumental>
        }

    where keys of the main dictionary are the filepaths to the audio files (feature path is exactly the same with a different extension
    of the file name).

    Despite not being loaded in this dataloader, the dataset includes features, which may be integrated to the loader in future releases. However
    these features may be easily computed following the instructions in the related paper. See BIBTEX.

    There are a total of 2161 audio excerpts, and while the CM collection includes aproximately 50% Carnatic and 50% Hindustani recordings, IITM and
    IISc collections are 100% Carnatic music. The excerpts vary a lot in duration. See [this webpage](https://compmusic.upf.edu/iam-tonic-dataset)
    for a detailed overview of the datasets.

    If you have any questions or comments about the dataset, please feel free to email: [sankalp (dot) gulati (at) gmail (dot) com], or
    [sankalp (dot) gulati (at) upf (dot) edu].
"""

import glob
import json
import os

import librosa
from smart_open import open

from mirdata import core, download_utils, jams_utils

BIBTEX = """@article{Gulati2014,
    author = {Gulati, S. and Bellur, A. and Salamon, J. and Ranjani, H. G. and Ishwar, V. and Murthy, H. A. and Serra, X.},
    journal = {Journal of New Music Research},
    pages = {55--73},
    volume = {43},
    number = {01},
    title = {{Automatic Tonic Identification in Indian Art Music: Approaches and Evaluation}},
    year = {2014}
}"""


INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="compmusic_indian_tonic_1.0.json"),
}

REMOTES = {
    "remote_data": download_utils.RemoteFileMetadata(
        filename="indian_art_music_tonic_1.0.zip",
        url="https://zenodo.org/record/1257114/files/indian_art_music_tonic_1.0.zip?download=1",
        checksum="47493d59d400dac459444b7a3bd2c572",  # the md5 checksum
    ),
}

DOWNLOAD_INFO = """
    The audio of this dataset is private, and it is only shared for research purposes. Please refer to:
    https://zenodo.org/record/7342372, request the audios clearly explaning why and how are you planning 
    to use it, and then simply move the "audio" folders to the respective center ID. An example here:
    take indian_art_music_tonic_1.0_audio/CM/audio and move it inside indian_art_music_tonic_1.0/CM, and so on.
"""


LICENSE_INFO = "Creative Commons Attribution Non Commercial Share Alike 4.0 International."


class Track(core.Track):
    """CompMusic Tonic Dataset track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.

    Attributes:
        track_id (str): track id
        audio_path (str): audio path

    Cached Properties:
        tonic (float): tonic annotation
        artist (str): performing artist
        gender (str): gender of the recording artists
        mbid (str): MusicBrainz ID of the piece (if available)
        type (str): type of piece (vocal, instrumental, etc.)
        tradition (str): tradition of the piece (Carnatic or Hindustani)
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

        self.audio_path = self.get_path("audio")

    @property
    def audio(self):
        """The track's audio.

        Returns:
           * np.ndarray - audio signal
           * float - sample rate
        """
        return load_audio(self.audio_path)

    @core.cached_property
    def tonic(self):
        return self._track_metadata.get("tonic")

    @core.cached_property
    def artist(self):
        return self._track_metadata.get("artist")

    @core.cached_property
    def gender(self):
        return self._track_metadata.get("gender")

    @core.cached_property
    def mbid(self):
        return self._track_metadata.get("mbid")

    @core.cached_property
    def type(self):
        return self._track_metadata.get("type")

    @core.cached_property
    def tradition(self):
        return self._track_metadata.get("tradition")

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={"tonic": self.tonic, "artist": self.artist},
        )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(audio_path):
    """Load a Indian Art Music Tonic audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(audio_path, sr=44100, mono=False)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The compmusic_indian_tonic dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="compmusic_indian_tonic",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        centers = ["CM", "IISc", "IITM"]
        meta_files = []
        for center in centers:
            meta_files = meta_files + glob.glob(
                os.path.join(
                    self.data_home,
                    "indian_art_music_tonic_1.0",
                    center,
                    "annotations",
                    "*.json",
                )
            )
        meta_files = [x for x in meta_files if "IITM1" not in x]

        metadata = {}
        try:
            for meta in meta_files:
                with open(meta, "r") as fhandle:
                    data = json.load(fhandle)
                    if "IITM" not in meta:
                        for k in list(data.keys()):
                            idx = k.split("/")[-1].replace(".mp3", "")
                            metadata[idx] = {
                                "tonic": float(data[k]["tonic"]),
                                "artist": data[k]["artist"],
                                "gender": data[k]["gender"],
                                "mbid": data[k]["mbid"],
                                "type": data[k]["type"],
                                "tradition": data[k]["tradition"],
                            }
                    else:
                        for k in list(data.keys()):
                            idx = k.split("/")[-1].replace(".mp3", "")
                            metadata[idx] = {
                                "tonic": float(data[k]["tonic"]),
                                "artist": data[k]["artist"],
                                "gender": data[k]["gender"],
                                "mbid": data[k]["mbid"],
                                "type": data[k]["type"],
                                "tradition": data[k]["tradition"],
                            }

        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata
