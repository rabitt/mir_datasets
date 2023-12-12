"""IDMT-SMT-Audio-Effects Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    IDMT-SMT-Audio-Effects is a large database for automatic detection of audio effects in recordings of electric guitar and bass and
    related signal processing. The overall duration of the audio material is approx. 30 hours.

    The dataset consists of 55044 WAV files (44.1 kHz, 16bit, mono) with single recorded notes:

    20592 monophonic bass notes
    20592 monophonic guitar notes
    13860 polyphonic guitar sounds
    Overall, 11 different audio effects are incorporated:
    feedback delay, slapback delay, reverb, chorus, flanger, phaser, tremolo, vibrato,
    distortion, overdrive, no effect (unprocessed notes/sounds)

    2 different electric guitars and 2 different electric bass guitars, each with two different pick-up settings and
    up to three different plucking styles (finger plucked - hard, finger plucked - soft, picked) were used for recording.
    The notes cover the common pitch range of a 4-string bass guitar from E1 (41.2 Hz) to G3 (196.0 Hz) or the common
    pitch range of a 6-string electric guitar from E2 (82.4 Hz) to E5 (659.3 Hz).
    Effects processing was performed using a digital audio workstation and a variety of mostly freely available effect
    plugins.

    To organize the database, lists in XML format are used, which record all relevant information and are provided with
    the database as well as a summary of the used effect plugins and parameter settings.

    In addition, most of this information is also encoded in the first part of the file name of the audio files using
    a simple alpha-numeric encoding scheme. The second part of the file name contains unique identification numbers.
    This provides an option for fast and flexible structuring of the data for various purposes.

    DOI
    10.5281/zenodo.7544032
"""
import os
import xml.etree.ElementTree as ET
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
from deprecated.sphinx import deprecated
from smart_open import open

from mirdata import core, download_utils, io, jams_utils

BIBTEX = """
@dataset{stein_michael_2023_7544032,
  author       = {Stein, Michael},
  title        = {IDMT-SMT-Audio-Effects Dataset},
  month        = jan,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.7544032},
  url          = {https://doi.org/10.5281/zenodo.7544032}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="idmt_smt_audio_effects_index.json"),
}

REMOTES = {
    "full_dataset": download_utils.RemoteFileMetadata(
        filename="IDMT-SMT-AUDIO-EFFECTS.zip",
        url="https://zenodo.org/record/7544032/files/IDMT-SMT-AUDIO-EFFECTS.zip?download=1",
        checksum="91e845a1b347352993ebd5ba948d5a7c",  # the md5 checksum
        destination_dir=".",  # relative path for where to unzip the data, or None
        unpack_directories=[""],
    ),
}

DOWNLOAD_INFO = """
        This loader will create the following folders in the dataset data_home path:
            > idmt_smt_audio_effects/
                > Bass monophon/
                > Bass monophon2/
                > Gitarre monophon/
                > Gitarre monophon2/
                > Gitarre polyphon/
                > Gitarre polyphon2/
"""

LICENSE_INFO = """
Creative Commons BY-NC-ND 4.0.
https://creativecommons.org/licenses/by-nc-nd/4.0/
"""


class Track(core.Track):
    """IDMT-SMT-Audio-Effects track class.

    Args:
        track_id (str): track id of the track.
        data_home (str): Local path where the dataset is stored.
        dataset_name (str): Name of the dataset.
        index (Dict): Index dictionary.
        metadata (Dict): Metadata dictionary.

    Attributes:
        audio_path (str): path to audio file.
        instrument (str): instrument used to record the track.
        midi_nr (int): midi number of the note.
        fx_group (int): effect group number.
        fx_type (int): effect type number.
        fx_setting (int): effect setting number.
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )
        """
        Args:
            track_id (str): track id of the track
            data_home (str): Local path where the dataset is stored. If `None`, looks for the data in the default directory, `~/mir_datasets/idmt_smt_audio_effects`
            dataset_name (str): Name of the dataset.
            index (Dict): Index dictionary.
            metadata (Dict): Metadata dictionary.
        """
        self.audio_path = self.get_path("audio")

    @property
    def instrument(self):
        return self._track_metadata["instrument"]

    @property
    def midi_nr(self):
        return self._track_metadata["midi_nr"]

    @property
    def fx_group(self):
        return self._track_metadata["fx_group"]

    @property
    def fx_type(self):
        return self._track_metadata["fx_type"]

    @property
    def fx_setting(self):
        return self._track_metadata["fx_setting"]

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        try:
            return load_audio(self.audio_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Audio file {self.audio_path} not found. Did you run .download?"
            )

    def to_jams(self):
        """Get the track's data in jams format.

        Returns:
            jams.JAMS: the track's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata=self._track_metadata,
        )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a IDMT-SMT-Audio Effect track.

    Args:
        fhandle (Union[str, BinaryIO]): Path to audio file or file-like object.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=44100, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The IDMT-SMT-Audio Effect dataset.

    Args:
        data_home (str): Directory where the dataset is located or will be downloaded.
        version (str): Dataset version. Default is "default".

    Attributes:
        name (str): Name of the dataset.
        track_class (Type[core.Track]): Track type.
        bibtex (str): BibTeX citation.
        indexes (Dict[str, core.Index]): Available versions.
        remotes (Dict[str, download_utils.RemoteFileMetadata]): Data to be downloaded.
        download_info (str): Instructions for downloading the dataset.
        license_info (str): Dataset license.
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="idmt_smt_audio_effects",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        """Return a dictionary containing metadata information parsed from XML
        files.

        Returns:
            dict: A dictionary containing metadata information parsed from XML files.

        Raises:
            FileNotFoundError: If metadata file not found.
            ValueError: If there's an error parsing the XML file.
            Exception: For unexpected errors during processing.
        """
        metadata = dict()
        metadata = {
            "fileID": {
                "list_id": str,
                "instrument": str,
                "midi_nr": str,
                "fx_group": int,
                "fx_type": int,
                "fx_setting": int,
            }
        }

        xml_files_count = 0

        for root, _, files in os.walk(self.data_home):
            for file in files:
                if file.endswith(".xml"):
                    xml_files_count += 1
                    xml_path = os.path.join(root, file)
                    try:
                        with open(xml_path, "r") as fhandle:
                            tree = ET.parse(fhandle)

                    except ET.ParseError:
                        raise ValueError(
                            f"Error parsing XML file {xml_path}. The file may be corrupted or not abailable, make sure you have all files."
                        )

                    root_xml = tree.getroot()
                    listID = root_xml.find("listinformation/listID").text

                    for audiofile in root_xml.findall("audiofile"):
                        name = audiofile.find("fileID").text
                        instrument = audiofile.find("instrument").text
                        midinr = audiofile.find("midinr").text
                        fxgroup = audiofile.find("fxgroup").text
                        fxtype = audiofile.find("fxtype").text
                        fxsetting = audiofile.find("fxsetting").text

                        metadata[name] = {
                            "list_id": listID,
                            "instrument": instrument,
                            "midi_nr": int(midinr),
                            "fx_group": int(fxgroup),
                            "fx_type": int(fxtype),
                            "fx_setting": int(fxsetting),
                        }

        if xml_files_count == 0:
            raise FileNotFoundError(
                f"No XML files found in {self.data_home}. Did you run .download?"
            )
        return metadata
