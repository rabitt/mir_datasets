# -*- coding: utf-8 -*-
"""TinySOL Dataset Loader.

TinySOL is an audio dataset of 1533 isolated notes from twelve instruments.

Attributes:
    DATA.index (dict): {track_id: track_data}.
        track_id is a JSON data loaded from 'indexes/'

    DATASET_DIR (str): The directory name for TinySOL.
        Set to `'TinySOL'`.

    DATA.metadata (dict): The metadata of TinySOL.
"""                                               # TODO complete this docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import jams
import librosa
import logging
import os

import mirdata.utils as utils
import mirdata.download_utils as download_utils
import mirdata.jams_utils as jams_utils

DATASET_DIR = "TinySOL"
ANNOTATION_REMOTE = download_utils.RemoteFileMetadata(
    filename="TinySOL_metadata.csv",
    url="",                                                         # TODO FILL
    checksum="",                                                    # TODO FILL
    destination_dir="annotation",
)
AUDIO_REMOTE = download_utils.RemoteFileMetadata(
    filename="TinySOL.tar.gz",
    url="",                                                          # TODO FILL
    checksum="",                                                     # TODO FILL
    destination_dir="audio",
)
STRING_ROMAN_NUMERALS = {1: "I", 2: "II", 3: "III", 4: "IV"}


def _load_metadata(data_home):
    metadata_path = os.path.join(
        data_home, "annotation", "TinySOL_metadata.csv"
    )

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    metadata_index = {}
    with open(metadata_path, "r") as fhandle:
        csv_reader = csv.reader(fhandle, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            metadata_index[row[0]] = {
                 "Family":                  row[1],
                 "Instrument (abbr.)":      row[2],
                 "Instrument (in full)":    row[3],
                 "Technique (abbr.)":       row[4],
                 "Technique (in full)":     row[5],
                 "Pitch":                   row[6],
                 "Pitch ID":                row[7],
                 "Dynamics":                row[8],
                 "Dynamics ID":             row[9],
            }
            if len(row)==11:
                metadata_index[row[0]]["String ID"] = row[10]

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = utils.LargeData("tinysol_index.json", _load_metadata)


class Track(object):
    """TinySOL track class

    Args:
        track_id (str): ID of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        family (str): instrument family encoded by its English name
        instrument_abbr (str): instrument abbreviation
        instrument_full (str): instrument encoded by its English name
        technique_abbr (str): playing technique abbreviation
        technique_full (str): playing technique encoded by its English name
        pitch (str): string containing English pitch class and octave number
        pitch_id (int): MIDI note index, where middle C ("C4") corresponds to 60
        dynamics (str): dynamics abbreviation. Ex: pp, mf, ff, etc.
        dynamics_id (int): pp=0, p=1, mf=2, f=3, ff=4
        string_id (int or None): string ID. By musical convention, the first
        string is the highest. On wind instruments, this is replaced by `None`.
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                "{} is not a valid track ID in TinySOL".format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                 "Family":                  None,
                 "Instrument (abbr.)":      None,
                 "Instrument (in full)":    None,
                 "Technique (abbr.)":       None,
                 "Technique (in full)":     None,
                 "Pitch":                   None,
                 "Pitch ID":                None,
                 "Dynamics":                None,
                 "Dynamics ID":             None,
                 "String ID":               None,
            }

        self.audio_path = os.path.join(
            self._data_home, self._track_paths["audio"][0])

        self.family = self._track_metadata["Family"]
        self.instrument_abbr = self._track_metadata["Instrument (abbr.)"]
        self.instrument_full = self._track_metadata["Instrument (in full)"]
        self.technique_abbr = self._track_metadata["Technique (abbr.)"]
        self.technique_full = self._track_metadata["Technique (in full)"]
        self.pitch = self._track_metadata["Pitch"]
        self.pitch_id = self._track_metadata["Pitch ID"]
        self.dynamics = self._track_metadata["Dynamics"]
        self.dynamics_id = self._track_metadata["Dynamics ID"]
        if "String ID" in self._track_metadata:
            self.string_id = self._track_metadata["String ID"]
        else
            self.string_id = None

    def __repr__(self):

        if self.string_id:
            repr_string = (
                "TinySOL Track(instrument={}, technique={}, "
                + "pitch={}, dynamics={}, string={})"
            ).format(
                self.instrument_full,
                self.technique_full,
                self.pitch,
                self.dynamics,
                STRING_ROMAN_NUMERALS[self.string_id],
            )
        else:
            repr_string = (
                "TinySOL Track(instrument={}, technique={}, "
                + "pitch={}, dynamics={}, string={})"
            ).format(
                self.instrument_full,
                self.technique_full,
                self.pitch,
                self.dynamics,
            )
        return repr_string

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=22050, mono=True)

    def to_jams(self):
        return jams_utils.jams_converter(
            metadata=self._track_metadata)             # TODO pass a note_data?


# def download(data_home=None):
#     """Download TinySOL.
#
#     Args:
#         data_home (str): Local path where the dataset is stored.
#             If `None`, looks for the data in the default directory, `~/mir_datasets`
#     """
#     if data_home is None:
#         data_home = utils.get_default_dataset_path(DATASET_DIR)
#
#     download_utils.downloader(
#         data_home,
#         tar_downloads=[AUDIO_REMOTE],
#         file_downloads=[ANNOTATION_REMOTE],
#         cleanup=True,
#     )


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def load(data_home=None):
    """Load TinySOL
    Args:
        data_home (str): Local path where TinySOL is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    tinysol_data = {}
    for key in DATA.index.keys():
        tinysol_data[key] = Track(key, data_home=data_home)
    return tinysol_data


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Cella, Carmine Emanuele, et al., "OrchideaSOL: A dataset of extended
instrumental techniques for computer-aided orchestration". Under review, 2020.

========== Bibtex ==========
@inproceedings{cella2020preprint,
author={Cella, Carmine Emanuele and Ghisi, Daniele and Lostanlen, Vincent and
Lévy, Fabien and Fineberg, Joshua and Maresz, Yan},
title={{OrchideaSOL}: {A} dataset of extended
instrumental techniques for computer-aided orchestration},
bootktitle={Under review},
year={2020}}
"""
    print(cite_data)
