"""BAF Loader.

.. admonition:: Dataset Info
    :class: dropdown

    BAF dataset is only available upon request. To download the audio request
    access in this link: https://doi.org/10.5281/zenodo.6868083. Then unzip the
    audio into the baf general dataset folder for the rest of annotations and
    files. Please include, in the justification field, your academic
    affiliation (if you have one) and a brief description of your research
    topics and why you would like to use this dataset.

    Overview

    Broadcast Audio Fingerprinting dataset is an open, available upon request,
    annotated dataset for the task of music monitoring in broadcast. It
    contains 2,000 tracks from Epidemic Sound's private catalogue as reference
    tracks that represent 74 hours. As queries, it contains over 57 hours of TV
    broadcast audio from 23 countries and 203 channels distributed with 3,425
    one-min audio excerpts.

    It has been annotated by six annotators in total and each query has been
    cross-annotated by three of them obtaining high inter-annotator agreement
    percentages, which validates the annotation methodology and ensures the
    reliability of the annotations.

    Purpose of the dataset

    This dataset aims to become the standard dataset to evaluate Audio
    Fingerprinting algorithms since it's built on real data, without the use of
    any data-augmentation techniques. It is also the first dataset to address
    background music fingerprinting, which is a real problem in royalties
    distribution.

    Dataset use

    This dataset is available for conducting non-commercial research related to
    audio analysis. It shall not be used for music generation or music
    synthesis.

    About the data

    - Sampling frequency: 8 kHz
    - Bit-depth: 16 bit
    - Number of channels: 1
    - Encoding: pcm_s16le
    - Audio format: .wav

    Annotations mark which tracks sound (either in foreground or background) in
    each query (if any) and also the specific times where it starts and ends
    sound in the query. Note that there are 88 queries that doesn't have any
    matches/annotations .

    For more information check the dedicated Github repository:
    https://github.com/guillemcortes/baf-dataset and the dataset datasheet
    included in the files.

    Ownership of the data

    Next, we specify the ownership of all the data included in BAF: Broadcast
    Audio Fingerprinting dataset. For licensing information, please refer to
    the “License” section.

    Reference tracks

    The reference tracks are owned by Epidemic Sound AB, which has given a
    worldwide, revocable, non-exclusive, royalty-free licence to use and
    reproduce this data collection consisting of 2,000 low-quality monophonic
    8kHz downsampled audio recordings.

    Query tracks

    The query tracks come from publicly available TV broadcast emissions so the
    ownership of each recording belongs to the channel that emitted the
    content. We publish them under the right of quotation provided by the Berne
    Convention.

    Annotations

    Guillem Cortès together with Alex Ciurana and Emilio Molina from BMAT Music
    Licensing S.L. have managed the annotation therefore the annotations belong
    to BMAT.

    Accessing the dataset

    The dataset is available upon request. Please include, in the justification
    field, your academic affiliation (if you have one) and a brief description
    of your research topics and why you would like to use this dataset. Bear in
    mind that this information is important for the evaluation of every access
    request.

    License

    .. code-block:: latex

        Given the different ownership of the elements of the dataset, the
        dataset is licensed under the following conditions:
            * User's access request
            * Research only, non-commercial purposes
            * No adaptations nor derivative works
            * Attribution to Epidemic Sound and the authors as it is indicated
                in the ”citation” section.

    Acknowledgments

    With the support of Ministerio de Ciencia Innovación y universidades
    through Retos-Colaboración call, reference: RTC2019-007248-7, and also with
    the support of the Industrial Doctorates Plan of the Secretariat of
    Universities and Research of the Department of Business and Knowledge of
    the Generalitat de Catalunya. Reference: DI46-2020.
"""
import os
from string import Template
from typing import Optional, Tuple

import librosa
import numpy as np
import pandas as pd

from mirdata import annotations, core, jams_utils

BIBTEX = """@inproceedings{cortes2022BAF,
  author       = {Guillem Cortès and
                  Alex Ciurana and
                  Emilio Molina and
                  Marius Miron and
                  Owen Meyers and
                  Joren Six and
                  Xavier Serra},
  title        = {BAF: An audio fingerprinting dataset for broadcast monitoring},
  booktitle    = {Proceedings of the 23rd International Society for Music Information Retrieval Conference},
  year         = 2022,
  pages        = {908-916},
  publisher    = {ISMIR},
  address      = {Bengaluru, India},
  month        = dec,
  venue        = {Bengaluru, India},
  doi          = {10.5281/zenodo.7316812},
  url          = {https://doi.org/10.5281/zenodo.7372162}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="baf_index_1.0.json"),
}

REMOTES = None

DOWNLOAD_INFO = (
    "BAF dataset is only available upon request. To download the audio "
    "request access in this link: https://doi.org/10.5281/zenodo.6868083. "
    "Then unzip the audio into the baf general dataset folder for the rest of "
    "annotations and files. Please include, in the justification field, your "
    "academic affiliation (if you have one) and a brief description of your "
    "research topics and why you would like to use this dataset.\n"
    "    baf/\n"
    "    ├── baf_datasheet.pdf\n"
    "    ├── annotations.csv\n"
    "    ├── changelog.md\n"
    "    ├── cross_annotations.csv\n"
    "    ├── queries_info.csv\n"
    "    ├── queries\n"
    "    │   ├── query_0001.wav\n"
    "    │   ├── query_0002.wav\n"
    "    │   ├── …\n"
    "    │   └── query_3425.wav\n"
    "    ├── queries_info.csv\n"
    "    └── references\n"
    "        ├── ref_0001.wav\n"
    "        ├── ref_0002.wav\n"
    "        ├── …\n"
    "        └── ref_2000.wav\n"
)

LICENSE_INFO = (
    "Given the different ownership of the elements of the dataset, the "
    "dataset is licensed under the following conditions:\n"
    "    * User's access request\n"
    "    * Research only, non-commercial purposes\n"
    "    * No adaptations nor derivative works\n"
    "    * Attribution to Epidemic Sound and the authors as it is indicated "
    "in the ”citation” section.\n"
)

#: Tag units
TAG_UNITS = {"open": "no scrict schema or units"}

FILENOTFOUND_MSG = Template(
    "$fname not found. Check that the file is found in the dataset root "
    "directory e.g. mir-datasets/baf/$fname"
)


@core.docstring_inherit(annotations.EventData)
class EventDataExtended(annotations.EventData):
    """EventDataExtended class. Inherits from annotations.EventData class. An
    event is defined here as a match query-reference, and the time interval in
    the query. This class adds the possibility to attach tags to each event,
    useful if there's a need to differenciate them. In BAF, tags are [single,
    majority, unanimity].

    Attributes:
        tags (list): list of tag labels (as strings)
        tag_unit (str): tag units, one of TAG_UNITS
    """

    def __init__(self, intervals, interval_unit, events, event_unit, tags, tag_unit):
        super().__init__(intervals, interval_unit, events, event_unit)
        annotations.validate_array_like(intervals, np.ndarray, float)
        annotations.validate_array_like(events, list, str)
        annotations.validate_array_like(tags, list, str)
        annotations.validate_lengths_equal([intervals, events, tags])
        annotations.validate_intervals(intervals, interval_unit)
        annotations.validate_unit(event_unit, annotations.EVENT_UNITS)
        annotations.validate_unit(tag_unit, TAG_UNITS)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.events = events
        self.event_unit = event_unit
        self.tags = tags
        self.tag_unit = tag_unit


class Track(core.Track):
    """BAF track class.

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/baf`

    Attributes:
        audio_path (str): audio path

    Properties:
        audio (Tuple[np.ndarray, float]): audio array
        country (str): country of emission
        channel (str): tv channel of the emission
        datetime (str): datetime of the TV emission in YYYY-MM-DD HH:mm:ssformat
        matches (list): list of matches for a specific query

    Returns:
        Track: BAF dataset track
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
    def country(self) -> str:
        return self._track_metadata.get("country")

    @property
    def channel(self) -> str:
        return self._track_metadata.get("channel")

    @property
    def datetime(self) -> str:
        return self._track_metadata.get("datetime")

    @property
    def audio(self) -> Tuple[np.ndarray, float]:
        """The track's audio.

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    @property
    def matches(self) -> Optional[EventDataExtended]:
        return load_matches(self._track_metadata)

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
def load_audio(fpath: str) -> Tuple[np.ndarray, float]:
    """Load a baf audio file.

    Args:
        fpath (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fpath, sr=8000, mono=True)


def load_matches(track_metadata: dict) -> Optional[EventDataExtended]:
    """Load the matches corresponding to a query track.

    Args:
        track_metadata (dict): track's metadata

    Returns:
        Optional[EventDataExtended]: Track's annotations in EvendDataExtended format
    """
    # intervals_list = deque()  # linked list
    intervals_list = []
    events = []
    tags = []
    if track_metadata["annotations"] == []:
        return None
    else:
        for ann in track_metadata["annotations"]:
            intervals_list.append([round(ann["query_start"], 3), round(ann["query_end"], 3)])
            events.append(ann["reference"])
            tags.append(ann["tag"])
        intervals = np.array(
            intervals_list, dtype=float
        )  # more efficient than appending to np.array
        return EventDataExtended(
            intervals=intervals,
            interval_unit="s",
            events=events,
            event_unit="open",
            tags=tags,
            tag_unit="open",
        )


def csv_to_pandas(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as not_found:
        raise FileNotFoundError(FILENOTFOUND_MSG.safe_substitute(fname=not_found.filename))
    return df


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The BAF dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="baf",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        """Ingest dataset metadata."""
        metadata_path = os.path.join(self.data_home, "queries_info.csv")
        xannotations_path = os.path.join(self.data_home, "cross_annotations.csv")
        metadata_df = csv_to_pandas(metadata_path)
        xannotations_df = csv_to_pandas(xannotations_path)
        metadata_df.rename(columns={"filename": "query"}, inplace=True)
        df = pd.merge(metadata_df, xannotations_df, on="query", how="outer")
        df = df.replace(np.nan, "")
        metadata = dict()
        for _, row in df.iterrows():
            identifier = row.get("query").split(".wav")[0]
            md = metadata.get(identifier)
            reference = row.get("reference")
            if row.get("reference") == "":
                metadata[identifier] = {
                    "country": row.get("country"),
                    "channel": row.get("channel"),
                    "datetime": row.get("datetime"),
                    "annotations": [],
                }
            else:
                reference = reference.split(".wav")[0]
                if md is None:
                    metadata[identifier] = {
                        "country": row.get("country"),
                        "channel": row.get("channel"),
                        "datetime": row.get("datetime"),
                        "annotations": [
                            {
                                "reference": reference,
                                "query_start": round(row.get("query_start"), 3),
                                "query_end": round(row.get("query_end"), 3),
                                "tag": row.get("x_tag"),
                            }
                        ],
                    }
                else:
                    md["annotations"].append(
                        {
                            "reference": reference,
                            "query_start": round(row.get("query_start"), 3),
                            "query_end": round(row.get("query_end"), 3),
                            "tag": row.get("x_tag"),
                        }
                    )
        return metadata
