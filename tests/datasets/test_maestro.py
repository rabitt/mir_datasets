import os
import shutil

import numpy as np
import pretty_midi

from mirdata import annotations, download_utils
from mirdata.datasets import maestro
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1"
    data_home = "tests/resources/mir_datasets/maestro"
    dataset = maestro.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1",
        "midi_path": os.path.join(
            data_home,
            "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi",
        ),
        "audio_path": os.path.join(
            data_home, "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav"
        ),
        "canonical_composer": "Alban Berg",
        "canonical_title": "Sonata Op. 1",
        "year": 2018,
        "duration": 698.661160312,
        "split": "train",
    }

    expected_property_types = {
        "notes": annotations.NoteData,
        "midi": pretty_midi.PrettyMIDI,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": [
            "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav",
            "1694d8431f01eeb2a18444196550b99d",
        ],
        "midi": [
            "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi",
            "4901b1578ee4fe8c1696e02f60924949",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000 * 2,)


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/maestro"
    dataset = maestro.Dataset(data_home)
    metadata = dataset._metadata
    default_trackid = "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1"

    assert metadata[default_trackid] == {
        "canonical_composer": "Alban Berg",
        "canonical_title": "Sonata Op. 1",
        "split": "train",
        "year": 2018,
        "midi_filename": "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi",
        "audio_filename": "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav",
        "duration": 698.661160312,
    }


def test_download_partial(httpserver):
    data_home = "tests/resources/mir_datasets/maestro_download"
    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    httpserver.serve_content(open("tests/resources/download/maestro-v2.0.0.json", "r").read())
    remotes = {
        "all": download_utils.RemoteFileMetadata(
            filename="1-maestro-v2.0.0.json",
            url=httpserver.url,
            checksum=("d41d8cd98f00b204e9800998ecf8427e"),
            unpack_directories=["maestro-v2.0.0"],
        ),
        "midi": download_utils.RemoteFileMetadata(
            filename="2-maestro-v2.0.0.json",
            url=httpserver.url,
            checksum=("d41d8cd98f00b204e9800998ecf8427e"),
            unpack_directories=["maestro-v2.0.0"],
        ),
        "metadata": download_utils.RemoteFileMetadata(
            filename="3-maestro-v2.0.0.json",
            url=httpserver.url,
            checksum=("d41d8cd98f00b204e9800998ecf8427e"),
        ),
    }
    dataset = maestro.Dataset(data_home)
    dataset.remotes = remotes
    dataset.download(None, False, False)
    assert os.path.exists(os.path.join(data_home, "1-maestro-v2.0.0.json"))
    assert not os.path.exists(os.path.join(data_home, "2-maestro-v2.0.0.json"))
    assert not os.path.exists(os.path.join(data_home, "3-maestro-v2.0.0.json"))

    if os.path.exists(data_home):
        shutil.rmtree(data_home)
    dataset.download(["all", "midi"], False, False)
    assert os.path.exists(os.path.join(data_home, "1-maestro-v2.0.0.json"))
    assert not os.path.exists(os.path.join(data_home, "2-maestro-v2.0.0.json"))
    assert not os.path.exists(os.path.join(data_home, "3-maestro-v2.0.0.json"))

    if os.path.exists(data_home):
        shutil.rmtree(data_home)
    dataset.download(["metadata", "midi"], False, False)
    assert not os.path.exists(os.path.join(data_home, "1-maestro-v2.0.0.json"))
    assert os.path.exists(os.path.join(data_home, "2-maestro-v2.0.0.json"))
    assert not os.path.exists(os.path.join(data_home, "3-maestro-v2.0.0.json"))

    if os.path.exists(data_home):
        shutil.rmtree(data_home)
    dataset.download(["metadata"], False, False)
    assert not os.path.exists(os.path.join(data_home, "1-maestro-v2.0.0.json"))
    assert not os.path.exists(os.path.join(data_home, "2-maestro-v2.0.0.json"))
    assert os.path.exists(os.path.join(data_home, "3-maestro-v2.0.0.json"))


def test_download(httpserver):
    data_home = "tests/resources/mir_datasets/maestro_download"
    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # download the full dataset
    httpserver.serve_content(open("tests/resources/download/maestro-v2.0.0.zip", "rb").read())

    remotes = {
        "all": download_utils.RemoteFileMetadata(
            filename="maestro-v2.0.0.zip",
            url=httpserver.url,
            checksum=("625180ffa41cd9f2ab7252dd954b9e8a"),
            unpack_directories=["maestro-v2.0.0"],
        )
    }
    dataset = maestro.Dataset(data_home)
    dataset.remotes = remotes
    dataset.download(None, False, False)

    assert os.path.exists(data_home)
    assert not os.path.exists(os.path.join(data_home, "maestro-v2.0.0"))

    assert os.path.exists(os.path.join(data_home, "maestro-v2.0.0.json"))
    assert os.path.exists(
        os.path.join(
            data_home,
            "2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.wav",
        )
    )
    assert os.path.exists(
        os.path.join(
            data_home,
            "2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.midi",
        )
    )

    # test downloading again
    dataset.download(None, False, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # test downloading twice with cleanup
    dataset.download(None, False, True)
    dataset.download(None, False, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # test downloading twice with force overwrite
    dataset.download(None, False, False)
    dataset.download(None, True, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # test downloading twice with force overwrite and cleanup
    dataset.download(None, False, True)
    dataset.download(None, True, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # download the midi-only zip
    httpserver.serve_content(open("tests/resources/download/maestro-v2.0.0-midi.zip", "rb").read())

    remotes = {
        "midi": download_utils.RemoteFileMetadata(
            filename="maestro-v2.0.0-midi.zip",
            url=httpserver.url,
            checksum=("c82283fff347ed2bd833693c09a9f01d"),
            unpack_directories=["maestro-v2.0.0"],
        )
    }
    dataset.remotes = remotes
    dataset.download(["midi"], False, False)

    assert os.path.exists(data_home)
    assert not os.path.exists(os.path.join(data_home, "maestro-v2.0.0"))

    assert os.path.exists(os.path.join(data_home, "maestro-v2.0.0.json"))
    assert not os.path.exists(
        os.path.join(
            data_home,
            "2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.wav",
        )
    )
    assert os.path.exists(
        os.path.join(
            data_home,
            "2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.midi",
        )
    )

    # test downloading again
    dataset.download(["midi"], False, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # download only the metadata
    httpserver.serve_content(open("tests/resources/download/maestro-v2.0.0.json", "rb").read())

    remotes = {
        "metadata": download_utils.RemoteFileMetadata(
            filename="maestro-v2.0.0.json",
            url=httpserver.url,
            checksum=("d41d8cd98f00b204e9800998ecf8427e"),
        )
    }
    dataset.remotes = remotes
    dataset.download(["metadata"], False, False)

    assert os.path.exists(data_home)
    assert not os.path.exists(os.path.join(data_home, "maestro-v2.0.0"))

    assert os.path.exists(os.path.join(data_home, "maestro-v2.0.0.json"))
    assert not os.path.exists(
        os.path.join(
            data_home,
            "2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.wav",
        )
    )
    assert not os.path.exists(
        os.path.join(
            data_home,
            "2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.midi",
        )
    )

    # test downloading again
    dataset.download(["metadata"], False, False)

    if os.path.exists(data_home):
        shutil.rmtree(data_home)
