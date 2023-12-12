import os
import shutil

import pretty_midi

from mirdata import annotations, download_utils
from mirdata.datasets import groove_midi
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "drummer1/eval_session/1"
    data_home = "tests/resources/mir_datasets/groove_midi"
    dataset = groove_midi.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "drummer": "drummer1",
        "session": "drummer1/eval_session",
        "track_id": "drummer1/eval_session/1",
        "style": "funk/groove1",
        "tempo": 138,
        "beat_type": "beat",
        "time_signature": "4-4",
        "midi_filename": "drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid",
        "audio_filename": "drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav",
        "midi_path": os.path.join(
            data_home, "drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid"
        ),
        "audio_path": os.path.join(
            data_home, "drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav"
        ),
        "duration": 27.872308,
        "split": "test",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "drum_events": annotations.EventData,
        "midi": pretty_midi.PrettyMIDI,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": [
            "drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav",
            "7f94a191506f70ac9d313b7978203c3c",
        ],
        "midi": [
            "drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid",
            "b01a609cee84cfbc2c154bb9b6566955",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 22050
    assert audio.shape == (613566,)

    # test midi loading functions
    midi_data = track.midi
    assert len(midi_data.instruments) == 1
    assert len(midi_data.instruments[0].notes) == 410
    assert midi_data.estimate_tempo() == 198.7695135305443
    assert midi_data.get_piano_roll().shape == (128, 2787)


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/groove_midi"
    dataset = groove_midi.Dataset(data_home)
    metadata = dataset._metadata

    assert metadata["drummer1/eval_session/1"] == {
        "drummer": "drummer1",
        "session": "drummer1/eval_session",
        "track_id": "drummer1/eval_session/1",
        "style": "funk/groove1",
        "tempo": 138,
        "beat_type": "beat",
        "time_signature": "4-4",
        "midi_filename": "drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid",
        "audio_filename": "drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav",
        "duration": 27.872308,
        "split": "test",
    }


def test_load_audio():
    audio_none, sr_none = groove_midi.load_audio(None)
    assert audio_none is None
    assert sr_none is None


def test_download(httpserver):
    data_home = "tests/resources/mir_datasets/groove_midi_download"
    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    httpserver.serve_content(open("tests/resources/download/groove-v1-0.0.zip", "rb").read())

    remotes = {
        "all": download_utils.RemoteFileMetadata(
            filename="groove-v1-0.0.zip",
            url=httpserver.url,
            checksum=("97a9a888d2a65cc87bb26e74df08b011"),
            unpack_directories=["groove"],
        )
    }
    dataset = groove_midi.Dataset(data_home)
    dataset.remotes = remotes
    dataset.download(None, False, False)

    assert os.path.exists(data_home)
    assert not os.path.exists(os.path.join(data_home, "groove"))

    assert os.path.exists(os.path.join(data_home, "info.csv"))
    track = dataset.track("drummer1/eval_session/1")
    assert os.path.exists(track.midi_path)
    assert os.path.exists(track.audio_path)

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
