import os

from mirdata.datasets import mridangam_stroke
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "224030"
    data_home = os.path.normpath("tests/resources/mir_datasets/mridangam_stroke")
    dataset = mridangam_stroke.Dataset(data_home)
    track = dataset.track(default_trackid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/mridangam_stroke/"),
            "mridangam_stroke_1.5/B/224030__akshaylaya__bheem-b-001.wav",
        ),
        "track_id": "224030",
        "stroke_name": "bheem",
        "tonic": "B",
    }

    run_track_tests(track, expected_attributes, {"audio": tuple})

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (35841,)


def test_to_jams():
    default_trackid = "224030"
    data_home = "tests/resources/mir_datasets/mridangam_stroke"
    dataset = mridangam_stroke.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate Mridangam schema
    assert jam.validate()

    # Test the stroke parser
    parsed_stroke = jam.annotations["tag_open"][0].data[0].value
    assert parsed_stroke == "bheem"
    assert (
        parsed_stroke in mridangam_stroke.STROKE_DICT
    ), "Stroke {} not in stroke dictionary".format(parsed_stroke)

    # Test the tonic parser
    parsed_tonic = jam.sandbox.tonic
    assert parsed_tonic == "B"
    assert parsed_tonic in mridangam_stroke.TONIC_DICT, "Stroke {} not in stroke dictionary".format(
        parsed_tonic
    )
