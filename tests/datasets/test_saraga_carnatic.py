import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import saraga_carnatic
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "116_Bhuvini_Dasudane"
    data_home = os.path.normpath("tests/resources/mir_datasets/saraga_carnatic")
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "116_Bhuvini_Dasudane",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.mp3.mp3",
        ),
        "audio_ghatam_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.multitrack-ghatam.mp3",
        ),
        "audio_mridangam_left_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.multitrack-mridangam-left.mp3",
        ),
        "audio_mridangam_right_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.multitrack-mridangam-right.mp3",
        ),
        "audio_violin_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.multitrack-violin.mp3",
        ),
        "audio_vocal_s_path": None,
        "audio_vocal_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.multitrack-vocal.mp3",
        ),
        "ctonic_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.ctonic.txt",
        ),
        "pitch_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.pitch.txt",
        ),
        "pitch_vocal_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.pitch-vocal.txt",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.tempo-manual.txt",
        ),
        "sama_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.sama-manual.txt",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.sections-manual-p.txt",
        ),
        "phrases_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.mphrases-manual.txt",
        ),
        "metadata_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_carnatic/"),
            "saraga1.5_carnatic/Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma/",
            "Bhuvini Dasudane/Bhuvini Dasudane.json",
        ),
    }

    expected_property_types = {
        "tempo": dict,
        "phrases": annotations.EventData,
        "pitch": annotations.F0Data,
        "pitch_vocal": annotations.F0Data,
        "sama": annotations.BeatData,
        "sections": annotations.SectionData,
        "tonic": float,
        "metadata": dict,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape[0] == 2


def test_to_jams():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    jam = track.to_jams()

    # Tonic
    assert jam["sandbox"].tonic == 201.740890

    # Pitch
    pitches = jam.search(namespace="pitch_contour")[0]["data"]
    assert len(pitches) == 6
    assert [pitch.time for pitch in pitches] == [
        0.0000000,
        0.0044444,
        0.0088889,
        0.0133333,
        0.0177778,
        0.0222222,
    ]
    assert [pitch.duration for pitch in pitches] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [pitch.value for pitch in pitches] == [
        {"index": 0, "frequency": 0.0000000, "voiced": False},
        {"index": 0, "frequency": 100.1200000, "voiced": True},
        {"index": 0, "frequency": 200.2300000, "voiced": True},
        {"index": 0, "frequency": 300.3400000, "voiced": True},
        {"index": 0, "frequency": 400.4300000, "voiced": True},
        {"index": 0, "frequency": 600.12300000, "voiced": True},
    ]
    assert [pitch.confidence for pitch in pitches] == [
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    pitches_vocal = jam.search(namespace="pitch_contour")[1]["data"]
    assert len(pitches_vocal) == 6
    assert [pitch_vocal.time for pitch_vocal in pitches_vocal] == [
        0.000000000000000000e00,
        2.902494331065759697e-03,
        5.804988662131519393e-03,
        8.707482993197278656e-03,
        1.160997732426303879e-02,
        1.451247165532879892e-02,
    ]
    assert [pitch_vocal.duration for pitch_vocal in pitches_vocal] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert [pitch_vocal.value for pitch_vocal in pitches_vocal] == [
        {"index": 0, "frequency": 0.000000000000000000e00, "voiced": False},
        {"index": 0, "frequency": 1.123456789012345678e02, "voiced": True},
        {"index": 0, "frequency": 2.234567890123456789e02, "voiced": True},
        {"index": 0, "frequency": 3.345678901234567890e02, "voiced": True},
        {"index": 0, "frequency": 4.456789012345678901e01, "voiced": True},
        {"index": 0, "frequency": 0.000000000000000000e00, "voiced": False},
    ]
    assert [pitch_vocal.confidence for pitch_vocal in pitches_vocal] == [
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    # Tempo
    parsed_tempo = jam["sandbox"].tempo
    assert parsed_tempo == {
        "tempo_apm": 330,
        "tempo_bpm": 82,
        "sama_interval": 5.827,
        "beats_per_cycle": 32,
        "subdivisions": 4,
    }

    # Sama
    samas = jam.search(namespace="beat")[0]["data"]
    assert len(samas) == 3
    assert [sama.time for sama in samas] == [4.894, 10.229, 15.724]
    assert [sama.duration for sama in samas] == [0.0, 0.0, 0.0]
    assert [sama.value for sama in samas] == [1, 2, 3]
    assert [sama.confidence for sama in samas] == [None, None, None]

    # Sections
    sections = jam.search(namespace="segment_open")[0]["data"]
    assert [section.time for section in sections] == [
        0.065306122,
        85.35510204,
        167.314285714,
    ]
    assert [section.duration for section in sections] == [
        85.289795918,
        81.95918367399999,
        142.02775510200001,
    ]
    assert [section.value for section in sections] == [
        "Pallavi",
        "Anupallavi",
        "Caraṇam",
    ]
    assert [section.confidence for section in sections] == [None, None, None]

    # Phrases
    phrases = jam.search(namespace="tag_open")[0]["data"]
    assert [phrase.time for phrase in phrases] == [0.224489795, 5.844897959, 8.50430839]
    assert [phrase.duration for phrase in phrases] == [
        2.4938775509999997,
        2.4734693870000006,
        2.2755555550000004,
    ]
    assert [phrase.value for phrase in phrases] == [
        "ndmdnsndn",
        "ndmdnsndn",
        "ndmdndmgr",
    ]
    assert [phrase.confidence for phrase in phrases] == [None, None, None]

    # Metadata
    metadata = jam["sandbox"].metadata
    assert metadata["raaga"] == [
        {
            "uuid": "42dd0ccb-f92a-4622-ae5d-a3be571b4939",
            "name": "Śrīranjani",
            "common_name": "shri ranjani",
        }
    ]
    assert metadata["form"] == [{"name": "Kriti"}]
    assert metadata["title"] == "Bhuvini Dasudane"
    assert metadata["work"] == [
        {"mbid": "4d05ce9b-c45e-4c85-9eca-941d68b61132", "title": "Bhuvini Dasudane"}
    ]
    assert metadata["length"] == 309000
    assert metadata["taala"] == [
        {
            "uuid": "c788c38a-b53a-48cb-b7bf-d11769260c4d",
            "name": "Ādi",
            "common_name": "adi",
        }
    ]
    assert metadata["album_artists"] == [
        {
            "mbid": "e09b0542-84e1-45ad-b09a-a05a9ad0cb83",
            "name": "Cherthala Ranganatha Sharma",
        }
    ]
    assert metadata["mbid"] == "9f5a5452-14cb-4af0-9289-4833854ee60d"
    assert metadata["artists"] == [
        {
            "instrument": {
                "mbid": "c5aa7d98-c14d-4ff1-8afb-f8743c62496c",
                "name": "Ghatam",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "19f93366-5d58-47f1-bc4f-9225ac7af6ba",
                "name": "N Guruprasad",
            },
        },
        {
            "instrument": {
                "mbid": "f689271c-37bc-4c49-92a3-a14b15ee5d0e",
                "name": "Mridangam",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "39c1d741-6154-418b-bf4b-12c77ba13873",
                "name": "Srimushnam V Raja Rao",
            },
        },
        {
            "instrument": {
                "mbid": "089f123c-0f7d-4105-a64e-49de81ca8fa4",
                "name": "Violin",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "a2df55e3-d141-4767-862e-77adca691d4b",
                "name": "B.U. Ganesh Prasad",
            },
        },
        {
            "instrument": {
                "mbid": "d92884b7-ee0c-46d5-96f3-918196ba8c5b",
                "name": "Voice",
            },
            "attributes": "lead vocals",
            "lead": True,
            "artist": {
                "mbid": "e09b0542-84e1-45ad-b09a-a05a9ad0cb83",
                "name": "Cherthala Ranganatha Sharma",
            },
        },
    ]
    assert metadata["concert"] == [
        {
            "mbid": "0816586d-c83e-4c79-a0aa-9b0e578f408d",
            "title": "Cherthala Ranganatha Sharma at Arkay",
        }
    ]


def test_load_tonic():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    tonic_path = track.ctonic_path
    parsed_tonic = saraga_carnatic.load_tonic(tonic_path)
    assert parsed_tonic == 201.740890
    assert saraga_carnatic.load_tonic(None) is None


def test_load_pitch():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    pitch_path = track.pitch_path
    parsed_pitch = saraga_carnatic.load_pitch(pitch_path)

    # Check types
    assert type(parsed_pitch) == annotations.F0Data
    assert type(parsed_pitch.times) is np.ndarray
    assert type(parsed_pitch.frequencies) is np.ndarray
    assert type(parsed_pitch.voicing) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_pitch.times,
        np.array([0.0000000, 0.0044444, 0.0088889, 0.0133333, 0.0177778, 0.0222222]),
    )
    assert np.array_equal(
        parsed_pitch.frequencies,
        np.array(
            [
                0.0000000,
                100.1200000,
                200.2300000,
                300.3400000,
                400.4300000,
                600.12300000,
            ]
        ),
    )
    assert np.array_equal(parsed_pitch.voicing, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    pitch_vocal_path = track.pitch_vocal_path
    parsed_vocal_pitch = saraga_carnatic.load_pitch(pitch_vocal_path)

    # Check types
    assert type(parsed_vocal_pitch) == annotations.F0Data
    assert type(parsed_vocal_pitch.times) is np.ndarray
    assert type(parsed_vocal_pitch.frequencies) is np.ndarray
    assert type(parsed_vocal_pitch.voicing) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_vocal_pitch.times,
        np.array(
            [
                0.000000000000000000e00,
                2.902494331065759697e-03,
                5.804988662131519393e-03,
                8.707482993197278656e-03,
                1.160997732426303879e-02,
                1.451247165532879892e-02,
            ]
        ),
    )
    assert np.array_equal(
        parsed_vocal_pitch.frequencies,
        np.array(
            [
                0.000000000000000000e00,
                1.123456789012345678e02,
                2.234567890123456789e02,
                3.345678901234567890e02,
                4.456789012345678901e01,
                0.000000000000000000e00,
            ]
        ),
    )
    assert np.array_equal(parsed_vocal_pitch.voicing, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]))

    assert saraga_carnatic.load_pitch(None) is None


def test_load_sama():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    sama_path = track.sama_path
    parsed_sama = saraga_carnatic.load_sama(sama_path)

    # Check types
    assert type(parsed_sama) == annotations.BeatData
    assert type(parsed_sama.times) is np.ndarray
    assert type(parsed_sama.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_sama.times, np.array([4.894, 10.229, 15.724]))
    assert np.array_equal(parsed_sama.positions, np.array([1, 2, 3]))
    assert saraga_carnatic.load_sama(None) is None

    track = dataset.track("117_Karuna_Nidhi_Illalo")
    parsed_sama = track.sama
    assert parsed_sama is None


def test_load_sections():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    sections_path = track.sections_path
    parsed_sections = saraga_carnatic.load_sections(sections_path)

    # Check types
    assert type(parsed_sections) == annotations.SectionData
    assert type(parsed_sections.intervals) is np.ndarray
    assert type(parsed_sections.labels) is list

    # Check values
    assert np.array_equal(
        parsed_sections.intervals[:, 0],
        np.array([0.065306122, 85.355102040, 167.314285714]),
    )
    assert np.array_equal(
        parsed_sections.intervals[:, 1],
        np.array([85.35510203999999, 167.314285714, 309.342040816]),
    )
    assert parsed_sections.labels == ["Pallavi", "Anupallavi", "Caraṇam"]

    assert saraga_carnatic.load_sections(None) is None


def test_load_phrases():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    phrases_path = track.phrases_path
    parsed_phrases = saraga_carnatic.load_phrases(phrases_path)

    # Check types
    assert type(parsed_phrases) is annotations.EventData
    assert type(parsed_phrases.intervals) is np.ndarray
    assert type(parsed_phrases.events) is list

    # Check values
    assert np.array_equal(
        parsed_phrases.intervals,
        np.array(
            [
                [0.224489795, 2.718367346],
                [5.844897959, 8.318367346],
                [8.50430839, 10.779863945],
            ]
        ),
    )
    assert parsed_phrases.events == ["ndmdnsndn", "ndmdnsndn", "ndmdndmgr"]
    assert saraga_carnatic.load_phrases(None) is None


def test_load_tempo():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    tempo_path = track.tempo_path
    parsed_tempo = saraga_carnatic.load_tempo(tempo_path)

    assert type(parsed_tempo) == dict
    assert type(parsed_tempo["tempo_apm"]) == int
    assert type(parsed_tempo["sama_interval"]) == float
    assert parsed_tempo == {
        "tempo_apm": 330,
        "tempo_bpm": 82,
        "sama_interval": 5.827,
        "beats_per_cycle": 32,
        "subdivisions": 4,
    }

    assert saraga_carnatic.load_tempo(None) is None

    track = dataset.track("115_Idhu_Thaano_Thillai_Sthalam")
    tempo_path = track.tempo_path
    parsed_tempo = saraga_carnatic.load_tempo(tempo_path)
    assert parsed_tempo is None


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    metadata_path = track.metadata_path
    parsed_metadata = saraga_carnatic.load_metadata(metadata_path)

    assert parsed_metadata["raaga"] == [
        {
            "uuid": "42dd0ccb-f92a-4622-ae5d-a3be571b4939",
            "name": "Śrīranjani",
            "common_name": "shri ranjani",
        }
    ]
    assert parsed_metadata["form"] == [{"name": "Kriti"}]
    assert parsed_metadata["title"] == "Bhuvini Dasudane"
    assert parsed_metadata["work"] == [
        {"mbid": "4d05ce9b-c45e-4c85-9eca-941d68b61132", "title": "Bhuvini Dasudane"}
    ]
    assert parsed_metadata["length"] == 309000
    assert parsed_metadata["taala"] == [
        {
            "uuid": "c788c38a-b53a-48cb-b7bf-d11769260c4d",
            "name": "Ādi",
            "common_name": "adi",
        }
    ]
    assert parsed_metadata["album_artists"] == [
        {
            "mbid": "e09b0542-84e1-45ad-b09a-a05a9ad0cb83",
            "name": "Cherthala Ranganatha Sharma",
        }
    ]
    assert parsed_metadata["mbid"] == "9f5a5452-14cb-4af0-9289-4833854ee60d"
    assert parsed_metadata["artists"] == [
        {
            "instrument": {
                "mbid": "c5aa7d98-c14d-4ff1-8afb-f8743c62496c",
                "name": "Ghatam",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "19f93366-5d58-47f1-bc4f-9225ac7af6ba",
                "name": "N Guruprasad",
            },
        },
        {
            "instrument": {
                "mbid": "f689271c-37bc-4c49-92a3-a14b15ee5d0e",
                "name": "Mridangam",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "39c1d741-6154-418b-bf4b-12c77ba13873",
                "name": "Srimushnam V Raja Rao",
            },
        },
        {
            "instrument": {
                "mbid": "089f123c-0f7d-4105-a64e-49de81ca8fa4",
                "name": "Violin",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "a2df55e3-d141-4767-862e-77adca691d4b",
                "name": "B.U. Ganesh Prasad",
            },
        },
        {
            "instrument": {
                "mbid": "d92884b7-ee0c-46d5-96f3-918196ba8c5b",
                "name": "Voice",
            },
            "attributes": "lead vocals",
            "lead": True,
            "artist": {
                "mbid": "e09b0542-84e1-45ad-b09a-a05a9ad0cb83",
                "name": "Cherthala Ranganatha Sharma",
            },
        },
    ]
    assert parsed_metadata["concert"] == [
        {
            "mbid": "0816586d-c83e-4c79-a0aa-9b0e578f408d",
            "title": "Cherthala Ranganatha Sharma at Arkay",
        }
    ]


def test_load_audio():
    data_home = "tests/resources/mir_datasets/saraga_carnatic"
    dataset = saraga_carnatic.Dataset(data_home)
    track = dataset.track("116_Bhuvini_Dasudane")
    audio_path = track.audio_path
    audio, sr = saraga_carnatic.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray
    assert audio.shape[0] == 2

    assert saraga_carnatic.load_audio(None) is None
