import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import compmusic_otmm_makam
from tests.test_utils import run_track_tests


def test_track():
    data_home = os.path.normpath("tests/resources/mir_datasets/compmusic_otmm_makam")
    track_id = "cafcdeaf-e966-4ff0-84fb-f660d2b68365"

    dataset = compmusic_otmm_makam.Dataset(data_home)
    track = dataset.track(track_id)

    expected_attributes = {
        "track_id": "cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        "pitch_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_otmm_makam/"),
            os.path.normpath(
                "MTG-otmm_makam_recognition_dataset-55ce75a/data/Kurdilihicazkar/cafcdeaf-e966-4ff0-84fb-f660d2b68365.pitch"
            ),
        ),
        "mb_tags_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_otmm_makam/"),
            os.path.normpath(
                "MTG-otmm_makam_recognition_dataset-55ce75a/data/Kurdilihicazkar/cafcdeaf-e966-4ff0-84fb-f660d2b68365.json"
            ),
        ),
        "form": "sarki",
        "instrumentation": "Solo vocal with accompaniment",
        "mb_url": "http://musicbrainz.org/work/cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        "title": "Türk Müziğinde 75 Büyük Bestekar/ 75 Great Composers In Turkish Classical Music",
        "artists": [
            {
                "mbid": "1aa10fb2-d9e1-489c-b66f-543e94cf0cbe",
                "attribute-list": ["violin"],
                "type": "instrument",
                "name": "Tuncay Düzagaç",
            },
            {
                "mbid": "1ed6c08e-9d5b-4e5c-a0a4-41acb3445000",
                "attribute-list": ["accordion"],
                "type": "instrument",
                "name": "Ceyhun Çelikten",
            },
            {
                "mbid": "30380155-5974-4cf5-aed9-60371470005d",
                "attribute-list": ["oud"],
                "type": "instrument",
                "name": "Yıldıran Güz",
            },
            {
                "mbid": "3411c847-4801-4a19-893c-521e5719cf44",
                "attribute-list": ["clarinet"],
                "type": "instrument",
                "name": "Şükrü Kabacı",
            },
            {
                "mbid": "4d3ff01d-71c1-4c1c-9a44-4e4448cfb9f8",
                "attribute-list": ["cello"],
                "type": "instrument",
                "name": "Özer Arkun",
            },
            {
                "mbid": "5121835d-84fb-44c6-8eed-8a8da4c444ba",
                "attribute-list": ["ney"],
                "type": "instrument",
                "name": "Ahmet Şahin",
            },
            {
                "mbid": "64944130-34f4-43bf-914f-e741a04e8350",
                "attribute-list": ["strings"],
                "type": "instrument",
                "name": "Kempa Yaylı Grubu",
            },
            {
                "mbid": "70c930f4-be11-48c6-97b8-b31547a31497",
                "attribute-list": ["viola"],
                "type": "instrument",
                "name": "İbrahim Şentürk",
            },
            {
                "mbid": "72a54b29-9c8d-4ec7-8c12-233bb9f4551a",
                "attribute-list": ["viola"],
                "type": "instrument",
                "name": "Metin Kabacı",
            },
            {
                "mbid": "8c968ab9-1a75-4ad2-b4a1-faa1f51d0006",
                "attribute-list": ["kanun"],
                "type": "instrument",
                "name": "Taner Sayacıoğlu",
            },
            {
                "mbid": "8c98c6c1-63a8-4573-9a76-31dc41fec8ad",
                "attribute-list": ["ney"],
                "type": "instrument",
                "name": "Eyüp Hamiş",
            },
            {
                "mbid": "9b6e98f8-b695-45da-8c9c-68d4bab460fd",
                "attribute-list": ["tanbur"],
                "type": "instrument",
                "name": "Murat Aydemir",
            },
            {
                "mbid": "b5e2f977-cadd-458a-8171-cafdfe6a331c",
                "attribute-list": ["percussion"],
                "type": "instrument",
                "name": "Fahrettin Yarkın",
            },
            {
                "mbid": "cab02cb9-d0b2-4ba5-8ea3-e5c6ab46b005",
                "attribute-list": ["violin"],
                "type": "instrument",
                "name": "Baki Kemancı",
            },
            {
                "mbid": "cdc69411-25f2-4fe3-88e5-41da2b8c474b",
                "attribute-list": ["violin"],
                "type": "instrument",
                "name": "Ayhan Şenyaylar",
            },
            {
                "mbid": "cfe6aea2-c362-4760-9ad1-b95db1605b6e",
                "attribute-list": ["cello"],
                "type": "instrument",
                "name": "Uğur Işık",
            },
            {
                "mbid": "d695dfaa-01f8-4a16-a538-5ccbad58de19",
                "attribute-list": ["oud"],
                "type": "instrument",
                "name": "Yurdal Tokcan",
            },
            {
                "mbid": "d858c208-8ab3-4588-b1d7-f54d24a72f20",
                "attribute-list": ["kanun"],
                "type": "instrument",
                "name": "Turgut Özüfler",
            },
            {
                "mbid": "df0ac610-a810-4e74-a767-57fe69495953",
                "attribute-list": ["classical kemençe"],
                "type": "instrument",
                "name": "Hasan Esen",
            },
            {
                "mbid": "ec6ed4c6-d7e4-4911-9234-30720f72abc8",
                "attribute-list": ["violin"],
                "type": "instrument",
                "name": "Timur Şenyaylar",
            },
            {
                "mbid": "f1b1b605-59d0-4ef7-b2a6-1f0256187caa",
                "attribute-list": ["violin"],
                "type": "instrument",
                "name": "Tarık Kemancı",
            },
            {
                "mbid": "fd8bceee-b3a4-4870-851a-0374f5542751",
                "type": "vocal",
                "name": "Bekir Ünlüataer",
            },
        ],
        "usul": "aksak",
        "work": "Aşka Merakım Ezelden",
        "makam": "Kurdilihicazkar",
        "mbid": "cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        "tonic": 260.0,
    }

    expected_property_types = {"pitch": annotations.F0Data, "mb_tags": dict}

    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/compmusic_otmm_makam"
    track_id = "cafcdeaf-e966-4ff0-84fb-f660d2b68365"

    dataset = compmusic_otmm_makam.Dataset(data_home)
    track = dataset.track(track_id)
    jam = track.to_jams()

    # Sandbox
    assert jam["sandbox"].tonic == 260.0
    assert jam["sandbox"].makam == "Kurdilihicazkar"
    assert jam["sandbox"].mbid == "cafcdeaf-e966-4ff0-84fb-f660d2b68365"

    # Pitch
    pitches = jam.search(namespace="pitch_contour")[0]["data"]
    assert len(pitches) == 9
    assert [pitch.time for pitch in pitches] == [
        0.0,
        0.0029024943310657597,
        0.005804988662131519,
        0.008707482993197279,
        0.011609977324263039,
        0.014512471655328799,
        0.017414965986394557,
        0.020317460317460317,
        0.023219954648526078,
    ]
    assert [pitch.duration for pitch in pitches] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert [pitch.value for pitch in pitches] == [
        {"index": 0, "frequency": 208.5, "voiced": True},
        {"index": 0, "frequency": 209.1, "voiced": True},
        {"index": 0, "frequency": 209.6, "voiced": True},
        {"index": 0, "frequency": 0.0, "voiced": False},
        {"index": 0, "frequency": 0.0, "voiced": False},
        {"index": 0, "frequency": 0.0, "voiced": False},
        {"index": 0, "frequency": 232.5, "voiced": True},
        {"index": 0, "frequency": 234.3, "voiced": True},
        {"index": 0, "frequency": 235.1, "voiced": True},
    ]
    assert [pitch.confidence for pitch in pitches] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    # Metadata
    metadata = jam["sandbox"].metadata
    assert metadata["usul"] == [
        {
            "attribute_key": "aksak",
            "mb_attribute": "Aksak",
            "source": "http://musicbrainz.org/work/753ff394-dec1-422b-991f-227d8f848532",
        },
        {
            "attribute_key": "aksak",
            "mb_tag": "aksak",
            "source": "http://musicbrainz.org/recording/cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        },
    ]
    assert metadata["makam"] == [
        {
            "attribute_key": "kurdilihicazkar",
            "mb_attribute": "K\u00fcrdilihicazkar",
            "source": "http://musicbrainz.org/work/753ff394-dec1-422b-991f-227d8f848532",
        },
        {
            "attribute_key": "kurdilihicazkar",
            "mb_tag": "k\u00fcrdilihicazkar",
            "source": "http://musicbrainz.org/recording/cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        },
    ]
    assert metadata["releases"] == [
        {
            "mbid": "aa4ec600-4b32-451c-9c51-226001dd51ef",
            "title": "T\u00fcrk M\u00fczi\u011finde 75 B\u00fcy\u00fck Bestekar/ 75 Great Composers In Turkish Classical Music",
        }
    ]
    assert metadata["title"] == "A\u015fka Merak\u0131m Ezelden"
    assert metadata["url"] == "http://musicbrainz.org/work/cafcdeaf-e966-4ff0-84fb-f660d2b68365"
    assert metadata["artist_credits"] == [
        {
            "mbid": "fd8bceee-b3a4-4870-851a-0374f5542751",
            "name": "Bekir \u00dcnl\u00fcataer",
        }
    ]
    assert metadata["sampling_frequency"] == 44100
    assert metadata["instrumentation_voicing"] == "Solo vocal with accompaniment"
    assert metadata["mbid"] == "cafcdeaf-e966-4ff0-84fb-f660d2b68365"
    assert metadata["form"] == [
        {
            "attribute_key": "sarki",
            "mb_attribute": "\u015eark\u0131",
            "source": "http://musicbrainz.org/work/753ff394-dec1-422b-991f-227d8f848532",
        },
        {
            "attribute_key": "sarki",
            "mb_tag": "\u015fark\u0131",
            "source": "http://musicbrainz.org/recording/cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        },
    ]
    assert metadata["bit_rate"] == 160
    assert metadata["artists"] == [
        {
            "mbid": "1aa10fb2-d9e1-489c-b66f-543e94cf0cbe",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Tuncay D\u00fczaga\u00e7",
        },
        {
            "mbid": "1ed6c08e-9d5b-4e5c-a0a4-41acb3445000",
            "attribute-list": ["accordion"],
            "type": "instrument",
            "name": "Ceyhun \u00c7elikten",
        },
        {
            "mbid": "30380155-5974-4cf5-aed9-60371470005d",
            "attribute-list": ["oud"],
            "type": "instrument",
            "name": "Y\u0131ld\u0131ran G\u00fcz",
        },
        {
            "mbid": "3411c847-4801-4a19-893c-521e5719cf44",
            "attribute-list": ["clarinet"],
            "type": "instrument",
            "name": "\u015e\u00fckr\u00fc Kabac\u0131",
        },
        {
            "mbid": "4d3ff01d-71c1-4c1c-9a44-4e4448cfb9f8",
            "attribute-list": ["cello"],
            "type": "instrument",
            "name": "\u00d6zer Arkun",
        },
        {
            "mbid": "5121835d-84fb-44c6-8eed-8a8da4c444ba",
            "attribute-list": ["ney"],
            "type": "instrument",
            "name": "Ahmet \u015eahin",
        },
        {
            "mbid": "64944130-34f4-43bf-914f-e741a04e8350",
            "attribute-list": ["strings"],
            "type": "instrument",
            "name": "Kempa Yayl\u0131 Grubu",
        },
        {
            "mbid": "70c930f4-be11-48c6-97b8-b31547a31497",
            "attribute-list": ["viola"],
            "type": "instrument",
            "name": "\u0130brahim \u015eent\u00fcrk",
        },
        {
            "mbid": "72a54b29-9c8d-4ec7-8c12-233bb9f4551a",
            "attribute-list": ["viola"],
            "type": "instrument",
            "name": "Metin Kabac\u0131",
        },
        {
            "mbid": "8c968ab9-1a75-4ad2-b4a1-faa1f51d0006",
            "attribute-list": ["kanun"],
            "type": "instrument",
            "name": "Taner Sayac\u0131o\u011flu",
        },
        {
            "mbid": "8c98c6c1-63a8-4573-9a76-31dc41fec8ad",
            "attribute-list": ["ney"],
            "type": "instrument",
            "name": "Ey\u00fcp Hami\u015f",
        },
        {
            "mbid": "9b6e98f8-b695-45da-8c9c-68d4bab460fd",
            "attribute-list": ["tanbur"],
            "type": "instrument",
            "name": "Murat Aydemir",
        },
        {
            "mbid": "b5e2f977-cadd-458a-8171-cafdfe6a331c",
            "attribute-list": ["percussion"],
            "type": "instrument",
            "name": "Fahrettin Yark\u0131n",
        },
        {
            "mbid": "cab02cb9-d0b2-4ba5-8ea3-e5c6ab46b005",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Baki Kemanc\u0131",
        },
        {
            "mbid": "cdc69411-25f2-4fe3-88e5-41da2b8c474b",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Ayhan \u015eenyaylar",
        },
        {
            "mbid": "cfe6aea2-c362-4760-9ad1-b95db1605b6e",
            "attribute-list": ["cello"],
            "type": "instrument",
            "name": "U\u011fur I\u015f\u0131k",
        },
        {
            "mbid": "d695dfaa-01f8-4a16-a538-5ccbad58de19",
            "attribute-list": ["oud"],
            "type": "instrument",
            "name": "Yurdal Tokcan",
        },
        {
            "mbid": "d858c208-8ab3-4588-b1d7-f54d24a72f20",
            "attribute-list": ["kanun"],
            "type": "instrument",
            "name": "Turgut \u00d6z\u00fcfler",
        },
        {
            "mbid": "df0ac610-a810-4e74-a767-57fe69495953",
            "attribute-list": ["classical kemen\u00e7e"],
            "type": "instrument",
            "name": "Hasan Esen",
        },
        {
            "mbid": "ec6ed4c6-d7e4-4911-9234-30720f72abc8",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Timur \u015eenyaylar",
        },
        {
            "mbid": "f1b1b605-59d0-4ef7-b2a6-1f0256187caa",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Tar\u0131k Kemanc\u0131",
        },
        {
            "mbid": "fd8bceee-b3a4-4870-851a-0374f5542751",
            "type": "vocal",
            "name": "Bekir \u00dcnl\u00fcataer",
        },
    ]
    assert metadata["duration"] == 275
    assert metadata["path"] == "../data/Kurdilihicazkar/cafcdeaf-e966-4ff0-84fb-f660d2b68365.mp3"
    assert metadata["works"] == [
        {
            "mbid": "753ff394-dec1-422b-991f-227d8f848532",
            "title": "A\u015fka Merak\u0131m Ezelden",
        }
    ]


def test_load_pitch():
    data_home = "tests/resources/mir_datasets/compmusic_otmm_makam"
    track_id = "cafcdeaf-e966-4ff0-84fb-f660d2b68365"

    dataset = compmusic_otmm_makam.Dataset(data_home)
    track = dataset.track(track_id)
    pitch_path = track.pitch_path
    parsed_pitch = compmusic_otmm_makam.load_pitch(pitch_path)

    # Check types
    assert type(parsed_pitch) == annotations.F0Data
    assert type(parsed_pitch.times) is np.ndarray
    assert type(parsed_pitch.frequencies) is np.ndarray
    assert type(parsed_pitch.voicing) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_pitch.times,
        np.array(
            [
                0.0,
                0.0029024943310657597,
                0.005804988662131519,
                0.008707482993197279,
                0.011609977324263039,
                0.014512471655328799,
                0.017414965986394557,
                0.020317460317460317,
                0.023219954648526078,
            ]
        ),
    )
    assert np.array_equal(
        parsed_pitch.frequencies,
        np.array([208.5, 209.1, 209.6, 0.0, 0.0, 0.0, 232.5, 234.3, 235.1]),
    )
    assert np.array_equal(
        parsed_pitch.voicing, np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    )

    assert compmusic_otmm_makam.load_pitch(None) is None


def test_load_metadata():
    data_home = os.path.normpath("tests/resources/mir_datasets/compmusic_otmm_makam")
    dataset = compmusic_otmm_makam.Dataset(data_home)
    metadata = dataset._metadata

    track_id = list(metadata.keys())[0]
    assert track_id == "cafcdeaf-e966-4ff0-84fb-f660d2b68365"

    assert metadata[track_id]["tonic"] == 260.0
    assert metadata[track_id]["makam"] == "Kurdilihicazkar"
    assert metadata[track_id]["mbid"] == "cafcdeaf-e966-4ff0-84fb-f660d2b68365"
    assert metadata["data_home"] == os.path.normpath(
        "tests/resources/mir_datasets/compmusic_otmm_makam"
    )


def test_load_mb_tags():
    data_home = "tests/resources/mir_datasets/compmusic_otmm_makam"
    track_id = "cafcdeaf-e966-4ff0-84fb-f660d2b68365"

    dataset = compmusic_otmm_makam.Dataset(data_home)
    track = dataset.track(track_id)
    mb_tags_path = track.mb_tags_path
    mb_tags = compmusic_otmm_makam.load_mb_tags(mb_tags_path)

    assert mb_tags["usul"] == [
        {
            "attribute_key": "aksak",
            "mb_attribute": "Aksak",
            "source": "http://musicbrainz.org/work/753ff394-dec1-422b-991f-227d8f848532",
        },
        {
            "attribute_key": "aksak",
            "mb_tag": "aksak",
            "source": "http://musicbrainz.org/recording/cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        },
    ]
    assert mb_tags["makam"] == [
        {
            "attribute_key": "kurdilihicazkar",
            "mb_attribute": "K\u00fcrdilihicazkar",
            "source": "http://musicbrainz.org/work/753ff394-dec1-422b-991f-227d8f848532",
        },
        {
            "attribute_key": "kurdilihicazkar",
            "mb_tag": "k\u00fcrdilihicazkar",
            "source": "http://musicbrainz.org/recording/cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        },
    ]
    assert mb_tags["releases"] == [
        {
            "mbid": "aa4ec600-4b32-451c-9c51-226001dd51ef",
            "title": "T\u00fcrk M\u00fczi\u011finde 75 B\u00fcy\u00fck Bestekar/ 75 Great Composers In Turkish Classical Music",
        }
    ]
    assert mb_tags["title"] == "A\u015fka Merak\u0131m Ezelden"
    assert mb_tags["url"] == "http://musicbrainz.org/work/cafcdeaf-e966-4ff0-84fb-f660d2b68365"
    assert mb_tags["artist_credits"] == [
        {
            "mbid": "fd8bceee-b3a4-4870-851a-0374f5542751",
            "name": "Bekir \u00dcnl\u00fcataer",
        }
    ]
    assert mb_tags["sampling_frequency"] == 44100
    assert mb_tags["instrumentation_voicing"] == "Solo vocal with accompaniment"
    assert mb_tags["mbid"] == "cafcdeaf-e966-4ff0-84fb-f660d2b68365"
    assert mb_tags["form"] == [
        {
            "attribute_key": "sarki",
            "mb_attribute": "\u015eark\u0131",
            "source": "http://musicbrainz.org/work/753ff394-dec1-422b-991f-227d8f848532",
        },
        {
            "attribute_key": "sarki",
            "mb_tag": "\u015fark\u0131",
            "source": "http://musicbrainz.org/recording/cafcdeaf-e966-4ff0-84fb-f660d2b68365",
        },
    ]
    assert mb_tags["bit_rate"] == 160
    assert mb_tags["artists"] == [
        {
            "mbid": "1aa10fb2-d9e1-489c-b66f-543e94cf0cbe",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Tuncay D\u00fczaga\u00e7",
        },
        {
            "mbid": "1ed6c08e-9d5b-4e5c-a0a4-41acb3445000",
            "attribute-list": ["accordion"],
            "type": "instrument",
            "name": "Ceyhun \u00c7elikten",
        },
        {
            "mbid": "30380155-5974-4cf5-aed9-60371470005d",
            "attribute-list": ["oud"],
            "type": "instrument",
            "name": "Y\u0131ld\u0131ran G\u00fcz",
        },
        {
            "mbid": "3411c847-4801-4a19-893c-521e5719cf44",
            "attribute-list": ["clarinet"],
            "type": "instrument",
            "name": "\u015e\u00fckr\u00fc Kabac\u0131",
        },
        {
            "mbid": "4d3ff01d-71c1-4c1c-9a44-4e4448cfb9f8",
            "attribute-list": ["cello"],
            "type": "instrument",
            "name": "\u00d6zer Arkun",
        },
        {
            "mbid": "5121835d-84fb-44c6-8eed-8a8da4c444ba",
            "attribute-list": ["ney"],
            "type": "instrument",
            "name": "Ahmet \u015eahin",
        },
        {
            "mbid": "64944130-34f4-43bf-914f-e741a04e8350",
            "attribute-list": ["strings"],
            "type": "instrument",
            "name": "Kempa Yayl\u0131 Grubu",
        },
        {
            "mbid": "70c930f4-be11-48c6-97b8-b31547a31497",
            "attribute-list": ["viola"],
            "type": "instrument",
            "name": "\u0130brahim \u015eent\u00fcrk",
        },
        {
            "mbid": "72a54b29-9c8d-4ec7-8c12-233bb9f4551a",
            "attribute-list": ["viola"],
            "type": "instrument",
            "name": "Metin Kabac\u0131",
        },
        {
            "mbid": "8c968ab9-1a75-4ad2-b4a1-faa1f51d0006",
            "attribute-list": ["kanun"],
            "type": "instrument",
            "name": "Taner Sayac\u0131o\u011flu",
        },
        {
            "mbid": "8c98c6c1-63a8-4573-9a76-31dc41fec8ad",
            "attribute-list": ["ney"],
            "type": "instrument",
            "name": "Ey\u00fcp Hami\u015f",
        },
        {
            "mbid": "9b6e98f8-b695-45da-8c9c-68d4bab460fd",
            "attribute-list": ["tanbur"],
            "type": "instrument",
            "name": "Murat Aydemir",
        },
        {
            "mbid": "b5e2f977-cadd-458a-8171-cafdfe6a331c",
            "attribute-list": ["percussion"],
            "type": "instrument",
            "name": "Fahrettin Yark\u0131n",
        },
        {
            "mbid": "cab02cb9-d0b2-4ba5-8ea3-e5c6ab46b005",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Baki Kemanc\u0131",
        },
        {
            "mbid": "cdc69411-25f2-4fe3-88e5-41da2b8c474b",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Ayhan \u015eenyaylar",
        },
        {
            "mbid": "cfe6aea2-c362-4760-9ad1-b95db1605b6e",
            "attribute-list": ["cello"],
            "type": "instrument",
            "name": "U\u011fur I\u015f\u0131k",
        },
        {
            "mbid": "d695dfaa-01f8-4a16-a538-5ccbad58de19",
            "attribute-list": ["oud"],
            "type": "instrument",
            "name": "Yurdal Tokcan",
        },
        {
            "mbid": "d858c208-8ab3-4588-b1d7-f54d24a72f20",
            "attribute-list": ["kanun"],
            "type": "instrument",
            "name": "Turgut \u00d6z\u00fcfler",
        },
        {
            "mbid": "df0ac610-a810-4e74-a767-57fe69495953",
            "attribute-list": ["classical kemen\u00e7e"],
            "type": "instrument",
            "name": "Hasan Esen",
        },
        {
            "mbid": "ec6ed4c6-d7e4-4911-9234-30720f72abc8",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Timur \u015eenyaylar",
        },
        {
            "mbid": "f1b1b605-59d0-4ef7-b2a6-1f0256187caa",
            "attribute-list": ["violin"],
            "type": "instrument",
            "name": "Tar\u0131k Kemanc\u0131",
        },
        {
            "mbid": "fd8bceee-b3a4-4870-851a-0374f5542751",
            "type": "vocal",
            "name": "Bekir \u00dcnl\u00fcataer",
        },
    ]
    assert mb_tags["duration"] == 275
    assert mb_tags["path"] == "../data/Kurdilihicazkar/cafcdeaf-e966-4ff0-84fb-f660d2b68365.mp3"
    assert mb_tags["works"] == [
        {
            "mbid": "753ff394-dec1-422b-991f-227d8f848532",
            "title": "A\u015fka Merak\u0131m Ezelden",
        }
    ]

    # test empty duration
    track_id = "2a3e1a6d-ab82-4a46-ae8b-0fb057b53de0"
    dataset = compmusic_otmm_makam.Dataset(data_home)
    track = dataset.track(track_id)
    mb_tags_path = track.mb_tags_path
    mb_tags = compmusic_otmm_makam.load_mb_tags(mb_tags_path)
    assert mb_tags["duration"] == 0.0


def test_special_turkish_characters():
    data_home = "tests/resources/mir_datasets/compmusic_otmm_makam"
    track_id = "cafcdeaf-e966-4ff0-84fb-f660d2b68365"

    dataset = compmusic_otmm_makam.Dataset(data_home)
    track = dataset.track(track_id)
    mb_tags_path = track.mb_tags_path
    special_characters = compmusic_otmm_makam.load_mb_tags(mb_tags_path)[
        "special_turkish_characters"
    ]

    assert special_characters == [
        "ç",
        "Ç",
        "ğ",
        "Ğ",
        "ı",
        "İ",
        "i",
        "İ",
        "ö",
        "Ö",
        "ş",
        "Ş",
        "ü",
        "Ü",
    ]
