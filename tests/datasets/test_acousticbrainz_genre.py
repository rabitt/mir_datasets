import os
import shutil

from mirdata import download_utils
from mirdata.datasets import acousticbrainz_genre
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "tagtraum#validation#be9e01e5-8f93-494d-bbaa-ddcc5a52f629#2b6bfcfd-46a5-3f98-a58f-2c51d7c9e960#trance########"
    data_home = os.path.normpath("tests/resources/mir_datasets/acousticbrainz_genre")

    dataset = acousticbrainz_genre.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "path": os.path.normpath(
            "tests/resources/mir_datasets/acousticbrainz_genre/acousticbrainz-mediaeval-validation/be/be9e01e5-8f93-494d-bbaa-ddcc5a52f629.json"
        ),
        "track_id": "tagtraum#validation#be9e01e5-8f93-494d-bbaa-ddcc5a52f629#2b6bfcfd-46a5-3f98-a58f-2c51d7c9e960#trance########",
        "genre": ["trance"],
        "mbid": "be9e01e5-8f93-494d-bbaa-ddcc5a52f629",
        "mbid_group": "2b6bfcfd-46a5-3f98-a58f-2c51d7c9e960",
        "split": "validation",
    }

    expected_property_types = {
        "artist": list,
        "title": list,
        "date": list,
        "file_name": str,
        "album": list,
        "tracknumber": list,
        "tonal": dict,
        "low_level": dict,
        "rhythm": dict,
        "acousticbrainz_metadata": dict,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_load_extractor():
    path = os.path.normpath(
        "tests/resources/mir_datasets/acousticbrainz_genre/acousticbrainz-mediaeval-validation/be/be9e01e5-8f93-494d-bbaa-ddcc5a52f629.json"
    )
    extractor_data = acousticbrainz_genre.load_extractor(path)

    assert isinstance(extractor_data, dict)


def test_to_jams():
    data_home = os.path.normpath("tests/resources/mir_datasets/acousticbrainz_genre")
    trackid = "tagtraum#validation#be9e01e5-8f93-494d-bbaa-ddcc5a52f629#2b6bfcfd-46a5-3f98-a58f-2c51d7c9e960#trance########"

    dataset = acousticbrainz_genre.Dataset(data_home, version="test")
    track = dataset.track(trackid)

    track.to_jams()


def test_filter_index():
    data_home = os.path.normpath("tests/resources/mir_datasets/acousticbrainz_genre")
    dataset = acousticbrainz_genre.Dataset(data_home, version="test")
    index = dataset.load_all_train()
    assert len(index) == 8
    index = dataset.load_all_validation()
    assert len(index) == 8
    index = dataset.load_tagtraum_validation()
    assert len(index) == 2
    index = dataset.load_tagtraum_train()
    assert len(index) == 2
    index = dataset.load_allmusic_validation()
    assert len(index) == 2
    index = dataset.load_lastfm_train()
    assert len(index) == 2
    index = dataset.load_lastfm_validation()
    assert len(index) == 2
    index = dataset.load_discogs_train()
    assert len(index) == 2
    index = dataset.load_discogs_validation()
    assert len(index) == 2


def test_download(httpserver):
    data_home = os.path.normpath("tests/resources/mir_datasets/acousticbrainz_genre_download")

    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    httpserver.serve_content(
        open(
            os.path.normpath("tests/resources/download/acousticbrainz_genre_index.json.zip"),
            "rb",
        ).read()
    )

    remotes = {
        "index": download_utils.RemoteFileMetadata(
            filename="acousticbrainz_genre_index.json.zip",
            url=httpserver.url,
            checksum="b32a663449c1da55de424d845521eb79",
        )
    }

    dataset = acousticbrainz_genre.Dataset(data_home, version="test")
    dataset.remotes = remotes
    dataset.download()

    assert os.path.exists(data_home)
    assert os.path.exists(os.path.join(data_home, "acousticbrainz_genre_index.json.zip"))

    httpserver.serve_content(
        open(
            os.path.normpath(
                "tests/resources/download/acousticbrainz-mediaeval-features-train-01.tar.bz2"
            ),
            "rb",
        ).read()
    )

    remotes = {
        "train-01": download_utils.RemoteFileMetadata(
            filename="acousticbrainz-mediaeval-features-train-01.tar.bz2",
            url=httpserver.url,
            checksum="eb155784e1d4de0f35aa23ded4d34849",
            destination_dir="acousticbrainz-mediaeval-train",
            unpack_directories=["acousticbrainz-mediaeval-train"],
        )
    }

    dataset.remotes = remotes
    dataset.download()

    assert os.path.exists(data_home)
    assert os.path.exists(os.path.join(data_home, "acousticbrainz-mediaeval-train"))
    assert os.path.exists(os.path.join(data_home, "acousticbrainz-mediaeval-train", "01"))
    assert os.path.exists(
        os.path.join(
            data_home,
            "acousticbrainz-mediaeval-train",
            "01",
            "01a0a332-d340-4806-a88b-cb60a05355c0.json",
        )
    )

    shutil.rmtree(data_home)
