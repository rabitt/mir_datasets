import os

import numpy as np
import pytest

import mirdata
from mirdata import core
from tests.test_utils import DEFAULT_DATA_HOME


def test_track():
    index = {
        "tracks": {
            "a": {
                "audio": (None, None),
                "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
            }
        }
    }
    track_id = "a"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    track = core.Track(track_id, data_home, dataset_name, index, lambda: None)

    assert track.track_id == track_id
    assert track._dataset_name == dataset_name
    assert track._data_home == data_home
    assert track._track_paths == {
        "audio": (None, None),
        "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
    }
    assert track._metadata() is None
    with pytest.raises(AttributeError):
        track._track_metadata

    with pytest.raises(NotImplementedError):
        track.to_jams()

    path_good = track.get_path("annotation")
    assert os.path.normpath(path_good) == os.path.normpath("tests/resources/mir_datasets/asdf/asdd")
    path_none = track.get_path("audio")
    assert path_none is None

    # tracks with metadata
    metadata_track_index = lambda: {"a": {"x": 1, "y": 2, "z": 3}}
    metadata_global = lambda: {"asdf": [1, 2, 3], "asdd": [4, 5, 6]}
    metadata_none = lambda: None

    track_metadata_tidx = core.Track(track_id, data_home, dataset_name, index, metadata_track_index)
    assert track_metadata_tidx._track_metadata == {"x": 1, "y": 2, "z": 3}

    track_metadata_global = core.Track(track_id, data_home, dataset_name, index, metadata_global)
    assert track_metadata_global._track_metadata == {
        "asdf": [1, 2, 3],
        "asdd": [4, 5, 6],
    }

    track_metadata_none = core.Track(track_id, data_home, dataset_name, index, metadata_none)
    with pytest.raises(AttributeError):
        track_metadata_none._track_metadata


def test_track_repr():
    class TestTrack(core.Track):
        def __init__(self):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50

        @property
        def f(self):
            """ThisObjectType: I have a docstring"""
            return None

        @property
        def g(self):
            """I have an improper docstring."""
            return None

        def h(self):
            return "I'm a function!"

    expected1 = """Track(\n  a="asdf",\n  b=1.2345678,\n  """
    expected2 = """c={1: 'a', 'b': 2},\n  e=None,\n  """
    expected3 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected4 = """f: ThisObjectType,\n  g: I have an improper docstring,\n)"""

    test_track = TestTrack()
    actual = test_track.__repr__()
    assert actual == expected1 + expected2 + expected3 + expected4

    with pytest.raises(NotImplementedError):
        test_track.to_jams()


def test_multitrack_repr():
    class TestTrack(core.Track):
        def __init__(self):
            self.a = "asdf"

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

        @property
        def f(self):
            """ThisObjectType: I have a docstring"""
            return None

        @property
        def g(self):
            """I have an improper docstring."""
            return None

        def h(self):
            return "I'm a function!"

    expected1 = """Track(\n  a="asdf",\n  b=1.2345678,\n  """
    expected2 = """c={1: \'a\', \'b\': 2},\n  e=None,\n  """
    expected3 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected4 = """mtrack_id="test",\n  track_ids=[\'a\', \'b\', \'c\'],\n  """
    expected5 = """f: ThisObjectType,\n  g: I have an improper docstring,\n  """
    expected6 = """track_audio_property: ,\n  tracks: ,\n)"""

    test_mtrack = TestMultiTrack("test", "foo")
    actual = test_mtrack.__repr__()
    assert actual == expected1 + expected2 + expected3 + expected4 + expected5 + expected6

    with pytest.raises(NotImplementedError):
        test_mtrack.to_jams()


def test_dataset():
    dataset = mirdata.initialize("acousticbrainz_genre")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("guitarset")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("rwc_jazz")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("ikala")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("phenicx_anechoic")
    assert isinstance(dataset, core.Dataset)

    print(dataset)  # test that repr doesn't fail


def test_dataset_versions():
    class VersionTest(core.Dataset):
        def __init__(self, data_home=None, version="default"):
            super().__init__(
                data_home,
                version,
                indexes={
                    "default": "1",
                    "test": "0",
                    "0": core.Index("blah_0.json"),
                    "1": core.Index("blah_1.json", url="https://google.com", checksum="asdf"),
                    "2": core.Index("blah_2.json"),
                    "real": core.Index("beatles_index_1.2.json"),
                },
            )

    dataset = VersionTest("asdf")
    assert dataset.version == "1"
    assert os.path.normpath(dataset.index_path) == os.path.normpath(
        "asdf/mirdata_indexes/blah_1.json"
    )

    dataset_default = VersionTest("asdf", version="default")
    assert dataset_default.version == "1"
    assert os.path.normpath(dataset_default.index_path) == os.path.normpath(
        "asdf/mirdata_indexes/blah_1.json"
    )

    dataset_1 = VersionTest("asdf", version="1")
    assert dataset_1.version == "1"
    assert os.path.normpath(dataset_1.index_path) == os.path.normpath(
        "asdf/mirdata_indexes/blah_1.json"
    )
    with pytest.raises(FileNotFoundError):
        dataset_1._index

    local_index_path = os.path.dirname(os.path.realpath(__file__))[:-5]
    dataset_test = VersionTest("asdf", version="test")
    assert dataset_test.version == "0"
    assert os.path.normpath(dataset_test.index_path) == os.path.join(
        local_index_path, os.path.normpath("mirdata/datasets/indexes/blah_0.json")
    )

    with pytest.raises(IOError):
        dataset_test._index

    dataset_0 = VersionTest("asdf", version="0")
    assert dataset_0.version == "0"
    assert os.path.normpath(dataset_0.index_path) == os.path.join(
        local_index_path, os.path.normpath("mirdata/datasets/indexes/blah_0.json")
    )

    dataset_2 = VersionTest("asdf", version="2")
    assert dataset_2.version == "2"
    assert os.path.normpath(dataset_2.index_path) == os.path.join(
        local_index_path, os.path.normpath("mirdata/datasets/indexes/blah_2.json")
    )

    dataset_real = VersionTest("asdf", version="real")
    assert dataset_real.version == "real"
    assert os.path.normpath(dataset_real.index_path) == os.path.join(
        local_index_path,
        os.path.normpath("mirdata/datasets/indexes/beatles_index_1.2.json"),
    )
    idx_test = dataset_real._index
    assert isinstance(idx_test, dict)

    with pytest.raises(ValueError):
        VersionTest("asdf", version="not_a_version")


def test_dataset_errors():
    with pytest.raises(ValueError):
        mirdata.initialize("not_a_dataset")

    d = mirdata.initialize("orchset")
    d._track_class = None
    with pytest.raises(AttributeError):
        d.track("asdf")

    with pytest.raises(AttributeError):
        d.multitrack("asdf")

    with pytest.raises(AttributeError):
        d.load_tracks()

    with pytest.raises(AttributeError):
        d.load_multitracks()

    with pytest.raises(AttributeError):
        d.choice_track()

    with pytest.raises(AttributeError):
        d.choice_multitrack()

    d = mirdata.initialize("acousticbrainz_genre")
    with pytest.raises(FileNotFoundError):
        d._index

    d = mirdata.initialize("phenicx_anechoic")
    with pytest.raises(ValueError):
        d._multitrack("a")


def test_multitrack():
    index_tracks = {
        "tracks": {
            "a": {
                "audio": (None, None),
                "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
            },
            "b": {
                "audio": (None, None),
                "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
            },
        }
    }
    index_mtracks = {
        "multitracks": {
            "ab": {
                "tracks": ["a", "b"],
                "audio_master": ("foo/bar", "asdfasdfasdfasdf"),
                "score": (None, None),
            }
        }
    }
    index = {}
    index.update(index_tracks)
    index.update(index_mtracks)
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = core.MultiTrack(mtrack_id, data_home, dataset_name, index, core.Track, lambda: None)

    path_good = mtrack.get_path("audio_master")
    assert os.path.normpath(path_good) == os.path.normpath("tests/resources/mir_datasets/foo/bar")
    path_none = mtrack.get_path("score")
    assert path_none is None

    assert mtrack.mtrack_id == mtrack_id
    assert mtrack._dataset_name == dataset_name
    assert mtrack._data_home == data_home
    assert list(mtrack.tracks.keys()) == ["a", "b"]

    assert mtrack._metadata() is None
    with pytest.raises(AttributeError):
        mtrack._multitrack_metadata

    with pytest.raises(NotImplementedError):
        mtrack.to_jams()

    with pytest.raises(KeyError):
        mtrack.get_target(["c"])

    with pytest.raises(NotImplementedError):
        mtrack.get_random_target()

    with pytest.raises(NotImplementedError):
        mtrack.get_mix()

    with pytest.raises(NotImplementedError):
        mtrack.track_audio_property

    # tracks with metadata
    metadata_mtrack_index = lambda: {"ab": {"x": 1, "y": 2, "z": 3}}
    metadata_global = lambda: {"asdf": [1, 2, 3], "asdd": [4, 5, 6]}
    metadata_none = lambda: None

    mtrack_metadata_tidx = core.MultiTrack(
        mtrack_id, data_home, dataset_name, index, core.Track, metadata_mtrack_index
    )
    assert mtrack_metadata_tidx._multitrack_metadata == {"x": 1, "y": 2, "z": 3}

    mtrack_metadata_global = core.MultiTrack(
        mtrack_id, data_home, dataset_name, index, core.Track, metadata_global
    )
    assert mtrack_metadata_global._multitrack_metadata == {
        "asdf": [1, 2, 3],
        "asdd": [4, 5, 6],
    }

    mtrack_metadata_none = core.MultiTrack(
        mtrack_id, data_home, dataset_name, index, core.Track, metadata_none
    )
    with pytest.raises(AttributeError):
        mtrack_metadata_none._multitrack_metadata

    class TestTrack(core.Track):
        def __init__(self, key, data_home="foo", dataset_name="foo", index=None, metadata=None):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
            super().__init__(mtrack_id, data_home, dataset_name, index, track_class, metadata)

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    # import pdb;pdb.set_trace()
    mtrack = TestMultiTrack1(mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None)
    mtrack.to_jams()
    mtrack.get_target(["a"])
    mtrack.get_random_target()


def test_multitrack_mixing():
    class TestTrack(core.Track):
        def __init__(self, key, data_home="foo", dataset_name="foo", index=None, metadata=None):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
            super().__init__(mtrack_id, data_home, dataset_name, index, track_class, metadata)

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None)

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (2, 100)
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["b", "c"], weights=[0.5, 0.2])
    assert target2.shape == (2, 100)
    assert np.max(np.abs(target2)) <= 1

    target3 = mtrack.get_target(["b", "c"], weights=[0.5, 5])
    assert target3.shape == (2, 100)
    assert np.max(np.abs(target3)) <= 1

    target4 = mtrack.get_target(["a", "c"], average=False)
    assert target4.shape == (2, 100)
    assert np.max(np.abs(target4)) <= 2

    target5 = mtrack.get_target(["a", "c"], average=False, weights=[0.1, 0.5])
    assert target5.shape == (2, 100)
    assert np.max(np.abs(target5)) <= 0.6

    random_target1, t1, w1 = mtrack.get_random_target(n_tracks=2)
    assert random_target1.shape == (2, 100)
    assert np.max(np.abs(random_target1)) <= 1
    assert len(t1) == 2
    assert len(w1) == 2
    assert np.all(w1 >= 0.3)
    assert np.all(w1 <= 1.0)

    random_target2, t2, w2 = mtrack.get_random_target(n_tracks=5)
    assert random_target2.shape == (2, 100)
    assert np.max(np.abs(random_target2)) <= 1
    assert len(t2) == 3
    assert len(w2) == 3
    assert np.all(w2 >= 0.3)
    assert np.all(w2 <= 1.0)

    random_target3, t3, w3 = mtrack.get_random_target()
    assert random_target3.shape == (2, 100)
    assert np.max(np.abs(random_target3)) <= 1
    assert len(t3) == 3
    assert len(w3) == 3
    assert np.all(w3 >= 0.3)
    assert np.all(w3 <= 1.0)

    random_target4, t4, w4 = mtrack.get_random_target(n_tracks=2, min_weight=0.1, max_weight=0.4)
    assert random_target4.shape == (2, 100)
    assert np.max(np.abs(random_target4)) <= 1
    assert len(t4) == 2
    assert len(w4) == 2
    assert np.all(w4 >= 0.1)
    assert np.all(w4 <= 0.4)

    mix = mtrack.get_mix()
    assert mix.shape == (2, 100)


def test_multitrack_unequal_len():
    class TestTrack(core.Track):
        def __init__(self, key, data_home="foo", dataset_name="foo", index=None, metadata=None):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, np.random.randint(50, 100))), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
            super().__init__(mtrack_id, data_home, dataset_name, index, track_class, metadata)

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None)

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])

    with pytest.raises(KeyError):
        mtrack.get_target(["d", "e"])

    target1 = mtrack.get_target(["a", "b", "c"], enforce_length=False)
    assert target1.shape[0] == 2
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["a", "b", "c"], average=False, enforce_length=False)
    assert target2.shape[0] == 2
    assert np.max(np.abs(target2)) <= 3


def test_multitrack_unequal_sr():
    class TestTrack(core.Track):
        def __init__(self, key, data_home="foo", dataset_name="foo", index=None, metadata=None):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), np.random.randint(10, 1000)

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
            super().__init__(mtrack_id, data_home, dataset_name, index, track_class, metadata)

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(
        mtrack_id,
        data_home,
        dataset_name,
        index,
        TestTrack,
        lambda: track_metadata_none,
    )

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])


def test_multitrack_mono():
    ### no first channel - audio shapes (100,)
    class TestTrack(core.Track):
        def __init__(self, key, data_home="foo", dataset_name="foo", index=None, metadata=None):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
            super().__init__(mtrack_id, data_home, dataset_name, index, track_class, metadata)

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None)

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2

    ### one channel mono shape (1, 100)
    class TestTrack1(core.Track):
        def __init__(self, key, data_home="foo", dataset_name="foo", index=None, metadata=None):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (1, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
            super().__init__(mtrack_id, data_home, dataset_name, index, track_class, metadata)

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None)

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2


def test_dataset_splits():
    empty_dataset = core.Dataset(name="test", indexes={"default": core.Index("asdf.json")})

    # test the case where there are no tracks
    with pytest.raises(AttributeError):
        empty_dataset.get_random_track_splits([0.9, 0.1])

    # test the case where there are no multitracks
    with pytest.raises(AttributeError):
        empty_dataset.get_random_mtrack_splits([0.9, 0.1])

    with pytest.raises(AttributeError):
        empty_dataset.get_track_splits()

    with pytest.raises(AttributeError):
        empty_dataset.get_mtrack_splits()

    # test the partition function
    items = [i for i in range(100)]
    list_sum_up_1 = [
        [0.7, 0.1, 0.1, 0.1],
        [0.4, 0.2, 0.2, 0.2],
        [0.5, 0.2, 0.2, 0.1],
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.9, 0.1, 0.0],
        [0.6, 0.3, 0.1],
        [0.5, 0.4, 0.1],
        [0.9, 0.05, 0.05],
        [0.8, 0.2],
        [0.1, 0.9],
    ]
    for right_combination in list_sum_up_1:
        splits = empty_dataset._get_partitions(items, right_combination, 42)
        # check that the right number of splits are created
        assert len(splits) == len(right_combination)
        # check that the number of total items matches
        assert len(items) == sum([len(i) for i in splits.values()])
        # check that all items are used
        assert set(items) == set([i for split_items in splits.values() for i in split_items])
        # check that splits are nonoverlapping
        used = set()
        for split in splits.values():
            this_split = set(split)
            assert not this_split.intersection(used)
            used.update(this_split)
        # check that the split is reproducable
        splits2 = empty_dataset._get_partitions(items, right_combination, 42)
        for split, split2 in zip(splits.values(), splits2.values()):
            assert np.array_equal(split, split2)

    # test partition names
    with pytest.raises(ValueError):
        splits = empty_dataset._get_partitions(items, [0.1, 0.9], 42, partition_names=["asdf"])

    splits = empty_dataset._get_partitions(items, [0.1, 0.9], 42)
    assert set(splits.keys()) == set([0, 1])
    assert len(splits[0]) == 10
    assert len(splits[1]) == 90

    splits = empty_dataset._get_partitions(items, [0.1, 0.9], 42, partition_names=["test", "train"])
    assert set(splits.keys()) == set(["test", "train"])
    assert len(splits["test"]) == 10
    assert len(splits["train"]) == 90

    list_not_sum_up_1 = [
        [0.8, 0.1, 0.3, 0.2],
        [0.3, 0.1, 0.3, 0.5],
        [0.8, 0.1, 0.3],
        [0.3, 0.1, 0.3],
        [0.9, 0.2, 0.3],
        [0.1, 0.1, 0.1],
        [0.95, 0.01, 0.01],
        [0.8, 0.1, 0.3],
        [0.8, 0.3],
        [0.1, 0.7],
    ]
    for wrong_combination in list_not_sum_up_1:
        with pytest.raises(ValueError):
            empty_dataset._get_partitions(items, wrong_combination, 42)

    track_mtrack_dataset = core.Dataset(
        name="test",
        indexes={"default": core.Index("slakh_index_baby.json")},
        track_class=core.Track,
        multitrack_class=core.MultiTrack,
    )

    with pytest.raises(NotImplementedError):
        track_mtrack_dataset.get_track_splits()

    with pytest.raises(NotImplementedError):
        track_mtrack_dataset.get_mtrack_splits()

    # test one real dataset
    test_dataset = mirdata.initialize("slakh", version="2100-redux")
    splits = test_dataset.get_track_splits()
    assert set(splits.keys()) == set(["train", "validation", "test", "omitted"])

    splits = test_dataset.get_mtrack_splits()
    assert set(splits.keys()) == set(["train", "validation", "test", "omitted"])
