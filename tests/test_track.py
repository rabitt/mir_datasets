# -*- coding: utf-8 -*-

import sys
import pytest
import numpy as np

from mirdata import track

if sys.version_info.major == 3:
    builtin_module_name = "builtins"
else:
    builtin_module_name = "__builtin__"


def test_track_repr():
    class TestTrack(track.Track):
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
            """I have an improper docstring"""
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

    class NoDocsTrack(track.Track):
        @property
        def no_doc(self):
            return "whee!"

    bad_track = NoDocsTrack()
    with pytest.raises(ValueError):
        bad_track.__repr__()


def test_multitrack():
    class TestTrack(track.Track):
        def __init__(self, key):
            self.key = key

        def f(self):
            return np.random.uniform(-1, 1, (100, 2))

    track_keys = ["a", "b", "c"]
    tracks = {k: TestTrack(k) for k in track_keys}

    mtrack = track.MultiTrack(tracks, "f")

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (100, 2)
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["b", "c"], weights=[0.5, 0.2])
    assert target2.shape == (100, 2)
    assert np.max(np.abs(target2)) <= 1

    target3 = mtrack.get_target(["b", "c"], weights=[0.5, 5])
    assert target3.shape == (100, 2)
    assert np.max(np.abs(target3)) <= 1

    random_target1, t1, w1 = mtrack.get_random_target(n_tracks=2)
    assert random_target1.shape == (100, 2)
    assert np.max(np.abs(random_target1)) <= 1
    assert len(t1) == 2
    assert len(w1) == 2
    assert np.all(w1 >= 0.3)
    assert np.all(w1 <= 1.0)

    random_target2, t2, w2 = mtrack.get_random_target(n_tracks=5)
    assert random_target2.shape == (100, 2)
    assert np.max(np.abs(random_target2)) <= 1
    assert len(t2) == 3
    assert len(w2) == 3
    assert np.all(w2 >= 0.3)
    assert np.all(w2 <= 1.0)

    random_target3, t3, w3 = mtrack.get_random_target()
    assert random_target3.shape == (100, 2)
    assert np.max(np.abs(random_target3)) <= 1
    assert len(t3) == 3
    assert len(w3) == 3
    assert np.all(w3 >= 0.3)
    assert np.all(w3 <= 1.0)

    random_target4, t4, w4 = mtrack.get_random_target(
        n_tracks=2, min_weight=0.1, max_weight=0.4
    )
    assert random_target4.shape == (100, 2)
    assert np.max(np.abs(random_target4)) <= 1
    assert len(t4) == 2
    assert len(w4) == 2
    assert np.all(w4 >= 0.1)
    assert np.all(w4 <= 0.4)

    mix = mtrack.get_mix()
    assert mix.shape == (100, 2)
