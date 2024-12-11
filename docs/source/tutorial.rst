.. _tutorial:

########
Tutorial
########

Installation
------------

To install Mirdata:

.. code-block:: console

    pip install mirdata

We recommend to do this inside a conda or virtual environment for reproducibility.

Command-Line Interface
----------------------

To get started with mirdata, execute the following in your console:

.. code-block:: console

    python -m mirdata --help

Print a list of all available dataset loaders by calling:

.. code-block:: console

    python -m mirdata --list
    python -m mirdata -l
    python -m mirdate  # If you don't specify a dataset, it defaults to listing datasets

Download one or more datasets by specifying their name as arguments

.. code-block:: console

    python -m mirdata orchset maestro

You can specify which version of a dataset you'd like to download with `--version`

.. code-block:: console

    python -m mirdata maestro --version=2.0.0
    python -m mirdata maestro -v 2.0.0

mirdata will choose a default location for all MIR datasets. If you'd like to redirect the download destination, you
can do the following

.. code-block:: console

    python -m mirdata maestro --output /opt/data/mir/maestro

By default mirdata will validate any downloaded dataset(s). If you want to skip validation you can use `--no-validate`

.. code-block:: console

    python -m mirdata maestro --no-validate

Or you can print just citations or licenses with `--citation` and `--license`.

.. code-block:: console

    python -m mirdata maestro --citation --license
    python -m mirdata maestro -c -L  #equivalent to above

When you ask for either a license or a citation, mirdata will not download/validate the dataset. If you want to
download _and_ print a license or citation, you can add the --download flag

.. code-block:: console

    python -m mirdata maestro --citation --download
    python -m mirdata maestro -cd


Initializing a dataset
^^^^^^^^^^^^^^^^^^^^^^

Mirdata is easily imported into your Python code by:

.. code-block:: python

    import mirdata



Print a list of all available dataset loaders by calling:

.. code-block:: python

    import mirdata
    print(mirdata.list_datasets())

To use a loader, (for example, ``orchset`) you need to initialize it by calling:

.. code-block:: python

    import mirdata
    orchset = mirdata.initialize('orchset', data_home='/choose/where/data/lives')

Now ``orchset`` is a ``Dataset`` object containing common methods, described below.

You can specify the directory where the Mirdata data is stored by passing a path to ``data_home``.

Mirdata supports working with multiple dataset versions.
To see all available versions of a specific dataset, run ``mirdata.list_dataset_versions('orchset')``.
Use ``version`` parameter if you wish to use a version other than the default one.

.. code-block:: python

    import mirdata
    dataset = mirdata.initialize('orchset', data_home='/choose/where/data/lives', version="1.0")


Downloading a dataset
^^^^^^^^^^^^^^^^^^^^^

All dataset loaders in Mirdata have a ``download()`` function that allows the user to download:

* The :ref:`canonical <faq>` version of the dataset (when available).
* The dataset index, which indicates the list of clips in the dataset and the paths to audio and annotation files.

The index, which is considered part of the source files of Mirdata, is specifically downloaded by running ``download(["index"])``.
Indexes will be directly stored in Mirdata's indexes folder (``mirdata/datasets/indexes``) whereas users can indicate where the dataset files will be stored via ``data_home``.

Downloading a dataset into the default folder:
    In this first example, ``data_home`` is not specified. Thus, ORCHSET will be downloaded and retrieved from ``mir_datasets``
    folder created at user root folder:

    .. code-block:: python

        import mirdata
        orchset = mirdata.initialize('orchset')
        orchset.download()  # Dataset is downloaded to ~/mir_datasets/orchset

Downloading a dataset into a specified folder:
    Now ``data_home`` is specified and so orchset will be read from / written to this custom location:

    .. code-block:: python

        orchset = mirdata.initialize('orchset', data_home='Users/leslieknope/Desktop/orchset123')
        orchset.download()  # Dataset is downloaded to the folder "orchset123" Leslie Knope's desktop


Partially downloading a dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``download()`` functions allows partial downloads of a dataset. In other words, if applicable, the user can
select which elements of the dataset they want to download. Each dataset has a ``REMOTES`` dictionary were all
the available elements are listed.

``cante100`` has different elements as seen in the ``REMOTES`` dictionary. Thus, we can specify which of these elements are
downloaded, by passing to the ``download()`` function the list of keys in ``REMOTES`` that we are interested in. This
list is passed to the ``download()`` function through the ``partial_download`` variable.

.. admonition:: Example REMOTES
    :class: dropdown

    .. code-block:: python

        REMOTES = {
            "spectrogram": download_utils.RemoteFileMetadata(
                filename="cante100_spectrum.zip",
                url="https://zenodo.org/record/1322542/files/cante100_spectrum.zip?download=1",
                checksum="0b81fe0fd7ab2c1adc1ad789edb12981",  # the md5 checksum
                destination_dir="cante100_spectrum",  # relative path for where to unzip the data, or None
            ),
            "melody": download_utils.RemoteFileMetadata(
                filename="cante100midi_f0.zip",
                url="https://zenodo.org/record/1322542/files/cante100midi_f0.zip?download=1",
                checksum="cce543b5125eda5a984347b55fdcd5e8",  # the md5 checksum
                destination_dir="cante100midi_f0",  # relative path for where to unzip the data, or None
            ),
            "notes": download_utils.RemoteFileMetadata(
                filename="cante100_automaticTranscription.zip",
                url="https://zenodo.org/record/1322542/files/cante100_automaticTranscription.zip?download=1",
                checksum="47fea64c744f9fe678ae5642a8f0ee8e",  # the md5 checksum
                destination_dir="cante100_automaticTranscription",  # relative path for where to unzip the data, or None
            ),
            "metadata": download_utils.RemoteFileMetadata(
                filename="cante100Meta.xml",
                url="https://zenodo.org/record/1322542/files/cante100Meta.xml?download=1",
                checksum="6cce186ce77a06541cdb9f0a671afb46",  # the md5 checksum
            ),
            "README": download_utils.RemoteFileMetadata(
                filename="cante100_README.txt",
                url="https://zenodo.org/record/1322542/files/cante100_README.txt?download=1",
                checksum="184209b7e7d816fa603f0c7f481c0aae",  # the md5 checksum
            ),
        }

A partial download example for ``cante100`` dataset could be:

.. code-block:: python

    cante100.download(partial_download=['spectrogram', 'melody', 'metadata'])


Validating a dataset
^^^^^^^^^^^^^^^^^^^^

Using the method ``validate()`` we can check if the files in the local version are the same than the available canonical version,
and the files were downloaded correctly (none of them are corrupted).

For big datasets: In future Mirdata versions, a random validation will be included. This improvement will reduce validation time for very big datasets.

Accessing annotations
^^^^^^^^^^^^^^^^^^^^^

We can choose a random track from a dataset with the ``choice_track()`` method.

.. admonition:: Loading annotations
    :class: dropdown

    .. code-block:: python

        random_track = orchset.choice_track()
        print(random_track)
        >>> Track(
               alternating_melody=True,
               audio_path_mono="user/mir_datasets/orchset/audio/mono/Beethoven-S3-I-ex1.wav",
               audio_path_stereo="user/mir_datasets/orchset/audio/stereo/Beethoven-S3-I-ex1.wav",
               composer="Beethoven",
               contains_brass=False,
               contains_strings=True,
               contains_winds=True,
               excerpt="1",
               melody_path="user/mir_datasets/orchset/GT/Beethoven-S3-I-ex1.mel",
               only_brass=False,
               only_strings=False,
               only_winds=False,
               predominant_melodic_instruments=['strings', 'winds'],
               track_id="Beethoven-S3-I-ex1",
               work="S3-I",
               audio_mono: (np.ndarray, float),
               audio_stereo: (np.ndarray, float),
               melody: F0Data,
            )


We can also access specific tracks by id.
The available track ids can be accessed via the ``.track_ids`` attribute.
In the next example we take the first track id, and then we retrieve the melody
annotation.

.. code-block:: python

    orchset_ids = orchset.track_ids  # the list of orchset's track ids
    orchset_data = orchset.load_tracks()  # Load all tracks in the dataset
    example_track = orchset_data[orchset_ids[0]]  # Get the first track

    # Accessing the track's melody annotation
    example_melody = example_track.melody


Alternatively, we don't need to load the whole dataset to get a single track.

.. code-block:: python

    orchset_ids = orchset.track_ids  # the list of orchset's track ids
    example_track = orchset.track(orchset_ids[0])  # load this particular track
    example_melody = example_track.melody  # Get the melody from first track


.. _Remote Data Example:

Accessing data on non-local filesystems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mirdata uses the smart_open_ library, which supports non-local filesystems such as GCS and AWS.
If your data lives, e.g. on Google Cloud Storage (GCS), simply set the ``data_home`` variable accordingly
when initializing a dataset. For example:

.. _smart_open: https://pypi.org/project/smart-open/

.. admonition:: Accessing annotations remotely
    :class: dropdown

    .. code-block:: python

        import mirdata

        orchset = mirdata.initialize("orchset", data_home="gs://my-bucket/my-subfolder/orchset")

        # everything should work the same as if the data were local
        orchset.validate()

        example_track = orchset.choice_track()
        melody = example_track.melody
        y, fs = example_track.audio_mono


    Note that the data on the remote file system must have identical folder structure to what is specified by ``dataset.download()``,
    and we do not support downloading (i.e. writing) to remote filesystems, only reading from them. To prepare a new dataset to use with mirdata,
    we recommend running ``dataset.download()`` on a local filesystem, and then manually transfering the folder contents to the remote
    filesystem.

.. admonition:: mp3 data
    :class: dropdown, warning

    For a variety of reasons, mirdata doesn't support remote reading of mp3 files, so some datasets with
    mp3 audio may have tracks unavailable attributes.


Annotation classes
^^^^^^^^^^^^^^^^^^

Mirdata defines annotation-specific data classes. These data classes are meant to standardize the format for
all loaders, and are compatibly with `jams <https://jams.readthedocs.io/en/stable/>`_ and `mir_eval <https://craffel.github.io/mir_eval/>`_.

The list and descriptions of available annotation classes can be found in :ref:`annotations`.

.. note:: These classes may be extended in the case that a loader requires it.

Iterating over datasets and annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general, most datasets are a collection of tracks, and in most cases each track has an audio file along with annotations.

With the ``load_tracks()`` method, all tracks are loaded as a dictionary with the ids as keys and
track objects (which include their respective audio and annotations, which are lazy-loaded on access) as values.

.. code-block:: python

    orchset = mirdata.initialize('orchset')
    for key, track in orchset.load_tracks().items():
        print(key, track.audio_path)


Alternatively, we can loop over the ``track_ids`` list to directly access each track in the dataset.

.. code-block:: python

    orchset = mirdata.initialize('orchset')
    for track_id in orchset.track_ids:

        print(track_id, orchset.track(track_id).audio_path)


Basic example: including mirdata in your pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we wanted to use ``orchset`` to evaluate the performance of a melody extraction algorithm
(in our case, ``very_bad_melody_extractor``), and then split the scores based on the
metadata, we could do the following:

.. admonition:: mirdata usage example
    :class: dropdown

    .. code-block:: python

        import mir_eval
        import mirdata
        import numpy as np
        import sox

        def very_bad_melody_extractor(audio_path):
            duration = sox.file_info.duration(audio_path)
            time_stamps = np.arange(0, duration, 0.01)
            melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)
            return time_stamps, melody_f0

        # Evaluate on the full dataset
        orchset = mirdata.initialize("orchset")
        orchset_scores = {}
        orchset_data = orchset.load_tracks()
        for track_id, track_data in orchset_data.items():
            est_times, est_freqs = very_bad_melody_extractor(track_data.audio_path_mono)

            ref_melody_data = track_data.melody
            ref_times = ref_melody_data.times
            ref_freqs = ref_melody_data.frequencies

            score = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
            orchset_scores[track_id] = score

        # Split the results by composer and by instrumentation
        composer_scores = {}
        strings_no_strings_scores = {True: {}, False: {}}
        for track_id, track_data in orchset_data.items():
            if track_data.composer not in composer_scores.keys():
                composer_scores[track_data.composer] = {}

            composer_scores[track_data.composer][track_id] = orchset_scores[track_id]
            strings_no_strings_scores[track_data.contains_strings][track_id] = \
                orchset_scores[track_id]


This is the result of the example above.

.. admonition:: Example result
    :class: dropdown

    .. code-block:: python

        print(strings_no_strings_scores)
        >>> {True: {
                'Beethoven-S3-I-ex1':OrderedDict([
                       ('Voicing Recall', 1.0),
                       ('Voicing False Alarm', 1.0),
                       ('Raw Pitch Accuracy', 0.029798422436459245),
                       ('Raw Chroma Accuracy', 0.08063102541630149),
                       ('Overall Accuracy', 0.0272654370489174)
                       ]),
                'Beethoven-S3-I-ex2': OrderedDict([
                       ('Voicing Recall', 1.0),
                       ('Voicing False Alarm', 1.0),
                       ('Raw Pitch Accuracy', 0.009221311475409836),
                       ('Raw Chroma Accuracy', 0.07377049180327869),
                       ('Overall Accuracy', 0.008754863813229572)]),
                ...

                'Wagner-Tannhauser-Act2-ex2': OrderedDict([
                       ('Voicing Recall', 1.0),
                       ('Voicing False Alarm', 1.0),
                       ('Raw Pitch Accuracy', 0.03685636856368564),
                       ('Raw Chroma Accuracy', 0.08997289972899729),
                       ('Overall Accuracy', 0.036657681940700806)])
                }}

You can see that ``very_bad_melody_extractor`` performs very badly!

.. _Using mirdata with tensorflow:

Using mirdata with tensorflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a simple example of a generator that can be used to create a tensorflow Dataset.

.. admonition:: mirdata with tf.data.Dataset example
    :class: dropdown

    .. code-block:: python

        import mirdata
        import numpy as np
        import tensorflow as tf

        def orchset_generator():
            # using the default data_home
            orchset = mirdata.initialize("orchset")
            track_ids = orchset.track_ids
            for track_id in track_ids:
                track = orchset.track(track_id)
                audio_signal, sample_rate = track.audio_mono
                yield {
                    "audio": audio_signal.astype(np.float32),
                    "sample_rate": sample_rate,
                    "annotation": {
                        "times": track.melody.times.astype(np.float32),
                        "freqs": track.melody.frequencies.astype(np.float32),
                    },
                    "metadata": {"track_id": track.track_id}
                }

        dataset = tf.data.Dataset.from_generator(
            orchset_generator,
            {
                "audio": tf.float32,
                "sample_rate": tf.float32,
                "annotation": {"times": tf.float32, "freqs": tf.float32},
                "metadata": {'track_id': tf.string}
            }
        )

In future Mirdata versions, generators for Tensorflow and Pytorch will be included.
