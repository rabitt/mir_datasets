"""Utilities for converting mirdata Annotation classes to jams format.
"""

import logging
import os

import jams
import librosa
from smart_open import open, parse_uri

from mirdata import annotations


def jams_converter(
    audio_path=None,
    spectrogram_path=None,
    beat_data=None,
    chord_data=None,
    note_data=None,
    f0_data=None,
    section_data=None,
    multi_section_data=None,
    tempo_data=None,
    event_data=None,
    key_data=None,
    lyrics_data=None,
    tags_gtzan_data=None,
    tags_open_data=None,
    metadata=None,
):
    """Convert annotations from a track to JAMS format.

    Args:
        audio_path (str or None):
            A path to the corresponding audio file, or None. If provided,
            the audio file will be read to compute the duration. If None,
            'duration' must be a field in the metadata dictionary, or the
            resulting jam object will not validate.
        spectrogram_path (str or None):
            A path to the corresponding spectrum file, or None.
        beat_data (list or None):
            A list of tuples of (annotations.BeatData, str), where str describes
            the annotation (e.g. 'beats_1').
        chord_data (list or None):
            A list of tuples of (annotations.ChordData, str), where str describes the annotation.
        note_data (list or None):
            A list of tuples of (annotations.NoteData, str), where str describes the annotation.
        f0_data (list or None):
            A list of tuples of (annotations.F0Data, str), where str describes the annotation.
        section_data (list or None):
            A list of tuples of (annotations.SectionData, str), where str describes the annotation.
        multi_section_data (list or None):
            A list of tuples. Tuples in multi_section_data should contain another
            list of tuples, indicating annotations in the different levels
            e.g. ([(segments0, level0), '(segments1, level1)], annotator) and a str
            indicating the annotator
        tempo_data (list or None):
            A list of tuples of (float, str), where float gives the tempo in bpm
            and str describes the annotation.
        event_data (list or None):
            A list of tuples of (annotations.EventData, str), where str describes the annotation.
        key_data (list or None):
            A list of tuples of (annotations.KeyData, str), where str describes the annotation.
        lyrics_data (list or None):
            A list of tuples of (annotations.LyricData, str), where str describes the annotation.
        tags_gtzan_data (list or None):
            A list of tuples of (str, str), where the first srt is the tag and the second
            is a descriptor of the annotation.
        tags_open_data (list or None):
            A list of tuples of (str, str), where the first srt is the tag and the second
            is a descriptor of the annotation.
        metadata (dict or None):
            A dictionary containing the track metadata.

    Returns:
        jams.JAMS: A JAMS object containing the annotations.

    """

    jam = jams.JAMS()

    # duration
    duration = None
    if audio_path is not None:
        file_uri = parse_uri(audio_path)
        if file_uri.scheme != "file" and audio_path.endswith(".mp3"):
            raise NotImplementedError(
                "mirdata does not currently support conversion of remote mp3 files to jams format"
            )

        if os.path.exists(audio_path):
            duration = librosa.get_duration(path=audio_path)
        else:
            raise FileNotFoundError(
                "jams conversion failed because the audio file "
                + "for this track cannot be found, and it is required "
                + "to compute duration."
            )
    if spectrogram_path is not None:
        if audio_path is None:
            duration = metadata["duration"]

    # metadata
    if metadata is not None:
        for key in metadata:
            if (
                key == "duration"
                and duration is not None
                and metadata[key] != duration
                and audio_path is not None
            ):
                logging.warning(
                    "Duration provided in metadata does not"
                    + "match the duration computed from the audio file."
                    + "Using the duration provided by the metadata."
                )

            if metadata[key] is None:
                continue

            if hasattr(jam.file_metadata, key):
                setattr(jam.file_metadata, key, metadata[key])
            else:
                setattr(jam.sandbox, key, metadata[key])

    if jam.file_metadata.duration is None:
        jam.file_metadata.duration = duration

    # beats
    if beat_data is not None:
        if not isinstance(beat_data, list):
            raise TypeError("beat_data should be a list of tuples")
        for beats in beat_data:
            if not isinstance(beats, tuple):
                raise TypeError(
                    "beat_data should be a list of tuples, "
                    + "but contains a {} element".format(type(beats))
                )
            jam.annotations.append(beats_to_jams(beats[0], beats[1]))

    # sections
    if section_data is not None:
        if not isinstance(section_data, list):
            raise TypeError("section_data should be a list of tuples")
        for sections in section_data:
            if not isinstance(sections, tuple):
                raise TypeError(
                    "section_data should be a list of tuples, "
                    + "but contains a {} element".format(type(sections))
                )
            jam.annotations.append(sections_to_jams(sections[0], sections[1]))

    # multi-sections (sections with multiple levels)
    if multi_section_data is not None:
        if not isinstance(multi_section_data, list):
            raise TypeError("multi_section_data should be a list of tuples")
        for sections in multi_section_data:
            if not isinstance(sections, tuple):
                raise TypeError(
                    "multi_section_data should be a list of tuples, "
                    + "but contains a {} element".format(type(sections))
                )
            if not (
                isinstance(sections[0], list) and isinstance(sections[0][0], tuple)
            ):
                raise TypeError(
                    "tuples in multi_section_data should contain a "
                    + "list of tuples, indicating annotations in the different "
                    + "levels, e.g. ([(segments0, level0), "
                    + "(segments1, level1)], annotator)"
                )
            jam.annotations.append(multi_sections_to_jams(sections[0], sections[1]))

    # tempo
    if tempo_data is not None:
        if type(tempo_data) != list:
            raise TypeError("tempo_data should be a list of tuples")
        for tempo in tempo_data:
            if type(tempo) != tuple:
                raise TypeError(
                    "tempo_data should be a list of tuples, "
                    + "but contains a {} element".format(type(tempo))
                )
            jam.annotations.append(tempos_to_jams(tempo[0], tempo[1]))

    # events
    if event_data is not None:
        if type(event_data) != list:
            raise TypeError("event_data should be a list of tuples")
        for events in event_data:
            if type(events) != tuple:
                raise TypeError(
                    "event_data should be a list of tuples, "
                    + "but contains a {} element".format(type(events))
                )
            jam.annotations.append(events_to_jams(events[0], events[1]))

    # chords
    if chord_data is not None:
        if not isinstance(chord_data, list):
            raise TypeError("chord_data should be a list of tuples")
        for chords in chord_data:
            if not isinstance(chords, tuple):
                raise TypeError(
                    "chord_data should be a list of tuples, "
                    + "but contains a {} element".format(type(chords))
                )
            jam.annotations.append(chords_to_jams(chords[0], chords[1]))

    # notes
    if note_data is not None:
        if not isinstance(note_data, list):
            raise TypeError("note_data should be a list of tuples")
        for notes in note_data:
            if not isinstance(notes, tuple):
                raise TypeError(
                    "note_data should be a list of tuples, "
                    + "but contains a {} element".format(type(notes))
                )
            jam.annotations.append(notes_to_jams(notes[0], notes[1]))

    # keys
    if key_data is not None:
        if not isinstance(key_data, list):
            raise TypeError("key_data should be a list of tuples")
        for keys in key_data:
            if not isinstance(keys, tuple):
                raise TypeError(
                    "key_data should be a list of tuples, "
                    + "but contains a {} element".format(type(keys))
                )
            jam.annotations.append(keys_to_jams(keys[0], keys[1]))

    # f0
    if f0_data is not None:
        if not isinstance(f0_data, list):
            raise TypeError("f0_data should be a list of tuples")
        for f0s in f0_data:
            if not isinstance(f0s, tuple):
                raise TypeError(
                    "f0_data should be a list of tuples, "
                    + "but contains a {} element".format(type(f0s))
                )
            jam.annotations.append(f0s_to_jams(f0s[0], f0s[1]))

    # lyrics
    if lyrics_data is not None:
        if not isinstance(lyrics_data, list):
            raise TypeError("lyrics_data should be a list of tuples")
        for lyrics in lyrics_data:
            if not isinstance(lyrics, tuple):
                raise TypeError(
                    "lyrics_data should be a list of tuples, "
                    + "but contains a {} element".format(type(lyrics))
                )
            jam.annotations.append(lyrics_to_jams(lyrics[0], lyrics[1]))

    # tags
    if tags_gtzan_data is not None:
        if not isinstance(tags_gtzan_data, list):
            raise TypeError("tags_gtzan_data should be a list of tuples")
        for tag in tags_gtzan_data:
            if not isinstance(tag, tuple):
                raise TypeError(
                    "tags_gtzan_data should be a list of tuples, "
                    + "but contains a {} element".format(type(tag))
                )
            jam.annotations.append(tag_to_jams(tag[0], "tag_gtzan", tag[1]))

    # tag open
    if tags_open_data is not None:
        if not isinstance(tags_open_data, list):
            raise TypeError("tags_open_data should be a list of tuples")
        for tag in tags_open_data:
            if not isinstance(tag, tuple):
                raise TypeError(
                    "tags_open_data should be a list of tuples, "
                    + "but contains a {} element".format(type(tag))
                )
            jam.annotations.append(tag_to_jams(tag[0], "tag_open", tag[1]))

    return jam


def beats_to_jams(beat_data, description=None):
    """Convert beat annotations into jams format.

    Args:
        beat_data (annotations.BeatData): beat data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_beat = jams.Annotation(namespace="beat")
    jannot_beat.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if beat_data is not None:
        if not isinstance(beat_data, annotations.BeatData):
            raise TypeError("Type should be BeatData.")
        if beat_data.positions is not None:
            for t, p in zip(beat_data.times, beat_data.positions):
                jannot_beat.append(time=t, duration=0.0, value=p)
        else:
            for t in beat_data.times:
                jannot_beat.append(time=t, duration=0.0, value=None)
    if description is not None:
        jannot_beat.sandbox = jams.Sandbox(name=description)
    return jannot_beat


def sections_to_jams(section_data, description=None):
    """Convert section annotations into jams format.

    Args:
        section_data (annotations.SectionData): section data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_seg = jams.Annotation(namespace="segment_open")
    jannot_seg.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if section_data is not None:
        if not isinstance(section_data, annotations.SectionData):
            raise TypeError("Type should be SectionData.")
        for inter, seg in zip(section_data.intervals, section_data.labels):
            jannot_seg.append(time=inter[0], duration=inter[1] - inter[0], value=seg)
    if description is not None:
        jannot_seg.sandbox = jams.Sandbox(name=description)
    return jannot_seg


def chords_to_jams(chord_data, description=None):
    """Convert chord annotations into jams format.

    Args:
        chord_data (annotations.ChordData): chord data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_chord = jams.Annotation(namespace="chord")
    jannot_chord.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if chord_data is not None:
        if not isinstance(chord_data, annotations.ChordData):
            raise TypeError("Type should be ChordData.")
        for beg, end, ch in zip(
            chord_data.intervals[:, 0], chord_data.intervals[:, 1], chord_data.labels
        ):
            jannot_chord.append(time=beg, duration=end - beg, value=ch)
    if description is not None:
        jannot_chord.sandbox = jams.Sandbox(name=description)
    return jannot_chord


def notes_to_jams(note_data, description):
    """Convert note annotations into jams format.

    Args:
        note_data (annotations.NoteData): note data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_note = jams.Annotation(namespace="note_hz")
    jannot_note.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if note_data is not None:
        if not isinstance(note_data, annotations.NoteData):
            raise TypeError("Type should be NoteData.")
        for beg, end, n in zip(
            note_data.intervals[:, 0], note_data.intervals[:, 1], note_data.pitches
        ):
            jannot_note.append(time=beg, duration=end - beg, value=n)
    if description is not None:
        jannot_note.sandbox = jams.Sandbox(name=description)
    return jannot_note


def keys_to_jams(key_data, description):
    """Convert key annotations into jams format.

    Args:
        key_data (annotations.KeyData): key data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_key = jams.Annotation(namespace="key_mode")
    jannot_key.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if key_data is not None:
        if not isinstance(key_data, annotations.KeyData):
            raise TypeError("Type should be KeyData.")
        for beg, end, key in zip(
            key_data.intervals[:, 0], key_data.intervals[:, 1], key_data.keys
        ):
            jannot_key.append(time=beg, duration=end - beg, value=key)
    if description is not None:
        jannot_key.sandbox = jams.Sandbox(name=description)
    return jannot_key


def multi_sections_to_jams(multisection_data, description):
    """Convert multi-section annotations into jams format.

    Args:
        multisection_data (list): list of tuples of the form [(SectionData, int)]
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    # sections with multiple annotators and multiple level annotations
    jannot_multi = jams.Annotation(namespace="multi_segment")
    jannot_multi.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")
    jannot_multi.annotation_metadata = jams.AnnotationMetadata(
        annotator={"name": description}
    )
    for sections in multisection_data:
        if sections[0] is not None:
            if not isinstance(sections[0], annotations.SectionData):
                raise TypeError("Type should be SectionData.")
            for inter, seg in zip(sections[0].intervals, sections[0].labels):
                jannot_multi.append(
                    time=inter[0],
                    duration=inter[1] - inter[0],
                    value={"label": seg, "level": sections[1]},
                )
    return jannot_multi


def tempos_to_jams(tempo_data, description=None):
    """Convert tempo annotations into jams format.

    Args:
        tempo_data (annotations.TempoData): tempo data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_tempo = jams.Annotation(namespace="tempo")
    jannot_tempo.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")
    if tempo_data is not None:
        if not isinstance(tempo_data, float) and not isinstance(tempo_data, int):
            raise TypeError("Type should be float or int.")
        jannot_tempo.append(time=0, duration=0, confidence=1, value=tempo_data)
    if description is not None:
        jannot_tempo.sandbox = jams.Sandbox(name=description)
    return jannot_tempo


def events_to_jams(event_data, description=None):
    """Convert events annotations into jams format.

    Args:
        event_data (annotations.EventData): event data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_events = jams.Annotation(namespace="tag_open")
    jannot_events.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if event_data is not None:
        if not isinstance(event_data, annotations.EventData):
            raise TypeError("Type should be EventData.")
        for beg, end, label in zip(
            event_data.intervals[:, 0], event_data.intervals[:, 1], event_data.events
        ):
            jannot_events.append(time=beg, duration=end - beg, value=label)
    if description is not None:
        jannot_events.sandbox = jams.Sandbox(name=description)
    return jannot_events


def f0s_to_jams(f0_data, description=None):
    """Convert f0 annotations into jams format.

    Args:
        f0_data (annotations.F0Data): f0 annotation object
        description (str): annotation descriptoin

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_f0 = jams.Annotation(namespace="pitch_contour")
    jannot_f0.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if f0_data is not None:
        if not isinstance(f0_data, annotations.F0Data):
            raise TypeError("Type should be F0Data.")
        if f0_data._confidence is None:
            conf = [None for t in f0_data.times]
        else:
            conf = f0_data._confidence
        for t, f, v, c in zip(
            f0_data.times, f0_data.frequencies, f0_data.voicing, conf
        ):
            jannot_f0.append(
                time=t,
                duration=0.0,
                value={"index": 0, "frequency": f, "voiced": v},
                confidence=c,
            )
    if description is not None:
        jannot_f0.sandbox = jams.Sandbox(name=description)
    return jannot_f0


def lyrics_to_jams(lyric_data, description=None):
    """Convert lyric annotations into jams format.

    Args:
        lyric_data (annotations.LyricData): lyric annotation object
        description (str): annotation descriptoin

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_lyric = jams.Annotation(namespace="lyrics")
    jannot_lyric.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")

    if lyric_data is not None:
        if not isinstance(lyric_data, annotations.LyricData):
            raise TypeError("Type should be LyricData.")
        for beg, end, lyric in zip(
            lyric_data.intervals[:, 0], lyric_data.intervals[:, 1], lyric_data.lyrics
        ):
            jannot_lyric.append(time=beg, duration=end - beg, value=lyric)
    if description is not None:
        jannot_lyric.sandbox = jams.Sandbox(name=description)
    return jannot_lyric


def tag_to_jams(tag_data, namespace="tag_open", description=None):
    """Convert tag annotations into jams format.

    Args:
        tags (annotations.Tags): tags annotation object
        namespace (str): the jams-compatible tag namespace
        description (str): annotation description (default is None)

    Returns:
        jams.Annotation: jams annotation object.

    """
    ann = jams.Annotation(namespace=namespace)
    ann.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")
    for t, c in zip(tag_data.labels, tag_data.confidence):
        ann.append(time=0.0, duration=duration, value=t, confidence=c)
    if description is not None:
        ann.sandbox = jams.Sandbox(name=description)
    return ann

    jannot_tag = jams.Annotation(namespace=namespace)
    jannot_tag.annotation_metadata = jams.AnnotationMetadata(data_source="mirdata")
    if tag_data is not None:
        if not isinstance(tag_data, str):
            raise TypeError("Type should be str.")
        jannot_tag.append(time=0.0, duration=0.0, value=tag_data)
    if description is not None:
        jannot_tag.sandbox = jams.Sandbox(name=description)
    return jannot_tag
