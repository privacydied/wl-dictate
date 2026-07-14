from wldictate.textproc import TextFormatter, clean_text


def test_clean_strips_annotations():
    assert clean_text(" (coughs) hello [music] world", utterance_start=True) == "hello world"


def test_clean_normalizes_punctuation_spacing():
    assert clean_text("a ,b .c", utterance_start=True) == "a, b. c"


def test_clean_preserves_ellipsis():
    out = clean_text("well ... maybe", utterance_start=True)
    assert "..." in out
    assert ". . ." not in out


def test_clean_strips_leading_punct_only_at_utterance_start():
    assert clean_text(", hello", utterance_start=True) == "hello"
    assert clean_text(", hello", utterance_start=False) == ", hello"


def test_formatter_spaces_across_commits():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" the quick") == "the quick"
    assert f.format_delta(" brown fox") == " brown fox"


def test_formatter_no_space_before_punctuation():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" hello") == "hello"
    # A delta that begins with punctuation continues the previous word.
    assert f.format_delta(", world") == ", world"


def test_formatter_spaces_across_utterances():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" First sentence.") == "First sentence."
    f.on_utterance_start()
    assert f.format_delta(" Second one.") == " Second one."


def test_formatter_empty_after_cleanup():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" (coughs) ") == ""
    # Nothing emitted yet, so the next real text needs no leading space.
    assert f.format_delta(" hi") == "hi"


def test_formatter_annotation_only_first_commit_keeps_leading_strip():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" [door slams] , yes") == "yes"
