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


def test_fresh_session_never_starts_with_space():
    # Regression: a worker-lifetime formatter leaked "needs a space" state
    # across dictation toggles, typing a stale leading space into whatever
    # text field the user had focused next.
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" Hello there.")[0] != " "


def test_continuation_token_not_split():
    # Whisper continuation tokens carry no leading space and must be glued.
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" some") == "some"
    assert f.format_delta("times") == "times"  # -> "sometimes"


def test_opening_quote_gets_separator():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" he said") == "he said"
    assert f.format_delta(' "stop') == ' "stop'


def test_trailing_space_after_sentence_end():
    f = TextFormatter()
    f.on_utterance_start()
    f.format_delta(" First sentence.")
    assert f.end_utterance() == " "
    # Next utterance must not double-space after the trailer.
    f.on_utterance_start()
    assert f.format_delta(" Second.") == "Second."


def test_no_trailing_space_mid_sentence():
    f = TextFormatter()
    f.on_utterance_start()
    f.format_delta(" trailing off and")
    assert f.end_utterance() == ""
    # ...and the next utterance supplies the separator instead.
    f.on_utterance_start()
    assert f.format_delta(" then continued") == " then continued"


def test_trailing_space_after_quoted_sentence_end():
    f = TextFormatter()
    f.on_utterance_start()
    f.format_delta(' he said "stop."')
    assert f.end_utterance() == " "


def test_end_utterance_idempotent_and_safe_when_empty():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.end_utterance() == ""  # nothing emitted: no stray space
    f.format_delta(" Done.")
    assert f.end_utterance() == " "
    assert f.end_utterance() == ""  # already spaced: never a double space


def test_trailing_space_configurable():
    f = TextFormatter(sentence_trailing_space=False)
    f.on_utterance_start()
    f.format_delta(" Done.")
    assert f.end_utterance() == ""


def test_never_double_spaces_across_any_boundary():
    # Property-style: arbitrary utterance/commit patterns never produce "  "
    # or a space-leading session.
    f = TextFormatter()
    out = ""
    for utterance in [[" One."], [" two,", " three!"], ["(cough)"], [" four"], ["...", " five."]]:
        f.on_utterance_start()
        for delta in utterance:
            out += f.format_delta(delta)
        out += f.end_utterance()
    assert "  " not in out
    assert not out.startswith(" ")
    assert out == "One. two, three! four... five. "
