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


# The spacing/annotation tests below isolate separator behavior, so they turn
# capitalization off to keep expectations about casing out of scope. Sentence
# capitalization has its own tests further down.


def test_formatter_spaces_across_commits():
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    assert f.format_delta(" the quick") == "the quick"
    assert f.format_delta(" brown fox") == " brown fox"


def test_formatter_no_space_before_punctuation():
    f = TextFormatter(capitalize_sentences=False)
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
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    assert f.format_delta(" (coughs) ") == ""
    # Nothing emitted yet, so the next real text needs no leading space.
    assert f.format_delta(" hi") == "hi"


def test_formatter_annotation_only_first_commit_keeps_leading_strip():
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    assert f.format_delta(" [door slams] , yes") == "yes"


def test_fresh_session_never_starts_with_space():
    # Regression: a worker-lifetime formatter leaked "needs a space" state
    # across dictation toggles, typing a stale leading space into whatever
    # text field the user had focused next.
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" Hello there.")[0] != " "


def test_word_tokens_never_glued_across_commits():
    # Two whole-word deltas must not fuse even when the second lost its leading
    # space upstream. In the streaming path every delta is a complete whisper
    # word token, so a missing leading space means a dropped separator, not a
    # mid-word continuation. Favor separating: gluing produced "thissucks".
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    assert f.format_delta(" some") == "some"
    assert f.format_delta("times") == " times"  # -> "some times", not "sometimes"


def test_dropped_leading_space_still_separates_words():
    # Regression: fast speech / distil models drop whisper's leading-space hint,
    # gluing whole utterances into "Allyouhavetodoisjusttalk.Andthenthissucks".
    f = TextFormatter()
    f.on_utterance_start()
    out = "".join(
        f.format_delta(t)
        for t in ["All", "you", "have", "to", "is", "just", "talk.", "And", "this", "sucks."]
    )
    assert out == "All you have to is just talk. And this sucks."


def test_word_glued_onto_sentence_punctuation_separates():
    # "talk." + "And" (no leading space) must become "talk. And", not "talk.And".
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    assert f.format_delta(" talk.") == "talk."
    assert f.format_delta("And") == " And"


def test_attached_punctuation_still_glues():
    # The separator safety net must NOT split attached punctuation.
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    assert f.format_delta(" hello") == "hello"
    assert f.format_delta(",") == ","  # -> "hello,"
    assert f.format_delta(" world") == " world"
    # contraction / possessive tails stay glued
    g = TextFormatter(capitalize_sentences=False)
    g.on_utterance_start()
    assert g.format_delta(" it") == "it"
    assert g.format_delta("'s") == "'s"  # -> "it's"


def test_opening_quote_gets_separator():
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    assert f.format_delta(" he said") == "he said"
    assert f.format_delta(' "stop') == ' "stop'


def test_capitalizes_session_start():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" hello there") == "Hello there"


def test_capitalizes_after_sentence_end():
    # Whisper lowercases the word after a pause; we fix it. "talk. and" -> "And".
    f = TextFormatter()
    f.on_utterance_start()
    out = "".join(f.format_delta(t) for t in [" all done.", " and then more"])
    assert out == "All done. And then more"


def test_capitalizes_across_utterance_pause():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" talk.") == "Talk."
    # end_utterance emits the trailing space, so the next delta is already
    # separated; it must still be capitalized (new sentence after the pause).
    assert f.end_utterance() == " "
    f.on_utterance_start()
    assert f.format_delta(" and then") == "And then"


def test_does_not_capitalize_mid_sentence():
    f = TextFormatter()
    f.on_utterance_start()
    assert f.format_delta(" the quick") == "The quick"
    assert f.format_delta(" brown fox") == " brown fox"  # 'brown' stays lowercase


def test_capitalize_can_be_disabled():
    f = TextFormatter(capitalize_sentences=False)
    f.on_utterance_start()
    out = "".join(f.format_delta(t) for t in [" all done.", " and more"])
    assert out == "all done. and more"


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
    # or a space-leading session. Capitalization off so the exact-string check
    # stays about spacing.
    f = TextFormatter(capitalize_sentences=False)
    out = ""
    for utterance in [[" One."], [" two,", " three!"], ["(cough)"], [" four"], ["...", " five."]]:
        f.on_utterance_start()
        for delta in utterance:
            out += f.format_delta(delta)
        out += f.end_utterance()
    assert "  " not in out
    assert not out.startswith(" ")
    assert out == "One. two, three! four... five. "


# ── peek (correcting mode's idempotent full-utterance render) ────────────────


def test_peek_matches_format_delta_without_mutating_state():
    f = TextFormatter()
    f.on_utterance_start()
    first = f.peek(" hello world.")
    assert f.peek(" hello world.") == first  # idempotent
    # The eventual state-mutating format of the same raw is identical…
    assert f.format_delta(" hello world.") == first
    # …and only IT advances the spacing state: the same raw now needs a
    # separator because the tail is no longer empty.
    assert f.peek(" hello world.") != first


def test_peek_respects_cross_utterance_tail():
    f = TextFormatter()
    f.on_utterance_start()
    f.format_delta(" First.")
    f.end_utterance()
    f.on_utterance_start()
    # Renders against the previous utterance's tail ("First. ") — no leading
    # separator needed, capitalized as a sentence start.
    assert f.peek(" and then") == "And then"
    assert f.format_delta(" and then") == "And then"


def test_peek_does_not_flip_utterance_has_output():
    f = TextFormatter()
    f.on_utterance_start()
    f.peek(" Done.")
    assert f.end_utterance() == ""  # nothing actually emitted yet
