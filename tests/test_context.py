"""Unit tests for the pure vocabulary-extraction logic."""

from heard.context import augment_prompt, extract_terms, merge_hotwords


def _terms(text):
    return extract_terms(text)


def test_keeps_technical_shapes():
    text = "the opossum-ec wrapper reads ec-image/PROD-EC-RUNBOOK.md via kubectl"
    terms = _terms(text)
    assert "opossum-ec" in terms  # kebab-case
    assert "ec-image" in terms  # kebab-case
    assert "PROD-EC-RUNBOOK" in terms  # ALLCAPS + hyphen
    assert "kubectl" in terms  # long plain proper-noun-ish
    # short plain lowercase words (e.g. the "md" extension) are low-signal and
    # intentionally dropped; only technical-shaped short tokens survive.
    assert "md" not in terms


def test_drops_plain_stopwords_and_short_words():
    terms = _terms("the wrapper reads a file and then it will run")
    lowered = {t.lower() for t in terms}
    for junk in ("the", "and", "then", "it", "will", "run", "a"):
        assert junk not in lowered
    assert "wrapper" in terms


def test_camelcase_and_snake_and_digits():
    terms = _terms("call buildBias() then read snake_case and check z20 s16 k3s")
    assert "buildBias" in terms
    assert "snake_case" in terms
    assert "z20" in terms
    assert "s16" in terms
    assert "k3s" in terms


def test_splits_paths_and_dotted():
    terms = _terms("edit src/heard/context.py and heard.transcribe module")
    assert "context" in terms  # split out of the path
    assert "heard" in terms
    assert "transcribe" in terms  # split out of the dotted module
    # separators themselves never survive
    assert "/" not in "".join(terms)
    assert "." not in "".join(terms)


def test_dedup_case_insensitive_keeps_first_casing():
    terms = _terms("Podman podman PODMAN")
    hits = [t for t in terms if t.lower() == "podman"]
    assert hits == ["Podman"]


def test_ranking_technical_before_plain():
    # A plain word repeated a lot should still rank below a technical token.
    text = "runbook runbook runbook runbook ec-image"
    terms = _terms(text)
    assert terms.index("ec-image") < terms.index("runbook")


def test_titlecase_words_are_not_technical():
    # Plain Title-case sentence words must rank as plain (below real
    # identifiers), not be misread as camelCase.
    text = "Welcome Fixed buildBias ec-image"
    terms = _terms(text)
    assert terms.index("buildBias") < terms.index("Welcome")
    assert terms.index("ec-image") < terms.index("Fixed")


def test_pure_numbers_and_punctuation_dropped():
    terms = _terms("2026 07 13 :: --- === 42")
    assert terms == []


def test_merge_hotwords_dedups_and_limits():
    out = merge_hotwords("git podman", ["podman", "kubectl", "nftables"], limit=3)
    assert out.split() == ["git", "podman", "kubectl"]


def test_merge_hotwords_static_first():
    out = merge_hotwords("alpha", ["beta", "gamma"], limit=10)
    assert out.split()[0] == "alpha"


def test_augment_prompt_appends_clause():
    out = augment_prompt("Base prompt.", ["foo", "bar"], limit=2)
    assert out == "Base prompt. On screen: foo, bar."


def test_augment_prompt_noop_when_disabled():
    assert augment_prompt("Base.", ["foo"], limit=0) == "Base."


def test_augment_prompt_no_static():
    assert augment_prompt("", ["foo"], limit=1) == "On screen: foo."
