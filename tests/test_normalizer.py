from __future__ import annotations

import pytest
from app.ingestion.normalizer import normalize_text


def test_normalize_empty_and_whitespace():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  \n  ") == ""


def test_preserves_policy_ids():
    policy_id = "POL-2024-V1.2"
    assert normalize_text(policy_id) == policy_id

    complex_id = "SEC-004-REV_A (ref: HR-BEN-401k)"
    assert normalize_text(complex_id) == complex_id


def test_preserves_numbers_and_currency():
    text = "The budget is $1,234,567.89 and accuracy was 99.9% (-42.0)."
    assert normalize_text(text) == text


def test_preserves_dates():
    dates_text = "Effective from 2026-10-20 to 10/20/2027 (October 20, 2026)."
    assert normalize_text(dates_text) == dates_text


def test_preserves_punctuation():
    punct_text = "Section § 4.2; item (a/b) [priority #1]: 'yes' / \"no\"."
    assert normalize_text(punct_text) == punct_text


def test_collapses_horizontal_whitespace_within_lines():
    text = "Line with     multiple   spaces  \tand \ttabs."
    assert normalize_text(text) == "Line with multiple spaces and tabs."


def test_normalizes_line_endings_and_paragraphs():
    text = "Paragraph 1\r\n\r\n\r\n\r\nParagraph 2\r\nParagraph 3"
    expected = "Paragraph 1\n\nParagraph 2\nParagraph 3"
    assert normalize_text(text) == expected


def test_removes_zero_width_and_soft_hyphens():
    # \u00ad is soft hyphen, \u200b is zero-width space
    text = "pol\u00adicy doc\u200bument"
    assert normalize_text(text) == "policy document"


def test_normalizes_unicode_spaces():
    # \u00a0 is non-breaking space, \u202f is narrow no-break space
    text = "Policy\u00a0Document\u202fVersion"
    assert normalize_text(text) == "Policy Document Version"


def test_normalizes_ligatures():
    # \ufb01 = fi, \ufb02 = fl
    text = "\ufb01nancial \ufb02ow"
    assert normalize_text(text) == "financial flow"


def test_strips_control_characters():
    text = "Clean\x00 text\x07 with\x1f nulls"
    assert normalize_text(text) == "Clean text with nulls"
