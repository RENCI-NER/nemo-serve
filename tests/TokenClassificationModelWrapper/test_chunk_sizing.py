"""
Regression tests for the token chunking / sliding-window logic.

Background: an input longer than the model's max_seq_length produced
out-of-range indices in a CUDA kernel, triggering a device-side assert that
permanently corrupted the CUDA context -- every subsequent request then failed
until the pod was restarted. These tests assert the invariant that fixes it:
NO chunk emitted toward the model may reach the model's window size, no matter
how long or how token-dense the input is.

The model tokenizer is faked so these run without the GPU/ML stack: most words
are 1 token, but any word containing a digit expands to 3 tokens (simulating
wordpiece expansion, where words != tokens).
"""

from ModelSingleton import TokenClassificationModelWrapper
from unittest.mock import Mock


class FakeModelTokenizer:
    """Emits WordPiece-style tokens with '##' continuation markers, like the
    real BiomegatronBERT tokenizer: a word with a digit expands to 3 pieces
    (one word-start + two continuations); other words are a single token.
    """
    def text_to_tokens(self, text):
        tokens = []
        for word in text.split():
            if any(c.isdigit() for c in word):
                tokens.extend([word, "##x", "##y"])
            else:
                tokens.append(word)
        return tokens


def _wrapper(window_size):
    """A TokenClassificationModelWrapper with only the pieces chunking needs."""
    w = Mock(spec=TokenClassificationModelWrapper)
    w.window_size = window_size
    w.model = Mock()
    w.model.tokenizer = FakeModelTokenizer()
    # Bind the real (unbound) methods so we exercise production code.
    w._get_token_length = lambda text: \
        TokenClassificationModelWrapper._get_token_length(w, text)
    w._word_token_counts = TokenClassificationModelWrapper._word_token_counts
    w._token_chunks = lambda text, ws: \
        TokenClassificationModelWrapper._token_chunks(w, text, ws)
    w._truncate_to_window = lambda q: \
        TokenClassificationModelWrapper._truncate_to_window(w, q)
    return w


WINDOW = 126  # matches deployed max_seq_length (128) minus [CLS]/[SEP]


def test_short_text_passes_through_unchanged():
    w = _wrapper(WINDOW)
    text = "Does the protocol include tools to screen for withdrawal ?"
    chunks = list(w._token_chunks(text, WINDOW))
    assert chunks == [(len(text.split()), text)]


def test_long_plain_text_never_exceeds_window():
    w = _wrapper(WINDOW)
    text = " ".join(["alpha"] * 2000)
    chunks = list(w._token_chunks(text, WINDOW))
    assert chunks, "expected the long input to be split"
    assert all(tok_count < WINDOW for tok_count, _ in chunks)
    # No words are dropped by chunking itself.
    assert sum(len(chunk.split()) for _, chunk in chunks) == 2000


def test_token_dense_text_never_exceeds_window():
    # Every word has a digit -> 3 tokens each; words-per-chunk must shrink.
    w = _wrapper(WINDOW)
    text = " ".join([f"tok{i}" for i in range(2000)])
    chunks = list(w._token_chunks(text, WINDOW))
    assert all(tok_count < WINDOW for tok_count, _ in chunks)
    assert sum(len(chunk.split()) for _, chunk in chunks) == 2000


def test_input_exactly_at_window_is_split():
    w = _wrapper(WINDOW)
    text = " ".join(["a"] * WINDOW)
    chunks = list(w._token_chunks(text, WINDOW))
    assert all(tok_count < WINDOW for tok_count, _ in chunks)


def test_truncate_guard_clamps_oversized_query():
    # A query that slips through over-window (e.g. one enormous word) must be
    # clamped before it reaches the model.
    w = _wrapper(WINDOW)
    oversized = " ".join(["num1"] * 100)  # 300 tokens
    truncated = w._truncate_to_window(oversized)
    assert w._get_token_length(truncated) <= WINDOW


def test_truncate_fast_path_leaves_short_query_unchanged():
    # Under-window queries must be returned unchanged with a single tokenizer
    # call (no truncation work).
    w = _wrapper(WINDOW)
    query = "the protocol includes opioid withdrawal screening tools"
    assert w._truncate_to_window(query) == query


def test_word_token_counts_maps_continuations_to_owning_word():
    # '##' continuation tokens belong to the preceding word-start token.
    tokenizer = FakeModelTokenizer()
    words = "plain num1 word".split()
    tokens = tokenizer.text_to_tokens(" ".join(words))  # plain=1, num1=3, word=1
    counts = TokenClassificationModelWrapper._word_token_counts(tokens, words)
    assert counts == [1, 3, 1]
