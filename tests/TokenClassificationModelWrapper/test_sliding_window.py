"""
A basic test for ModelSingleton.
"""

from ModelSingleton import TokenClassificationModelWrapper
from unittest.mock import Mock


def test_basic():
    tcmw = Mock(spec=TokenClassificationModelWrapper)

    # In this example text, every sentence has one more word than the previous sentence.
    example = "Word. A sentence. Three word sentence. Four words in sentence. Five words in this sentence."

    # Setting the window_size to 1 combines the first two sentences.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example, 1)) == [
        "Word. A sentence. ",           # 3 words
        "Three word sentence. ",        # 3 words
        "Four words in sentence. ",     # 4 words
        "Five words in this sentence."  # 5 words
    ]

    # Setting the window_size to 2 does not change the output.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example, 2)) == [
        "Word. A sentence. ",            # 3 words
        "Three word sentence. ",         # 3 words
        "Four words in sentence. ",      # 4 words
        "Five words in this sentence."   # 5 words
    ]

    # Setting the window_size to 3 does not change the output.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example, 3)) == [
        "Word. A sentence. ",                # 3 words
        "Three word sentence. ",             # 3 words
        "Four words in sentence. ",          # 4 words
        "Five words in this sentence."       # 5 words
    ]

    # Setting the window_size to 4 does not change the output.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example, 4)) == [
        "Word. A sentence. ",               # 3 words
        "Three word sentence. ",            # 3 words
        "Four words in sentence. ",         # 4 words
        "Five words in this sentence."      # 5 words
    ]

    # In this example, the sentences have decreasing size.
    example_decreasing = "Five words in this sentence. Four words in sentence. Three word sentence. A sentence. Word."

    # Setting the window_size to 1 returns all sentences separately.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example_decreasing, 1)) == [
        "Five words in this sentence. ", "Four words in sentence. ", "Three word sentence. A sentence. ", "Word."
    ]

    # Setting the window size to 2 has no effect.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example_decreasing, 2)) == [
        "Five words in this sentence. ", "Four words in sentence. ", "Three word sentence. A sentence. ", "Word."
    ]

    # But setting the window size to 3 causes the last two sentences to be merged together.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example_decreasing, 3)) == [
        "Five words in this sentence. ",        # 5 words
        "Four words in sentence. ",             # 4 words
        "Three word sentence. A sentence. ",    # 5 words
        "Word."                                 # 1 word
    ]

    # ... and so on.
    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example_decreasing, 4)) == [
        "Five words in this sentence. ",        # 5 words
        "Four words in sentence. ",             # 4 words
        "Three word sentence. A sentence. ",    # 5 words
        "Word."                                 # 1 word
    ]

    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example_decreasing, 5)) == [
        "Five words in this sentence. ",        # 5 words
        "Four words in sentence. ",             # 4 words
        "Three word sentence. A sentence. ",    # 5 words
        "Word."                                 # 1 word
    ]

    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example_decreasing, 6)) == [
        "Five words in this sentence. ",                                # 5 words
        "Four words in sentence. Three word sentence. A sentence. ",    # 9 words
        "Word."                                                         # 1 word
    ]

    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, example_decreasing, 7)) == [
        "Five words in this sentence. Four words in sentence. ",    # 9 words
        "Three word sentence. A sentence. Word."                    # 6 words
    ]