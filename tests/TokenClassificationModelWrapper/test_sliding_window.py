"""
A basic test for ModelSingleton.
"""

from src.ModelSingleton import TokenClassificationModelWrapper
from unittest.mock import Mock

def test_basic():
    tcmw = Mock(spec=TokenClassificationModelWrapper)

    lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut " \
                  "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris " \
                  "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit " \
                  "esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, " \
                  "sunt in culpa qui officia deserunt mollit anim id est laborum. "

    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, lorem_ipsum, 2)) == [
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et '
        'dolore magna aliqua.',
        'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.',
        'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.',
        'Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est '
        'laborum.'
    ]

    assert list(TokenClassificationModelWrapper.sliding_window(tcmw, lorem_ipsum, 50)) == [
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et '
        'dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex '
        'ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat '
        'nulla pariatur.',
        'Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est '
        'laborum.'
    ]
