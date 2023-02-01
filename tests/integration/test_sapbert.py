"""
This test file tests SAPBert via the Nemo-serve API.

This is an integration test: it will test the endpoint you set the
NEMOSAPBERT_URL environmental variable to, but will default to
https://med-nemo-sapbert.apps.renci.org/
"""

import os
import urllib.parse

import requests

NEMOSAPBERT_URL = os.getenv('NEMOSAPBERT_URL', 'https://med-nemo-sapbert.apps.renci.org/')
ANNOTATE_ENDPOINT = urllib.parse.urljoin(NEMOSAPBERT_URL, '/annotate/')


def test_identify():
    """
    Test whether SAPBert can identify some concepts.
    """
    test_cases = [
        ['human brain', {'label': 'Brain', 'curie': 'D001921'}],
        ['human skulls', {'label': 'Skull', 'curie': 'D012886'}],
        ['influenza', {'label': 'Influenza, Human', 'curie': 'D007251'}],
        ['virus', {'label': 'Viruses', 'curie': 'D014780'}]
    ]

    for test_case in test_cases:
        query_text = test_case[0]
        expected_result = test_case[1]

        # Retrieve a single term for the input.
        response = requests.post(ANNOTATE_ENDPOINT, json={
            "text": query_text,
            "model_name": "sapbert",
            "count": 1
        })
        assert response.status_code == 200
        annotated = response.json()
        assert len(annotated) == 1
        first_result = annotated[0]
        assert first_result['label'] == expected_result['label']
        assert first_result['curie'] == expected_result['curie']
