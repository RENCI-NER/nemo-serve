"""
This test file tests SAPBert via the Nemo-serve API.

This is an integration test: it will test the endpoint you set the
NEMOSERVE_URL environmental variable to, but will default to
https://med-nemo-sapbert.apps.renci.org/
"""

import os
import urllib.parse

import requests

NEMOSERVE_URL = os.getenv('NEMOSERVE_URL', 'https://med-nemo-sapbert.apps.renci.org/')
ANNOTATE_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/annotate/')


def test_identify():
    """
    Test whether SAPBert can identify some concepts.
    """
    test_cases = [
        ['human brain', ['Brain', 'D001921']],
        ['human skulls', ['Skull', 'D012886']],
        ['influenza', ['Influenza, Human', 'D007251']],
        ['virus', ['Viruses', 'D014780']]
    ]

    for test_case in test_cases:
        query_text = test_case[0]
        expected_result = test_case[1]

        response = requests.post(ANNOTATE_ENDPOINT, json={
            "text": query_text,
            "model_name": "sapbert"
        })
        assert response.status_code == 200
        annotated = response.json()
        assert annotated == expected_result
