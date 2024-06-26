"""
This test file tests the Nemo-serve API as it currently exists.

This is an integration test: it will test the endpoint you set the
NEMOSERVE_URL environmental variable to, but will default to
https://med-nemo.apps.renci.org/
"""

import os
import urllib.parse

import requests

NEMOSERVE_URL = os.getenv('NEMOSERVE_URL', 'https://med-nemo.apps.renci.org/')
OPENAPI_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/openapi.json')
DOCS_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/docs/')
ANNOTATE_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/annotate/')
MODELS_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/models/')
MODELS_EXPECTED = os.getenv('MODELS_EXPECTED', 'token_classification').split('|')


def test_openapi():
    response = requests.get(OPENAPI_ENDPOINT)
    assert response.status_code == 200
    openapi_response = response.json()
    assert openapi_response['openapi'] == '3.1.0'
    assert openapi_response['paths'].keys() == {'/annotate/', '/models/'}


def test_docs():
    response = requests.get(DOCS_ENDPOINT)
    assert response.status_code == 200

    html = response.content.decode('utf8')
    assert '<div id="swagger-ui">' in html
    assert "url: '/openapi.json'" in html


def test_models():
    response = requests.get(MODELS_ENDPOINT)
    assert response.status_code == 200
    models = response.json()
    assert len(models) > 0

    # At the moment, we only support a single model.
    assert models == MODELS_EXPECTED


def test_annotate():
    request = {
        "text": "Human brains fit inside human skulls. Influenza is caused by a virus.",
        "model_name": MODELS_EXPECTED[0]
    }
    expected_denotations = [
        {
         'obj': 'biolink:NamedThing',
         'span': {'begin': 0,
                  'end': 5},
         'text': 'Human'},
        {
         'obj': 'biolink:AnatomicalEntity',
         'span': {'begin': 6,
                  'end': 12},
         'text': 'brains'},
        {
         'obj': 'biolink:NamedThing',
         'span': {'begin': 24,
                  'end': 29},
         'text': 'human'},
        {
         'obj': 'biolink:AnatomicalEntity',
         'span': {'begin': 30,
                  'end': 36},
         'text': 'skulls'},
        {
         'obj': 'biolink:Disease',
         'span': {'begin': 38,
                  'end': 47},
         'text': 'Influenza'},
        {
         'obj': 'biolink:OrganismTaxon',
         'span': {'begin': 63,
                  'end': 68},
         'text': 'virus'}
        ]

    response = requests.post(ANNOTATE_ENDPOINT, json=request)
    assert response.status_code == 200
    annotated = response.json()
    assert annotated['text'] == request['text']
    assert 'denotations' in annotated
    assert len(annotated['denotations']) == len(expected_denotations)
    for den, exp_den in zip(annotated['denotations'], expected_denotations):
        assert den['obj'] == exp_den['obj']
        assert den['span'] == exp_den['span']
        assert den['text'] == exp_den['text']
