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


def test_openapi():
    response = requests.get(OPENAPI_ENDPOINT)
    assert response.status_code == 200
    openapi_response = response.json()
    assert openapi_response['openapi'] == '3.0.2'
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
    assert models == ['tokenClassificationModel']


def test_annotate():
    request = {
        "text": "Human brains fit inside human skulls. Influenza is caused by a virus.",
        "model_name": "tokenClassificationModel"
    }
    response = requests.post(ANNOTATE_ENDPOINT, json=request)
    assert response.status_code == 200
    annotated = response.json()
    assert annotated == [("Human[B-biolink:NamedThing] brains[B-biolink:GrossAnatomicalStructure] fit[0] inside[0] "
                          "human[B-biolink:NamedThing] skulls[B-biolink:AnatomicalEntity]. "
                          "Influenza[B-biolink:Disease] is[0] caused[0] by[0] a[0] virus[B-biolink:NamedThing].")]
