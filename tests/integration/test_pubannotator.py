"""
This test file tests annotations against the online PubAnnotator website.
It does this by keeping a list of all the PubMed identifiers with annotations
on that website, including both abstracts and full-text documents, and then:
1. Using PubAnnotator to get full text and annotations for the PMID.
2. Get annotations from Nemo and compare.

This is an integration test: it will test the endpoint you set the
NEMOSERVE_URL environmental variable to, but will default to
https://med-nemo.apps.renci.org/
"""
import logging
import os
import urllib.parse

import requests
import pytest

NEMOSERVE_URL = os.getenv('NEMOSERVE_URL', 'https://med-nemo.apps.renci.org/')
ANNOTATE_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/annotate/')

MAX_TEXT = 100
PUBMED_IDS_TO_TEST = [
    'PMID:7837719',         # https://pubmed.ncbi.nlm.nih.gov/7837719/
    # Korn et al (2021) COVID-KOP: integrating emerging COVID-19 data with the ROBOKOP database
    #'PMID:33175089',        # https://pubmed.ncbi.nlm.nih.gov/33175089/
    #'PMC7890668'            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7890668/
]

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize('pubmed_id', PUBMED_IDS_TO_TEST)
def test_with_pubannotator(pubmed_id):
    if pubmed_id.startswith('PMID:'):
        url = f'https://pubannotation.org/docs/sourcedb/PubMed/sourceid/{pubmed_id[5:]}/annotations.json'
    elif pubmed_id.startswith('PMC'):
        url = f'https://pubannotation.org/docs/sourcedb/PMC/sourceid/{pubmed_id[3:]}/annotations.json'
    else:
        pytest.fail(f"Could not get URL for identifier {pubmed_id}")
        return

    pubannotator_response = requests.get(url)
    assert pubannotator_response.status_code == 200
    pubannotator = pubannotator_response.json()

    logging.info(f"Target: {pubannotator['target']} ({pubannotator['sourcedb']}:{pubannotator['sourceid']}) [{pubannotator['source_url']}]")
    text = pubannotator['text']
    logging.info(f" - Text [{len(text)}]: {text}")

    if len(text) < 10:
        pytest.fail(f"Text for {pubmed_id} is too small to be processed: {text}")
        return

    text = text[:MAX_TEXT]

    request = {
        "text": text,
        "model_name": 'tokenClassificationModel'
    }
    print(f"Request: {request}")
    response = requests.post(ANNOTATE_ENDPOINT, json=request)
    print(f"Response: {response.content}")
    assert response.status_code == 200
    annotated = response.json()
    logging.info(f" - Nemo: {annotated}")
