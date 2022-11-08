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
import json
import logging
import os
import re
import urllib.parse

import requests
import pytest

NEMOSERVE_URL = os.getenv('NEMOSERVE_URL', 'https://med-nemo.apps.renci.org/')
TOKEN_ANNOTATE_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/annotate/')

SAPBERT_URL = os.getenv('SAPBERT_URL', 'https://med-nemo-sapbert.apps.renci.org/')
SAPBERT_ANNOTATE_ENDPOINT = urllib.parse.urljoin(SAPBERT_URL, '/annotate/')

OUTPUT_DIR = "tests/integration/data/test_pubannotator"

MIN_TEXT = 10
MAX_TEXT = 100
PUBMED_IDS_TO_TEST = [
    'PMID:7837719',         # https://pubmed.ncbi.nlm.nih.gov/7837719/
    # Korn et al (2021) COVID-KOP: integrating emerging COVID-19 data with the ROBOKOP database
    'PMID:33175089',        # https://pubmed.ncbi.nlm.nih.gov/33175089/
    'PMC7890668'            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7890668/
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

    if len(text) < MIN_TEXT:
        pytest.fail(f"Text for {pubmed_id} is too small to be processed: {text}")
        return

    # TODO: use /models to get model_name

    request = {
        "text": text,
        "model_name": 'token_classification'
    }
    print(f"Request: {request}")
    response = requests.post(TOKEN_ANNOTATE_ENDPOINT, json=request)
    print(f"Response: {response.content}")
    assert response.status_code == 200
    annotated = response.json()
    logging.info(f" - Nemo: {annotated}")

    # For each annotation, query it with SAPBERT.
    track_sapbert = []
    track_token_classification = annotated['denotations']
    for token in track_token_classification:
        text = token['text']

        if not text:
            logging.error(f"Token {token} does not have any text!")
            continue

        logging.debug(f"Querying SAPBERT with {token['text']}")
        request = {
            "text": token['text'],
            "model_name": "sapbert"
        }
        response = requests.post(SAPBERT_ANNOTATE_ENDPOINT, json=request)
        logging.debug(f"Response from SAPBERT: {response.content}")
        assert response.status_code == 200

        result = response.json()
        if not result:
            logging.warning(f"Could not annotate text {token['text']} in Sapbert: {response}")
            continue

        denotation = dict(token)
        denotation['obj'] = "MESH:" + result[1] + " (" + result[0] + ")"
        #denotation['label'] = result[0]
        track_sapbert.append(
            denotation
        )

    annotated['tracks'] = [{
        'project': TOKEN_ANNOTATE_ENDPOINT,
        'denotations': track_token_classification
    }, {
        'project': SAPBERT_ANNOTATE_ENDPOINT,
        'denotations': track_sapbert
    }]
    del annotated['denotations']

    # Write this out an output file.
    filename = re.sub(r'[^A-Za-z0-9_]', '_', pubmed_id) + ".json"
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(annotated, f, sort_keys=True, indent=2)

