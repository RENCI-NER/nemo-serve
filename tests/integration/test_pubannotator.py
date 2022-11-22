"""
This test file tests annotations by using the PubAnnotator website's API.
It does this by querying a list of PubMed IDs (in the PMID:... format) and
PubMed Central IDs (in the PMC:... format) from the PubAnnotator website's
API, running the text through BioMegatron, and then running each annotation
through SAPBERT to carry out entity linking. This is then written out into
the tests/integration/data/test_pubannotator folder for further comparison
in the PubAnnotator format, allowing them to be viewed in a PubAnnotator editor
like TextAE (https://textae.pubannotation.org/editor.html?mode=edit).

In this version of this test, we only test that we get one or more annotations
from Nemo-Serve -- nothing else is actually tested. In the future, we might
want to:
1. Whether Nemo-Serve can find the same annotations as the PubAnnotator website has.
2. Build some sort of active-learning tool that encapsulates this functionality with
   a human-annotation tool that generates annotated documents that can be used for
   further BioMegatron testing.

This is an integration test: it will use two API endpoints that can be configured
by environmental variables:
- The Nemo-Serve URL, which defaults to https://med-nemo.apps.renci.org/
- The SAPBERT URL, which defaults to https://med-nemo-sapbert.apps.renci.org/
"""

import json
import logging
import os
import re
import urllib.parse

import requests
import pytest

# Configuration: get the Nemo-Serve URL and figure out the annotate path.
NEMOSERVE_URL = os.getenv('NEMOSERVE_URL', 'https://med-nemo.apps.renci.org/')
NEMOSERVE_ANNOTATE_ENDPOINT = urllib.parse.urljoin(NEMOSERVE_URL, '/annotate/')
NEMOSERVE_MODEL_NAME = "token_classification"

# Configuration: get the SAPBERT URL and figure out the annotate path.
SAPBERT_URL = os.getenv('SAPBERT_URL', 'https://med-nemo-sapbert.apps.renci.org/')
SAPBERT_ANNOTATE_ENDPOINT = urllib.parse.urljoin(SAPBERT_URL, '/annotate/')
SAPBERT_MODEL_NAME = "sapbert"

# Where should these output files be written out?
OUTPUT_DIR = "tests/integration/data/test_pubannotator"

# Which PubMed IDs (starting with 'PMID:') or PubMed Central IDs (starting with
# 'PMC...') should be downloaded from PubAnnotator?
PUBMED_IDS_TO_TEST = [
    'PMID:7837719',         # https://pubmed.ncbi.nlm.nih.gov/7837719/
    # Korn et al (2021) COVID-KOP: integrating emerging COVID-19 data with the ROBOKOP database
    'PMID:33175089',        # https://pubmed.ncbi.nlm.nih.gov/33175089/
    'PMC7890668'            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7890668/
]

logging.basicConfig(level=logging.INFO)

# We parameterize our test using the list of PubMed IDs to test -- this method will be
# run once for each identifier.
@pytest.mark.parametrize('pubmed_id', PUBMED_IDS_TO_TEST)
def test_with_pubannotator(pubmed_id):
    # Figure out the PubAnnotator URL where the PubMed ID can be found.
    if pubmed_id.startswith('PMID:'):
        url = f'https://pubannotation.org/docs/sourcedb/PubMed/sourceid/{pubmed_id[5:]}/annotations.json'
    elif pubmed_id.startswith('PMC'):
        url = f'https://pubannotation.org/docs/sourcedb/PMC/sourceid/{pubmed_id[3:]}/annotations.json'
    else:
        pytest.fail(f"Could not get URL for identifier {pubmed_id}")
        return

    # Download the PubAnnotator
    pubannotator_response = requests.get(url)
    assert pubannotator_response.status_code == 200
    pubannotator = pubannotator_response.json()

    # Note that we ignore the annotations from PubAnnotator -- these could be compared to our
    # output later!

    # Log what we're doing with this result.
    logging.info(f"Target: {pubannotator['target']} ({pubannotator['sourcedb']}:{pubannotator['sourceid']}) [{pubannotator['source_url']}]")
    text = pubannotator['text']
    logging.info(f" - Text [{len(text)}]: {text}")

    # Make a request to Nemo-Serve to annotate this text.
    request = {
        "text": text,
        "model_name": NEMOSERVE_MODEL_NAME
    }
    logging.debug(f"Request: {request}")
    response = requests.post(NEMOSERVE_ANNOTATE_ENDPOINT, json=request)
    logging.debug(f"Response: {response.content}")
    assert response.status_code == 200
    annotated = response.json()
    logging.info(f" - Nemo result: {annotated}")

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
            "model_name": SAPBERT_MODEL_NAME
        }
        response = requests.post(SAPBERT_ANNOTATE_ENDPOINT, json=request)
        logging.debug(f"Response from SAPBERT: {response.content}")
        assert response.status_code == 200

        result = response.json()
        if not result:
            logging.warning(f"Could not annotate text {token['text']} in Sapbert: {response}")
            continue

        denotation = dict(token)
        denotation['obj'] = f"MESH:{result[1]} ({result[0]})"
        # This is fine for PubAnnotator format (I think?), but PubAnnotator editors
        # don't render this.
        # denotation['label'] = result[0]
        track_sapbert.append(
            denotation
        )

    # Write the annotations from Nemo-Serve and SAPBERT into the output file as separate tracks.
    annotated['tracks'] = [{
        'project': NEMOSERVE_ANNOTATE_ENDPOINT,
        'denotations': track_token_classification
    }, {
        'project': SAPBERT_ANNOTATE_ENDPOINT,
        'denotations': track_sapbert
    }]
    del annotated['denotations']

    # Write this out to an output file.
    filename = re.sub(r'[^A-Za-z0-9_]', '_', pubmed_id) + ".json"
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(annotated, f, sort_keys=True, indent=2)

