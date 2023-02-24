"""
This test file tests annotations by using the BioC API (part of bionlp) as
described in https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/

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
    #'PMID:7837719',         # https://pubmed.ncbi.nlm.nih.gov/7837719/
    # Korn et al (2021) COVID-KOP: integrating emerging COVID-19 data with the ROBOKOP database
    #'PMID:33175089',        # https://pubmed.ncbi.nlm.nih.gov/33175089/
    #'PMC7890668'            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7890668/

    # Articles from NIH HEAL publication website (https://heal.nih.gov/research/publications)
    'PMID:33460838'
]

logging.basicConfig(level=logging.INFO)

# Helper methods
def get_text_from_pmid(pmid):
    """
    Download the text from a PubMed abstract by PMID.
    :param pmid: PMID to look up.
    :return: Text to process.
    """

    url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode'

    # Download the PubAnnotator
    bioc_json_response = requests.get(url)
    assert bioc_json_response.status_code == 200
    bioc_json = bioc_json_response.json()

    chunks = []
    for document in bioc_json['documents']:
        for passage in document['passages']:
            chunks.append(passage['text'])
    text = "\n\n".join(chunks)

    # Log what we're doing with this result.
    logging.info(f"Target: PubMed ID {pmid}")
    logging.info(f" - Text [{len(text)}]: {text}")

    return text

def get_text_from_pmcid(pmcid):
    """
    Download the text from a PubMed abstract by PMCID.
    :param pmcid: PMCID to look up.
    :return: Text to process.
    """

    url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode'

    # Download the PubAnnotator
    bioc_json_response = requests.get(url)
    assert bioc_json_response.status_code == 200
    bioc_json = bioc_json_response.json()

    chunks = []
    for document in bioc_json['documents']:
        for passage in document['passages']:
            chunks.append(passage['text'])
    text = "\n\n".join(chunks)

    # Log what we're doing with this result.
    logging.info(f"Target: PubMedCentral ID {pmcid}")
    logging.info(f" - Text [{len(text)}]: {text}")

    return text

# We parameterize our test using the list of PubMed IDs to test -- this method will be
# run once for each identifier.
@pytest.mark.parametrize('pubmed_id', PUBMED_IDS_TO_TEST)
def test_with_pubannotator(pubmed_id):
    # Figure out the PubAnnotator URL where the PubMed ID can be found.
    if pubmed_id.startswith('PMID:'):
        text = get_text_from_pmid(pubmed_id[5:])
    elif pubmed_id.startswith('PMC'):
        text = get_text_from_pmcid(pubmed_id)
    else:
        pytest.fail(f"Could not get text for identifier {pubmed_id}")
        return

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
    count_sapbert_annotations = 0
    track_sapbert = []
    track_token_classification = annotated['denotations']
    for token in track_token_classification:
        text = token['text']

        assert text, f"Token {token} does not have any text!"

        logging.debug(f"Querying SAPBERT with {token['text']}")
        request = {
            "text": token['text'],
            "model_name": SAPBERT_MODEL_NAME
        }
        response = requests.post(SAPBERT_ANNOTATE_ENDPOINT, json=request)
        logging.debug(f"Response from SAPBERT: {response.content}")
        assert response.status_code == 200

        result = response.json()
        assert result, f"Could not annotate text {token['text']} in Sapbert: {response}"
        first_result = result[0]

        denotation = dict(token)
        denotation['obj'] = f"MESH:{first_result['curie']} ({first_result['label']}, score: {first_result['distance_score']})"
        count_sapbert_annotations += 1
        # This is fine for PubAnnotator format (I think?), but PubAnnotator editors
        # don't render this.
        # denotation['label'] = result[0]
        track_sapbert.append(
            denotation
        )

    assert count_sapbert_annotations > 0, f"No SAPBERT annotations found for {pubmed_id}, given these BioMegatron annotations: {track_token_classification}"

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

