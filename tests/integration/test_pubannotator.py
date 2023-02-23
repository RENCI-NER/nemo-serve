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
    #'PMID:7837719',         # https://pubmed.ncbi.nlm.nih.gov/7837719/
    # Korn et al (2021) COVID-KOP: integrating emerging COVID-19 data with the ROBOKOP database
    #'PMID:33175089',        # https://pubmed.ncbi.nlm.nih.gov/33175089/
    #'PMC7890668'            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7890668/

    # Articles from NIH HEAL publication website (https://heal.nih.gov/research/publications)
    'PMC8630357',
    'PMC8612015',
    'PMC8378649',
    'PMC8759280',
    'PMC8397502',
    'PMC8561170',
    'PMC7954776',
    'PMC8883586',
    'PMC8316251',
    'PMC8279089',
    'PMC8313952',
    'PMC7708799',
    'PMC8217104',
    'PMC8491168',
    'PMC8263807',
    'PMC8852331',
    'PMC8898543',
    'PMC7898779',
    'PMC8197748',
    'PMC8319218',
    'PMC8427378',
    'PMC8384642',
    'PMC7790977',
    'PMC8384644',
    'PMC8380634',
    'PMC8255321',
    'PMC8940666',
    'PMC8570526',
    'PMC8192586',
    'PMC8310396',
    'PMC8568621',
    'PMC8267049',
    'PMC8384640',
    'PMC8561170',
    'PMC7568493',
    'PMC8453131',
    'PMC8196445',
    'PMC8028689',
    'PMC8474938',
    'PMC8929750',
    'PMC8330022',
    'PMC8605598',
    'PMC8812358',
    'PMC8977050',
    'PMC8283009',
    'PMC7852151',
    'PMC8115873',
    'PMC8294463',
    'PMC7855765',
    'PMC7919109',
    'PMC8318873',
    'PMC8762360',
    'PMC8365463',
    'PMC8861425',
    'PMC8476647',
    'PMC8267558',
    'PMC8483646',
    'PMC8363827',
    'PMC8246150',
    'PMC8578417',
    'PMC7804920',
    'PMC7780957',
    'PMC8697722',
    'PMC7765737',
    'PMC8207994',
    'PMC8792317',
    'PMC8083988',
    'PMC8740641',
    'PMC7753951',
    'PMC7873818',
    'PMC8161858',
    'PMC8691659',
    'PMC8530874',
    'PMC7492292',
    'PMC7765737',
    'PMC7312719',
    'PMC8502004',
    'PMC8323399',
    'PMC7988301',
    'PMC7600449',
    'PMC8326284',
    'PMC7329612',
    'PMC7190928',
    'PMC8486152',
    'PMC8776895',
    'PMC7041488',
    'PMC7572566',
    'PMC8556407',
    'PMC8776619',
    'PMC8058295',
    'PMC8049958',
    'PMC8741281',
    'PMC8760468',
    'PMC7819318',
    'PMC8049961',
    'PMC8040486',
    'PMC7230033',
    'PMC8111462',
    'PMC7515905',
    'PMC8720329',
    'PMC7671950',
    'PMC7982354',
    'PMC8026774',
    'PMC7655591',
    'PMC7856001',
    'PMC8193687',
    'PMC7778216',
    'PMC7794656',
    'PMC7816461',
    'PMC8027703',
    'PMC7917437',
    'PMC8009840',
    'PMC8119287',
    'PMC8022197',
    'PMC7965235',
    'PMC8113153',
    'PMC8179724',
    'PMC7985616',
    'PMC8082467',
    'PMC8422285',
    'PMC8192039',
    'PMC8410661',
    'PMC7927322',
    'PMC8388127',
    'PMC8388127',
    'PMC8328003',
    'PMC8190897',
    'PMC8398878',
    'PMC8429200',
    'PMC8603354',
    'PMC8388132',
    'PMC8956128',
    'PMC8407529',
    'PMC8452258',
    'PMC8608747',
    'PMC8490288',
    'PMC8534031',
    'PMC8980111',
    'PMC8547829',
    'PMC8580056',
    'PMC8639162',
    'PMC8314867',
    'PMC8693769',
    'PMC8657553',
    'PMC8098772',
    'PMC8882208',
    'PMC8691745',
    'PMC8397270',
    'PMC8621475',
    'PMC8119312',
    'PMC7356605',
    'PMC6969143',
    'PMC8005450',
    'PMC7978415',
    'PMC8382852',
    'PMC8217081',
    'PMC8019775'
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

