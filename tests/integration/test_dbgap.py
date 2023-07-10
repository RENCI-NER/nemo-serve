"""
This test file tests annotations by using multiple NER tools against a source
of data that is of immediate use to [Dug](https://github.com/helxplatform/dug):
data dictionaries from dbGaP. The code to do this is closely related to that
used by [`bin/get_dbgap_data_dicts.py`](https://github.com/helxplatform/dug/blob/4741844d51e18cd59a64f7537aa515690239da49/bin/get_dbgap_data_dicts.py),
except that this code is simpler: we only download dbGaP dictionaries of
interest directly.

As part of this test, we write two files to tests/integration/data/test_dbgap:
- The data dict filename (e.g. `phs000810.v1.pht004715.v1.HCHS_SOL_Cohort_Subject_Phenotypes.data_dict.xml`).
- A custom JSON format which consists of a list of PubAnnotation entries, one for each field in the data dictionary
  and tracks for each methodology that we are using, with a separate track for normalized identifiers where possible.

This is an integration test: it will use two API endpoints that can be configured
by environmental variables:
- The Nemo-Serve URL, which defaults to https://med-nemo.apps.renci.org/
- The SAPBERT URL, which defaults to https://med-nemo-sapbert.apps.renci.org/

You can run this individual test by running `pytest tests/integration/test_dbgap.py`.
"""

import json
import logging
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

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

# Configuration: the `/get_normalized_nodes` endpoint on a Node Normalization instance to use.
NODE_NORM_ENDPOINT = os.getenv('NODE_NORM_ENDPOINT', 'https://nodenormalization-sri.renci.org/1.3/get_normalized_nodes')

# Where should these output files be written out?
OUTPUT_DIR = "tests/integration/data/test_dbgap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Which dbGaP data dictionaries should we test? This should be a URL that points directly to a data_dict.xml file.
DBGAP_DATA_DICTS_TO_TEST = [
    'https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs000810/phs000810.v1.p1/pheno_variable_summaries/phs000810.v1.pht004715.v1.HCHS_SOL_Cohort_Subject_Phenotypes.data_dict.xml'
]

logging.basicConfig(level=logging.INFO)

# We parameterize our test using the list of PubMed IDs to test -- this method will be
# run once for each identifier.
@pytest.mark.parametrize('dbgap_data_dict_url', DBGAP_DATA_DICTS_TO_TEST)
def test_with_pubannotator(dbgap_data_dict_url):
    logging.info(f"Downloading {dbgap_data_dict_url}")

    # Download the data dictionary.
    data_dict_response = requests.get(dbgap_data_dict_url)
    assert data_dict_response.status_code == 200
    data_dict_xml = data_dict_response.text
    tree = ET.fromstring(data_dict_xml)
    data_table = tree

    dbgap_table_id = data_table.get('id')
    dbgap_study_id = data_table.get('study_id')
    dbgap_date_created = data_table.get('date_created')
    dbgap_description = data_table.find('description').text

    print(f"dbGaP data dictionary {dbgap_data_dict_url}")
    print(f" - Table: {dbgap_table_id}, Study: {dbgap_study_id}, Date created: {dbgap_date_created}")
    print(f" - Description: {dbgap_description}")

    for variable in data_table.findall('variable'):
        var_id = variable.get('id')
        var_name = variable.find('name').text
        var_desc = variable.find('description').text
        # var_type = variable.find('type').text
        values = map(lambda val: val.get('code') + ": " + val.text, variable.findall('value'))

        print(f" - Variable {var_id} ({var_name}): {var_desc}")
        for value in values:
            print(f"   - Value: {value}")

    assert False


    # Log what we're doing with this result.
    logging.info(f"Target: {pubannotator['target']} ({pubannotator['sourcedb']}:{pubannotator['sourceid']}) [{pubannotator.get('source_url', '(none)')}]")
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
        denotation['mesh'] = f"MESH:{first_result['curie']}"
        count_sapbert_annotations += 1
        # This is fine for PubAnnotator format (I think?), but PubAnnotator editors
        # don't render this.
        # denotation['label'] = result[0]
        track_sapbert.append(
            denotation
        )

    assert count_sapbert_annotations > 0, f"No SAPBERT annotations found for {pubmed_id}, given these BioMegatron annotations: {track_token_classification}"

    # Normalize nodes.
    node_norm_tracks = []
    for sapbert_denot in track_sapbert:
        # Try to normalize the MeSH.
        mesh = sapbert_denot['mesh']

        response = requests.get(NODE_NORM_ENDPOINT, {
            'curie': mesh,
            'conflate': 'true'
        })
        if not response.ok:
            pass
        else:
            result = response.json().get(mesh, {})
            if result:
                normalized_id = result.get('id', {})
                normalized_identifier = normalized_id.get('identifier')

                if not normalized_identifier:
                    pass
                else:
                    nn_denot = sapbert_denot.copy()
                    nn_denot['obj'] = normalized_identifier
                    label = normalized_id.get('label')
                    if label:
                        nn_denot['obj'] = nn_denot['obj'] + " (" + label + ")"
                    node_norm_tracks.append(nn_denot)

    # Write the annotations from Nemo-Serve and SAPBERT into the output file as separate tracks.
    annotated['tracks'] = [{
        'project': NEMOSERVE_ANNOTATE_ENDPOINT,
        'denotations': track_token_classification
    }, {
        'project': SAPBERT_ANNOTATE_ENDPOINT,
        'denotations': track_sapbert
    }, {
        'project': NODE_NORM_ENDPOINT,
        'denotations': node_norm_tracks
    }]
    del annotated['denotations']

    # Write this out to an output file.
    filename = re.sub(r'[^A-Za-z0-9_]', '_', pubmed_id) + ".json"
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(annotated, f, sort_keys=True, indent=2)

