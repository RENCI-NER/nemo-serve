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
SAPBERT_URL = os.getenv('SAPBERT_URL', 'https://babel-sapbert.apps.renci.org/')
SAPBERT_ANNOTATE_ENDPOINT = urllib.parse.urljoin(SAPBERT_URL, '/annotate/')
SAPBERT_MODEL_NAME = "sapbert"

# Configuration: the `/get_normalized_nodes` endpoint on a Node Normalization instance to use.
NODE_NORM_ENDPOINT = os.getenv('NODE_NORM_ENDPOINT', 'https://nodenormalization-sri.renci.org/1.3/get_normalized_nodes')

# Configuration: the Monarch SciGraph endpoint.
MONARCH_SCIGRAPH_URL = 'https://api.monarchinitiative.org/api/nlp/annotate/entities?min_length=4&longest_only=false&include_abbreviation=false&include_acronym=false&include_numbers=false&content='

# Where should these output files be written out?
OUTPUT_DIR = "tests/integration/data/test_dbgap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Which dbGaP data dictionaries should we test? This should be a URL that points directly to a data_dict.xml file.
DBGAP_DATA_DICTS_TO_TEST = [
    'https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs000810/phs000810.v1.p1/pheno_variable_summaries/phs000810.v1.pht004715.v1.HCHS_SOL_Cohort_Subject_Phenotypes.data_dict.xml'
]

logging.basicConfig(level=logging.INFO)


def annotate_variable_using_babel_nemoserve(var_id, var_name, desc, permissible_values):
    """
    Annotate a variable using the Babel/NemoServe system we're developing.

    :param var_id: The variable identifier.
    :param var_name: The variable name.
    :param desc: The variable description.
    :param permissible_values: A list of permissible values as strings for this variable,
        where each string is in the format `value: description`.
    :return:
    """
    annotations = []

    # Make a request to Nemo-Serve to annotate all the text: variable name, description, values.
    text = var_name + " " + desc + " " + " ".join(permissible_values)
    request = {
        "text": text,
        "model_name": NEMOSERVE_MODEL_NAME
    }
    logging.debug(f"Request to {NEMOSERVE_MODEL_NAME}: {request}")
    response = requests.post(NEMOSERVE_ANNOTATE_ENDPOINT, json=request)
    logging.debug(f"Response from {NEMOSERVE_MODEL_NAME}: {response.content}")
    if response.status_code != 403:
        logging.error(f"Querying {NEMOSERVE_MODEL_NAME} returned {response.status_code} ({response.content}): {text}")
        return []
    annotated = response.json()
    logging.info(f" - Nemo result: {annotated}")

    # For each annotation, query it with SAPBERT.
    count_sapbert_annotations = 0
    track_token_classification = annotated['denotations']

    # In addition to the text that SAPBERT found, try to find the entire variable name and each value
    # as separate SAPBERT terms.
    track_token_classification.append({'text': var_name})
    for v in permissible_values:
        track_token_classification.append({'text': v.split('\\s*:\\s*')[1]})

    for token in track_token_classification:
        text = token['text']

        # Determine if there is a Biolink type for this token.
        bl_type = ''
        if 'obj' in token and token['obj'].startswith('biolink:'):
            bl_type = token['obj'][8:]

        assert text, f"Token {token} does not have any text!"

        logging.debug(f"Querying SAPBERT with {text}")
        request = {
            "text": text,
            "model_name": SAPBERT_MODEL_NAME,
            "args": {
                "bl_type": bl_type
            }
        }
        logging.debug(f"Request to {SAPBERT_MODEL_NAME}: {request}")
        response = requests.post(SAPBERT_ANNOTATE_ENDPOINT, json=request)
        logging.debug(f"Response from {SAPBERT_MODEL_NAME}: {response.content}")
        if not response.status_code == 200:
            logging.error(f"Server error from SAPBERT for text '{text}': {response}")
            continue

        result = response.json()
        if len(result) == 0:
            logging.info(f"Could not annotate text {token['text']} in Sapbert: {response}, {response.content}")
            continue

        first_result = result[0]

        denotation = dict(token)
        denotation['text'] = token['text']
        denotation['obj'] = f"{first_result['curie']} ({first_result['name']}, score: {first_result['score']})"
        denotation['name'] = first_result['name']
        denotation['score'] = first_result['score']
        denotation['id'] = f"{first_result['curie']}"

        count_sapbert_annotations += 1
        # This is fine for PubAnnotator format (I think?), but PubAnnotator editors
        # don't render this.
        # denotation['label'] = result[0]
        annotations.append(
            denotation
        )

    # Normalize nodes.
    for annot in annotations:
        # Try to normalize the ID.
        mesh = annot['id']

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
                normalized_label = normalized_id.get('label', '')

                if not normalized_identifier:
                    pass
                else:
                    annot['nn_id'] = normalized_identifier
                    annot['nn_label'] = normalized_label

    return annotations


def annotate_variable_using_scigraph(var_id, var_name, desc, permissible_values):
    """
    Annotate a variable using SciGraph.

    :param var_id: The variable identifier.
    :param var_name: The variable name.
    :param desc: The variable description.
    :param permissible_values: A list of permissible values as strings for this variable,
        where each string is in the format `value: description`.
    :return:
    """
    annotations = []

    # Make a request to Monarch SciGraph to annotate all the text: variable name, description, values.
    text: str = var_name + " " + desc + " " + " ".join(permissible_values)
    request_url = MONARCH_SCIGRAPH_URL + urllib.parse.quote(text)
    logging.debug(f"Request to SciGraph: {request_url}")
    response = requests.post(request_url)
    logging.debug(f"Response from SciGraph: {response.content}")
    assert response.status_code == 200
    annotated = response.json()
    logging.info(f" - SciGraph result: {json.dumps(annotated)}")

    for span in annotated['spans']:
        index_start = span['start']
        index_end = span['end']
        text = span['text']
        tokens = span['token']

        for token in tokens:
            token_id = token['id']
            token_category = token['category']
            token_terms = '|'.join(token['terms'])

            denotation = dict(token)
            denotation['text'] = text
            denotation['name'] = token_terms
            denotation['id'] = token_id
            denotation['category'] = token_category
            denotation['obj'] = f"{token_id} ({token_category}: {token_terms})"

            annotations.append(denotation)

    # Normalize identifiers.
    for annot in annotations:
        # Try to normalize the ID.
        mesh = annot['id']

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
                normalized_label = normalized_id.get('label', '')

                if not normalized_identifier:
                    pass
                else:
                    annot['nn_id'] = normalized_identifier
                    annot['nn_label'] = normalized_label

    return annotations

# We parameterize our test using the list of PubMed IDs to test -- this method will be
# run once for each identifier.
@pytest.mark.parametrize('dbgap_data_dict_url', DBGAP_DATA_DICTS_TO_TEST)
def test_with_dbgap(dbgap_data_dict_url):
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

        print(f"   - Variable {var_id} ({var_name}): {var_desc}")
        for value in values:
            print(f"     - Value: {value}")

        annotations = annotate_variable_using_babel_nemoserve(var_id, var_name, var_desc, values)
        # annotations = annotate_variable_using_scigraph(var_id, var_name, var_desc, values)
        for annotation in annotations:
            nn_id_str = ""
            # Not needed on Babel-SAPBERT, since entries are pre-normalized.
            if 'nn_id' in annotation:
                nn_id_str = f" ({annotation['nn_id']} \"{annotation['nn_label']}\")"
            print(f"     - Annotated \"{annotation['text']}\" to {annotation['id']}{nn_id_str}: {annotation['obj']}")
        print()
