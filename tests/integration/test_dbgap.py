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

You can run all tests in this file by running `pytest tests/integration/test_dbgap.py` or running individual
test in this file by using -k command line option to select tests to run based on their name,
for example, `pytest tests/integration/test_dbgap.py -k 'test_download_dbgap_data_dict'`.
"""

import json
import logging
import os
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import csv
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
NODE_NORM_ENDPOINT = os.getenv('NODE_NORM_ENDPOINT', 'https://nodenormalization-sri.renci.org/get_normalized_nodes')

# Configuration: the Monarch SciGraph endpoint.
MONARCH_SCIGRAPH_URL = 'https://api.monarchinitiative.org/api/nlp/annotate/entities?min_length=4&longest_only=false&include_abbreviation=false&include_acronym=false&include_numbers=false&content='

# Configuration: NameRes
NAMERES_URL = 'http://name-resolution-sri.renci.org/lookup?offset=0&limit=10&string='

# Where should these output files be written out?
OUTPUT_DIR = "tests/integration/data/test_dbgap"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DBGAP_DATA_DICT_FILE = os.path.join(OUTPUT_DIR, "dbgap_data_dict.xml")
OUTPUT_SAPBERT_ANNOTATION_FILE = os.path.join(OUTPUT_DIR,
                                              "sapbert_annot_output.txt")
OUTPUT_SCIGRAPH_ANNOTATION_FILE = os.path.join(OUTPUT_DIR,
                                               "scigraph_annot_output.txt")
OUTPUT_NAMERES_ANNOTATION_FILE = os.path.join(OUTPUT_DIR,
                                              "nameres_annot_output.txt")
OUTPUT_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "annotation_summary_output.csv")
# Which dbGaP data dictionaries should we test? This should be a URL that points directly to a data_dict.xml file.
DBGAP_DATA_DICTS_TO_TEST = [
    'https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs000810/phs000810.v1.p1/pheno_variable_summaries/phs000810.v1.pht004715.v1.HCHS_SOL_Cohort_Subject_Phenotypes.data_dict.xml',
    'https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs001387/phs001387.v3.p1/pheno_variable_summaries/phs001387.v3.pht008970.v2.TOPMed_WGS_THRV_Subject_Phenotypes.data_dict.xml',
]

logging.basicConfig(level=logging.INFO)

def make_annotation_text(var_name, desc, permissible_values):
    "Return a string for annotation"
    text: str = var_name + " " + desc + " " + " ".join(permissible_values)
    return text

def annotate_variable_using_babel_nemoserve(var_name, desc, permissible_values, method='sapbert'):
    """
    Annotate a variable using the Babel/NemoServe system we're developing with a default method sapbert,
    or with NameRes method that can be specified in the method input parameter.
    :param var_name: The variable name.
    :param desc: The variable description.
    :param permissible_values: A list of permissible values as strings for this variable,
        where each string is in the format `value: description`.
    :param method: sapbert or nameres
    :return:
    """
    assert method == 'sapbert' or method == 'nameres'

    annotations = []

    # Make a request to Nemo-Serve to annotate all the text: variable name, description, values.
    text = make_annotation_text(var_name, desc, permissible_values)
    request = {
        "text": text,
        "model_name": NEMOSERVE_MODEL_NAME
    }
    logging.debug(f"Request to {NEMOSERVE_MODEL_NAME}: {request}")
    response = requests.post(NEMOSERVE_ANNOTATE_ENDPOINT, json=request)
    logging.debug(f"Response from {NEMOSERVE_MODEL_NAME}: {response.content}")
    assert response.status_code == 200, f'{response.status_code} ({response.content}): {text}'

    annotated = response.json()
    logging.info(f" - Nemo result: {annotated}")

    # For each annotation, query it with specified method service.
    count_annotations = 0
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
        bl_prefix = 'biolink:'
        if 'obj' in token and token['obj'].startswith(bl_prefix):
            bl_type = token['obj'][len(bl_prefix):]

        assert text, f"Token {token} does not have any text!"

        logging.debug(f"Querying {method} with {text}")
        if method == 'sapbert':
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
        elif method == "nameres":
            nameres_query = NAMERES_URL + urllib.parse.quote(text)
            if bl_type:
                nameres_query = nameres_query + '&biolink_type=' + bl_type
            logging.debug(f"Request to NameRes: {nameres_query}")
            response = requests.post(nameres_query)
            logging.debug(f"Response from nameres: {response.content}")
            if not response.status_code == 200:
                logging.error(f"Server error from NameRes for text '{text}': {response}")
                continue

        result = response.json()
        if len(result) == 0:
            logging.info(f"Could not annotate text {token['text']} in Sapbert: {response}, {response.content}")
            continue

        first_result = result[0]

        denotation = dict(token)
        denotation['text'] = text
        if method == 'sapbert':
            denotation['score'] = first_result['score']
            denotation['name'] = first_result['name']
            denotation['obj'] = f"{first_result['curie']} ({first_result['name']}, score: {first_result['score']})"
        else:  # nameres
            denotation['name'] = first_result['label']
            denotation['obj'] = f"{first_result['curie']} ({first_result['types'][0]}: {first_result['label']})"
        denotation['id'] = f"{first_result['curie']}"

        # These should already be normalized. So let's set nn_id and nn_label.
        denotation['nn_id'] = denotation['id']
        denotation['nn_label'] = denotation['name']

        count_annotations += 1
        # This is fine for PubAnnotator format (I think?), but PubAnnotator editors
        # don't render this.
        # denotation['label'] = result[0]
        annotations.append(
            denotation
        )

    return annotations


def annotate_variable_using_scigraph(var_name, desc, permissible_values):
    """
    Annotate a variable using SciGraph.

    :param var_name: The variable name.
    :param desc: The variable description.
    :param permissible_values: A list of permissible values as strings for this variable,
        where each string is in the format `value: description`.
    :return:
    """
    annotations = []

    # Make a request to Monarch SciGraph to annotate all the text: variable name, description, values.
    text = make_annotation_text(var_name, desc, permissible_values)
    request_url = MONARCH_SCIGRAPH_URL + urllib.parse.quote(text)
    logging.debug(f"Request to SciGraph: {request_url}")
    response = requests.post(request_url)
    logging.debug(f"Response from SciGraph: {response.content}")
    assert response.status_code == 200
    annotated = response.json()
    logging.info(f" - SciGraph result: {json.dumps(annotated)}")

    for span in annotated['spans']:
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


def annotate_dbgap_data_dict(method):
    output_file = ''
    if method == 'sapbert':
        output_file = OUTPUT_SAPBERT_ANNOTATION_FILE
    elif method == 'scigraph':
        output_file = OUTPUT_SCIGRAPH_ANNOTATION_FILE
    elif method == 'nameres':
        output_file = OUTPUT_NAMERES_ANNOTATION_FILE
    else:
        assert Exception("input method must be sapbert, scigraph, or nameres.")

    tree = ET.parse(OUTPUT_DBGAP_DATA_DICT_FILE)
    data_table = tree.getroot()

    dbgap_table_id = data_table.get('id')
    dbgap_study_id = data_table.get('study_id')
    dbgap_date_created = data_table.get('date_created')
    dbgap_description = data_table.find('description').text
    assert dbgap_table_id is not None
    assert dbgap_study_id is not None
    assert dbgap_date_created is not None
    assert dbgap_description is not None
    with open(output_file, 'w') as f:
        f.write(f"dbGaP data dictionary {OUTPUT_DBGAP_DATA_DICT_FILE}\n")
        f.write(f" - Table: {dbgap_table_id}, Study: {dbgap_study_id}, Date created: {dbgap_date_created}\n")
        f.write(f" - Description: {dbgap_description}\n")

        for variable in data_table.findall('variable'):
            var_id = variable.get('id')
            var_name = variable.find('name').text
            var_desc = variable.find('description').text
            # var_type = variable.find('type').text
            values = map(lambda val: val.get('code') + ": " + val.text, variable.findall('value'))

            f.write(f"   - Variable {var_id} ({var_name}): {var_desc}\n")
            for value in values:
                f.write(f"     - Value: {value}\n")

            if method == 'sapbert' or method == 'nameres':
                annotations = annotate_variable_using_babel_nemoserve(var_name, var_desc, values,
                                                                      method=method)
            elif method == 'scigraph':
                annotations = annotate_variable_using_scigraph(var_name, var_desc, values)
            else:
                assert Exception("input method must be sapbert, scigraph, or nameres.")

            for annotation in annotations:
                nn_id_str = ""
                # Not needed on Babel-SAPBERT, since entries are pre-normalized.
                if 'nn_id' in annotation:
                    nn_id_str = f" ({annotation['nn_id']} \"{annotation['nn_label']}\")"
                f.write(f"     - Annotated \"{annotation['text']}\" to {annotation['id']}{nn_id_str}: "
                        f"{annotation['obj']}\n")

def annotation_string(annotation):
    "Take an annotation, return a three-part string that represents it"
    return ":".join((
        annotation['nn_id'],
        annotation['nn_label'],
        annotation['text'],))

def run_summary_report():
    """Run all three annotations together, generate a summary report"""

    fieldnames = [
        'dbgap_url',
        'var_id',
        'var_name',
        'dataset_url',
        'var_text',
        'annotation_common',
        'ann_sapbert_adds',
        'ann_nameres_adds',
        'ann_scigraph_adds']

    with open(OUTPUT_SUMMARY_FILE, 'w') as csv_file:
        # Write the header -- we do this once for all the dbGaP data dictionaries we want to test.
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for dbgap_data_dict_url in DBGAP_DATA_DICTS_TO_TEST:
            logging.info(f"Downloading {dbgap_data_dict_url}")

            # Download the data dictionary.
            data_dict_response = requests.get(dbgap_data_dict_url)
            assert data_dict_response.status_code == 200

            data_table = ET.fromstring(data_dict_response.content)
            dbgap_table_id = data_table.get('id')
            dbgap_study_id = data_table.get('study_id')
            dbgap_date_created = data_table.get('date_created')
            dbgap_description = data_table.find('description').text
            assert all([dbgap_table_id, dbgap_study_id, dbgap_date_created,
                        dbgap_description])

            variables = data_table.findall('variable')
            logging.info(f"Annotating {len(list(variables))} variables from {dbgap_data_dict_url}")

            for variable in variables:
                var_id = variable.get('id')
                var_name = variable.find('name').text
                var_desc = variable.find('description').text
                values = map(lambda val: val.get('code') + ": " + val.text,
                             variable.findall('value'))

                logging.info(f"Annotating {var_name}: {var_desc} ({values}) using Biomegatron/SAPBERT")

                # get sets of 3-tuples for all annotations
                sapbert_annotations = annotate_variable_using_babel_nemoserve(
                    var_name, var_desc, values, method='sapbert')
                sapbert_set = {annotation_string(an)
                               for an in sapbert_annotations if 'nn_id' in an}

                logging.info(f"Annotating {var_name}: {var_desc} ({values}) using Biomegatron/NameRes")

                nameres_annotations = annotate_variable_using_babel_nemoserve(
                    var_name, var_desc, values, method='nameres')
                nameres_set = {annotation_string(an)
                               for an in nameres_annotations if 'nn_id' in an}

                logging.info(f"Annotating {var_name}: {var_desc} ({values}) using Monarch Scigraph")

                scigraph_annotations = annotate_variable_using_scigraph(
                    var_name, var_desc, values)
                scigraph_set = {annotation_string(an)
                                for an in scigraph_annotations if 'nn_id' in an}

                # What concepts does everybody agree on?
                annotation_common = sapbert_set & nameres_set & scigraph_set

                output = {
                    'dbgap_url': dbgap_data_dict_url,
                    'var_id': var_id,
                    'var_name': var_name,
                    'dataset_url': '',
                    'var_text': make_annotation_text(var_name, var_desc, values),
                    'annotation_common': ";".join(annotation_common),
                    'ann_sapbert_adds': ";".join(sapbert_set - annotation_common),
                    'ann_nameres_adds': ";".join(nameres_set - annotation_common),
                    'ann_scigraph_adds': ";".join(scigraph_set - annotation_common)
                }

                writer.writerow(output)

@pytest.mark.parametrize('dbgap_data_dict_url', DBGAP_DATA_DICTS_TO_TEST)
def test_download_dbgap_data_dict(dbgap_data_dict_url):
    logging.info(f"Downloading {dbgap_data_dict_url}")

    # Download the data dictionary.
    data_dict_response = requests.get(dbgap_data_dict_url)
    assert data_dict_response.status_code == 200
    with open(OUTPUT_DBGAP_DATA_DICT_FILE, 'w') as f:
        f.write(data_dict_response.text)
    assert os.path.exists(OUTPUT_DBGAP_DATA_DICT_FILE)


def test_sapbert_annotation_with_dbgap():
    annotate_dbgap_data_dict('sapbert')


def test_scigraph_annotation_with_dbgap():
    annotate_dbgap_data_dict('scigraph')


def test_nameres_annotation_with_dbgap():
    annotate_dbgap_data_dict('nameres')

if __name__ == '__main__':
    # If we're calling this directly from the command line, let's run a summary
    # report of all three annotation methods
    run_summary_report()
