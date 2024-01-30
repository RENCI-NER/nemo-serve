#!/usr/bin/env python3

import csv
import json
import sys
import requests
import logging

from requests.adapters import HTTPAdapter, Retry

# Eventually this will be a fully functional annotation tool, but for now, I'm going to use it for a specific use case.
logging.basicConfig(level=logging.INFO)
CONCEPT_RESOLVER_URL = 'https://concept-resolver.137.120.31.102.nip.io/lookup?string='
MAX_RETRIES = Retry(total=5,
                    backoff_factor=0.1)

# Set up a Requests session with retries.
session = requests.Session()

session.mount('http://', HTTPAdapter(max_retries=MAX_RETRIES))
session.mount('https://', HTTPAdapter(max_retries=MAX_RETRIES))

# Read CSV file from argv[1] using the DictReader
if len(sys.argv) < 1:
    raise RuntimeError(f"One argument needed: a CSV file to process")

with open(sys.argv[1], 'r') as input_file:
    input_data = csv.DictReader(input_file)
    input_fields = list(input_data.fieldnames)

    # Add some fields for concept_resolver results.
    input_fields.extend([
        'concept_resolver_curie',
        'concept_resolver_label',
        'concept_resolver_types',
        'concept_resolver_score',
        'concept_resolver_error'
    ])

    # Write the CSV file out again using the DictWriter
    output_data = csv.DictWriter(sys.stdout, fieldnames=input_fields)
    output_data.writeheader()

    count_rows = 0
    for input_row in input_data:
        count_rows += 1

        row = dict(input_row)
        text = row['Text']

        logging.info(f"Processing row {count_rows} ('{text}')")

        response = session.get(CONCEPT_RESOLVER_URL + text)
        if response.status_code == 200:
            results = response.json()

            if len(results) > 0:
                concept_data = results[0]
                row['concept_resolver_curie'] = concept_data.get('curie', '')
                row['concept_resolver_label'] = concept_data.get('label', '')
                row['concept_resolver_score'] = concept_data.get('score', '')

                types = concept_data.get('types', [])
                if len(types) > 0:
                    row['concept_resolver_types'] = str([types[0]])
                else:
                    row['concept_resolver_types'] = str([])
                    row['concept_resolver_error'] = 'ERR no types'

            logging.info(f"Annotated '{text}' in row {count_rows}: {json.dumps(results, indent=2, sort_keys=True)}")
        else:
            row['concept_resolver_error'] = f"ERR {response.status_code}"

        output_data.writerow(row)