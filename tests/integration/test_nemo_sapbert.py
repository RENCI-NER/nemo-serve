"""
This test file tests SAPBert via the Nemo-serve API.

This is an integration test: it will test the endpoints you set the
NEMOSERVE_URL and NEMOSAPBERT_URL environmental variable to, but will default to
https://med-nemo.apps.renci.org/ and https://med-nemo-sapbert.apps.renci.org/
"""
import logging
import os
import urllib.parse
import json

import requests

logging.basicConfig(level=logging.INFO)

NEMOSERVE_URL = os.getenv('NEMOSERVE_URL', 'https://med-nemo.apps.renci.org/')
NEMOSAPBERT_URL = os.getenv('NEMOSAPBERT_URL', 'https://med-nemo-sapbert.apps.renci.org/')


def test_nemo_serve():
    """
    Test whether we can use Nemo-serve and Nemo-Sapbert together.
    """
    pubmed_30925593_abstract = "Discovering small molecules as Wnt inhibitors that promote heart regeneration and " \
                               "injury repair There are intense interests in discovering proregenerative medicine " \
                               "leads that can promote cardiac differentiation and regeneration, as well as repair " \
                               "damaged heart tissues. We have combined zebrafish embryo-based screens with " \
                               "cardiomyogenesis assays to discover selective small molecules that modulate heart " \
                               "development and regeneration with minimal adverse effects. Two related compounds with " \
                               "novel structures, named as Cardiomogen 1 and 2 (CDMG1 and CDMG2), were identified for " \
                               "their capacity to promote myocardial hyperplasia through expansion of the cardiac " \
                               "progenitor cell population. We find that Cardiomogen acts as a Wnt inhibitor by " \
                               "targeting β-catenin and reducing Tcf/Lef-mediated transcription in cultured cells. " \
                               "CDMG treatment of amputated zebrafish hearts reduces nuclear β-catenin in injured " \
                               "heart tissue, increases cardiomyocyte (CM) proliferation, and expedites wound " \
                               "healing, thus accelerating cardiac muscle regeneration. Importantly, Cardiomogen can " \
                               "alleviate the functional deterioration of mammalian hearts after myocardial " \
                               "infarction. Injured hearts exposed to CDMG1 display increased newly formed CMs and " \
                               "reduced fibrotic scar tissue, which are in part attributable to the β-catenin " \
                               "reduction. Our findings indicate Cardiomogen as a Wnt inhibitor in enhancing " \
                               "injury-induced CM proliferation and heart regeneration, highlighting the values of " \
                               "embryo-based small molecule screens in discovery of effective and safe medicine leads. "

    request = {
        "text": pubmed_30925593_abstract,
        "model_name": "token_classification"
    }
    response = requests.post(urllib.parse.urljoin(NEMOSERVE_URL, '/annotate/'), json=request)
    assert response.status_code == 200
    annotated: dict = response.json()
    logging.warning("Denotations: " + json.dumps(annotated, indent=4, sort_keys=True))

    denotations = annotated['denotations']
    assert len(denotations) > 0

    # For each denotation, try to link it with an identifier using SAPBert
    for denotation in denotations:
        text = denotation['text']
        if len(text.strip()) == 0:
            logging.warning(f"Skipping denotation {json.dumps(denotation, indent=4, sort_keys=True)}, no text.")
            continue
        response = requests.post(urllib.parse.urljoin(NEMOSAPBERT_URL, '/annotate/'), json={
            "text": text,
            "model_name": "sapbert"
        })
        sapbert: dict = response.json()
        logging.warning(f"Translated '{text}' ({denotation.get('obj', '')}) to {json.dumps(sapbert, indent=4, sort_keys=True)}")

    assert len(denotations) == 0