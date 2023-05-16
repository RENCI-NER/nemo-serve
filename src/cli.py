import argparse
import json
import yaml

from src.ModelSingleton import ModelFactory, init_models
from src.index import index_docs


from logging import getLogger

log = getLogger("")


def json_line_iterator(file_path):
    """
    Iterates over jsonline formatted file
    :param file_path:
    :return: dict of a line
    """
    with open(file_path) as stream:
        for line in stream:
            yield json.loads(line)

def annotate_text(text, model_name, count=1):
    """
    run Model on text
    :param text: input text to run predictions on
    :param model_name: name of the model to use
    :return: prediction result
    """
    return ModelFactory.query_model(model_name=model_name,query_text=text, query_count=count)

def append_json_to_file(data, output_path):
    """
    Appends a json line to an output file
    :param data: data to append
    :param output_path: file to append to
    :return: None
    """
    with open(output_path, 'a') as stream:
        stream.write(json.dumps(data) + "\n")

def process(input_file_path, output_file_path, model_name):
    """
    Runs model predications on input and writes to output
    :param input_file_path: input file path
    :param output_file_path: output file path
    :param model_name: name of model to predict with
    :return: None
    """
    progress = 0
    log.info(f"starting to process {input_file_path}")
    for json_line in json_line_iterator(input_file_path):
        log.info(f"processing line {progress}")
        prediction = annotate_text(json_line['text'], model_name)
        append_json_to_file(prediction, output_file_path)
        log.info(f"wrote prediction")
        progress += 1
    log.info(f"finished processing {progress + 1} lines")

def main(args):
    command = args.sub_command
    config = args.config_file

    if command == "annotate":
        input_file = args.input_file
        output_file = args.output_file
        model_name = args.model
        init_models(config_file_path=config)
        process(input_file_path=input_file,
                output_file_path=output_file,
                model_name=model_name)
    elif command == "index":
        # making this static for now, we only have sapbert that we want to index to elastic.
        # so we will read its previous config here.
        model_name = "sapbert"
        storage_backend = args.store
        with open(config) as stream:
            config_yaml = yaml.load(stream, Loader=yaml.FullLoader)
        gt_predictions_path = config_yaml[model_name]['ground_truth_predictions_path']
        gt_id_name_path = config_yaml[model_name]['ground_truth_id_name_pairs_path']
        gt_id_type_path = config_yaml[model_name]['ground_truth_data_id_type_pairs_path']
        storage_backend = config_yaml[model_name]['storage']
        connection_params = config_yaml[model_name]['connectionParams']
        index_docs(storage = storage_backend,
                   connection_params=connection_params,
                   np_file=gt_predictions_path,
                   name_id_file=gt_id_name_path,
                   id_type_file=gt_id_type_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CLI toolkit for nemo-serve. Supported operations include:\n"
                                                 "\t - Offline annotation model\n"
                                                 "\t - Building Elasticsearch index for SAPBert Entity linking")
    sub_parsers = parser.add_subparsers(dest="sub_command", help="Sub commands")
    # Nemo-serve config file for model and parameter config
    parser.add_argument("-c", "--config-file", help="Config file path", required=True)

    # annotate command and options
    parser_annotate = sub_parsers.add_parser("annotate", help="Perform an offline annotation on an input file via "
                                                              "provided model")
    parser_annotate.add_argument("-i", "--input-file", help="Input file path", default=None)
    parser_annotate.add_argument("-o", "--output-file", help="Output file path", default=None)
    parser_annotate.add_argument("-m", "--model", help="Model to run", default=None)

    # index command and options
    parser_index = sub_parsers.add_parser("index", help="Index SAPBert ground truth to elasticsearch. Note this uses `sapbert` config section in config file passed as paramter."
                                                        "Please refer to ../config.yaml `sapbert` section for details.")

    args = parser.parse_args()

    main(args)






