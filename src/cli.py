from src.ModelSingleton import ModelFactory, init_models
import yaml
import json
import argparse


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

def annotate_text(text, model_name):
    """
    run Model on text
    :param text: input text to run predictions on
    :param model_name: name of the model to use
    :return: prediction result
    """
    return ModelFactory.query_model(model_name=model_name,query_text=text)

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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Annotate long text offline.")
    parser.add_argument("-i", "--input-file", help="Input file path", default=None)
    parser.add_argument("-o", "--output-file", help="Output file path", default=None)
    parser.add_argument("-c", "--config-file", help="Config file path", default=None)
    parser.add_argument("-m", "--model", help="Model to run", default=None)
    args = parser.parse_args()

    config = args.config_file
    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model
    init_models(config_file_path=config)
    process(input_file_path=input_file,
            output_file_path=output_file,
            model_name=model_name)






