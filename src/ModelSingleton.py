import numpy as np
import pandas as pd
import logging
from functools import reduce
from transformers import AutoTokenizer, AutoModel
from nemo.collections.nlp.models import TokenClassificationModel
from scipy.spatial.distance import cdist
from src.utils.tokenizer import tokenizer
from src.utils.SAPElastic import SAPElastic
import yaml


logger = logging.Logger("gunicorn.error")


class ModelNotFoundError(Exception):
    pass


class ModelWrapper:
    """ Inherit this class and do any model intialization here"""
    def __init__(self):
        pass

    def __call__(self, query_text):
        """ Make a call to initialized model's Predict function"""
        raise NotImplementedError("Call to wrapped model is required")


class TokenClassificationModelWrapper(ModelWrapper):
    def __init__(self, model_path):
        """ Initializes NLP Model
        :param model_path: Path to model to load
        """
        super(TokenClassificationModelWrapper, self).__init__()
        self.model = TokenClassificationModel.restore_from(model_path)

    def sliding_window(self, text, window_size=512):
        """
        Tokenize original query into smaller chunks that the model is able to process
        :param text: Text to split up
        :param window_size: Max token size to split
        :return: Array of split text
        """
        sentences = tokenizer.tokenize(text)
        window_end = False
        current_index = 0
        splitted = 0
        while not window_end:
            current_string = []
            for index, sentence in enumerate(sentences[current_index:]):
                if reduce(lambda x, y: x + len(y.split(" ")), current_string, 0) >= window_size:
                    yield "".join(current_string)
                    current_index += index
                    break
                current_string.append(sentence)
                splitted += 1

            if splitted == len(sentences):
                window_end = True
                yield "".join(current_string)

    def _pubannotate(self, q, inferred):
        queries = [q.strip().split() for q in q]
        ids_to_labels = {v: k for k, v in self.model._cfg.label_ids.items()}
        start_idx = 0
        end_idx = 0
        denotations = []
        for query in queries:
            end_idx += len(query)
            # extract predictions for the current query from the list of all predictions
            preds = inferred[start_idx:end_idx]
            start_idx = end_idx
            for j, word in enumerate(query):
                # strip out the punctuation to attach the entity tag to the word not to a punctuation mark
                # that follows the word
                if not word[-1].isalpha():
                    word = word[:-1]
                pad = 0 if j == 0 else 1
                span_start = len(' '.join(query[:j])) + pad
                span_end = span_start + len(word)

                label = ids_to_labels[preds[j]]

                is_not_pad_label = (label != self.model._cfg.dataset.pad_label and label != '0')

                if not is_not_pad_label:
                    # For things like fitness to practice where model labels it as fitness[B-biolink:NamedThing] to[0] practice[I-biolink:NamedThing]
                    # @TODO: investigate why [ De no ##vo ma ##li ##gna ##ncy following re ##nal transplant ##ation ] would return I-biolink without a B-
                    if len(denotations) and j + 1 < len(query) and ids_to_labels[preds[j + 1]].startswith('I-'):
                        denotations[-1]['text'] += " " + word
                else:
                    if label.startswith('I-') and len(denotations):
                        denotations[-1]['span']['end'] = span_end
                        denotations[-1]['text'] += " " + word
                    else:
                        label = label.replace('B-', '').replace('I-', '')
                        denotation = {
                            'id': f'I{j}-',
                            'span': {
                                'begin': span_start,
                                'end': span_end
                            },
                            'obj': label,
                            'text': word
                        }
                        denotations.append(denotation)
        return {
            'text': ''.join(q),
            'denotations': denotations
        }

    def __add_predictions(
            self, queries, batch_size: int = 32
    ):
        """
        Add predicted token labels to the queries. Use this method for debugging and prototyping.
        Args:
            queries: text
            batch_size: batch size to use during inference.
        Returns:
            result: text with added entities
        """
        inferred = self.model._infer(queries, batch_size)
        return self._pubannotate(queries, inferred)

    @staticmethod
    def _merge_pub_annotator_annotations(annotations):
        result = {
            "text": "",
            "denotations": []
        }
        for index, a in enumerate(annotations):
            if index == 0:
                result = a
                continue
            offset = len(result['text'])
            denotations = a['denotations']
            new_denotations = [{
                'id': span['id'] + f'{index}',
                'span': {
                    'begin': span['span']['begin'] + offset,
                    'end': span['span']['end'] + offset
                },
                'obj': span['obj'],
                'text': span['text']
            } for span in denotations]
            result['text'] += a['text']
            result['denotations'] += new_denotations
        return result

    def __call__(self, query_text):
        """ Runs prediction on text"""
        try:
            queries = [x for x in self.sliding_window(query_text, 100)]
            all_predictions = [self.__add_predictions([x]) for x in queries]
            return self._merge_pub_annotator_annotations(all_predictions)
        except Exception as E:
            raise E
        finally:
            # reset the model, recover
            self.model.train(mode=self.model.training)


class TokenClassificationModelWrapperMock(ModelWrapper):
    def __init__(self, model_path):
        """ Initializes NLP Model"""
        print('hey')

    def __call__(self, query_text):
        """ Runs prediction on text"""
        return ['woop']


class SapbertModelWrapper(ModelWrapper):
    """
        host: "http://localhost:9200"
    username: "elastic"
    password: ""
    index: "sap_index"
    """
    def __init__(self, model_path, elastic_search_config):
        """ Initializes NLP Model"""
        super(SapbertModelWrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).cuda(0)
        self.elastic_client = SAPElastic(
            **elastic_search_config
        )

    def __call__(self, query_text, count=10, similarity="cosine", bl_type=""):
        """ Runs prediction on text"""
        toks = self.tokenizer.batch_encode_plus([query_text], padding="max_length", max_length=25, truncation=True,
                                                return_tensors="pt")
        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.cuda(0)
        output = self.model(**toks_cuda)
        cls_rep = output[0][:, 0, :]
        vector = cls_rep.cpu().detach().numpy()
        print(type(vector))
        print(vector)
        if similarity == "cosine":
            return self.elastic_client.search_cosine(
                query_vector=vector,
                top_n=count,
                bl_type=bl_type
            )
        elif similarity == "knn":
            return self.elastic_client.search_knn(
                query_vector=vector,
                top_n=count,
                bl_type=bl_type
            )



class ModelFactory:
    # this is populated by calling load_model
    # it stores instances
    models = {}
    # register classes here
    # when defining wrapper please register here
    # when using the config the class to be used has to be registered here.
    model_classes = {
        "TokenClassificationWrapper": TokenClassificationModelWrapper,
        'SapbertWrapper': SapbertModelWrapper
    }

    def __init__(self):
        pass

    @staticmethod
    def load_model(name, path, model_class, extra_params=None):
        if name in ModelFactory.models.keys():
            logger.info(f"Model {name} already in class skipping initialization")
            return
        else:
            logger.info(f"Initializing model {name}")
            assert issubclass(model_class, ModelWrapper), "Error please provide a subclass type of ModelWrapper"
            # initializes model and makes its prediction a callable
            if extra_params:
                ModelFactory.models[name] = model_class(path, **extra_params)
            else:
                ModelFactory.models[name] = model_class(path)

    @staticmethod
    def query_model(model_name, query_text, query_count=1, **kwargs):
        if model_name not in ModelFactory.models:
            raise ModelNotFoundError(f"Model {model_name} not found")
        # since we have model as a callable class we can just treat it like a function
        return ModelFactory.models[model_name](query_text, query_count, **kwargs)

    @staticmethod
    def get_model_names():
        return list(ModelFactory.models.keys())


def init_models(config_file_path):
    """
    Initializes models based on configuration
    :param config_file_path:
    :return:
    """
    with open(config_file_path) as config_stream:
        config = yaml.load(config_stream, Loader=yaml.SafeLoader)
    logger.info(config)
    for model_name in config:
        logger.info(model_name)
        cls = ModelFactory.model_classes.get(config[model_name]['class'], None)
        if cls is None:
            raise ValueError(
                f"model class {config[model_name]['class']} not found please use one of {ModelFactory.model_classes.keys()}, "
                f"Or add your wrapper to ModelFactory.model_classes dictionary")

        path = config[model_name]['path']
        extra_params = None

        if model_name == 'sapbert':
            extra_params = {"elastic_search_config": config[model_name].get('elasticsearch', None)}

        ModelFactory.load_model(name=model_name, path=path, model_class=cls, extra_params=extra_params)

        logger.info(f"Loaded {cls} model from {path} as {model_name}")


# test this factory by setting the model path
if __name__ == '__main__':
    model_path = "/models/SapBERT-fine-tuned-babel"
    with open('../config.yaml') as stream:
        x = yaml.load(stream, yaml.FullLoader)

    ModelFactory.load_model('sap', path=model_path, model_class=SapbertModelWrapper,extra_params={
        "elastic_search_config": x["sapbert"]["elasticsearch"]
    })
    text = "asthma"
    result = ModelFactory.query_model('sap', text)
    print(result)
