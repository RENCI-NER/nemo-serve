import logging
import math
import re
import yaml
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from nemo.collections.nlp.models import TokenClassificationModel
import torch
from src.utils.SAPRedis import RedisMemory
from src.utils.SAPQdrant import SAPQdrant
from src.utils.tokenizer import tokenizer

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
        # Make this an instance variable so that it's easier to mock in
        # testing:
        self.sentence_tokenizer = tokenizer

    def _get_token_length(self, input_text):
        """Return the length in tokens as understood by the model's own internal
        tokenizer.
        """
        tokens = self.model.tokenizer.text_to_tokens(input_text)
        return len(tokens)

    def _token_chunks(self, input_text, window_size=512):
        """Break sentences into chunks. Yields a tuple of token count, string.

        Each chunk will be smaller than window_size tokens.
        This function is a generator
        """
        logger.debug("_token_chunks called on text: %s", input_text)
        tokens = self.model.tokenizer.text_to_tokens(input_text)
        token_count = len(tokens)
        logger.debug("Text tokenized to %d tokens", token_count)

        if token_count < window_size:
            # Chunk has been sufficiently split before now. Just kick it back.
            yield (token_count, input_text)

        else:
            # This should be something close to equal blocks of tokens. Chunking
            # it this way reduces the chance of a small chunk at the end of a
            # string of text.

            # This calculates the number of chunks to split.
            # There are often more tokens than words in a string, so we use
            # a factor of 4 to cover of that. Pieces may be re-assembled in
            # sliding_window.
            nchunks = math.ceil(token_count/(window_size * 4))
            if not nchunks:
                logger.debug("Zero tokens found, returning None")
                return []
            chunk_size = math.ceil(token_count/nchunks)
            logger.debug(
                "Sentence will be broken into %d chunks of size %d words",
                nchunks, chunk_size)

            for match in re.finditer(r'((?:\S+\s+){1,100}(?:\S+\s*))',
                                     failing_text):
                if match and match.lastindex > 0:
                    chunk = match.group(1)
                    chunk_token_count = self._get_token_length(chunk)
                    yield (chunk_token_count, chunk)

    def _sentences_to_chunks(self, sentences, window_size):
        """
        Take a list of sentences, return an array of lists of otkens that are
        all smaller than window_size.
        """
        for sentence in sentences:
            sentence_token_length = self._get_token_length(sentence)
            if sentence_token_length >= window_size:
                logger.debug("Found an extra-long sentence, "
                             "breaking it up:\n%s\n", sentence)

                # Try splitting on semicolons into sentence fragments
                # re.split with a lookbehind pattern includes the semicolon on
                # the split text.
                split_list = re.split(r'(?<=\;\s)|(?<=\;)', sentence)
                logger.debug("Semicolon split broke it into %d pieces",
                             len(split_list))

                for fragment in split_list:
                    # Break any fragment over window_size into bits.
                    yield from self._token_chunks(fragment, window_size)
            else:
                logger.debug("Sentence is under chunk size (has %d tokens, "
                             "returning: %s", sentence_token_length, sentence)
                yield (sentence_token_length, sentence)

    def sliding_window(self, input_text, window_size=512):
        """
        Tokenize original query into smaller chunks that the model can process

        This refactored function uses a stack data structure instead of a
        rolling window.
        :param text: Text to split up
        :param window_size: Max token size to split
        :return: Array of split text
        """
        logger.debug("sliding window called with text of length %d chars",
                     len(input_text))
        sentences = self.sentence_tokenizer.tokenize(input_text)
        logger.debug("Text broken into %d sentences", len(sentences))

        current_string = ""
        current_token_length = 0
        for (chunk_token_length, chunk) in self._sentences_to_chunks(
                sentences, window_size):
            logger.debug("sliding_window is working on sentence chunk %s",
                         str(chunk))
            if current_token_length + chunk_token_length >= window_size:
                # New sentence would make the chunk too long. Yield the
                # existing chunk and start a new chunk.
                if current_string:
                    yield current_string
                current_string = chunk
                current_token_length = chunk_token_length
            else:
                current_string += chunk
                current_token_length += chunk_token_length
                logger.debug("current_string is %d tokens long",
                             current_token_length)
        yield current_string

    def _pubannotate(self, q, inferred):
        queries = [q.strip().split() for q in q]
        ids_to_labels = {v: k for k, v in self.model._cfg.label_ids.items()}
        start_idx = 0
        end_idx = 0
        denotations = []
        for query in queries:
            end_idx += len(query)
            # extract predictions for the current query from the list of all
            # predictions
            preds = inferred[start_idx:end_idx]
            start_idx = end_idx
            for j, word in enumerate(query):
                # strip out the punctuation to attach the entity tag to the word
                # not to a punctuation mark that follows the word
                if not word[-1].isalpha():
                    word = word[:-1]
                pad = 0 if j == 0 else 1
                span_start = len(' '.join(query[:j])) + pad
                span_end = span_start + len(word)

                label = ids_to_labels[preds[j]]

                is_not_pad_label = (label != self.model._cfg.dataset.pad_label
                                    and label != '0')

                if not is_not_pad_label:
                    # For things like fitness to practice where model labels it as fitness[B-biolink:NamedThing] to[0] # practice[I-biolink:NamedThing]
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
        Add predicted token labels to the queries.

        Use this method for debugging and prototyping.
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
    def __call__(self, query_text, *args, **kwargs):
        """ Runs prediction on text"""
        try:
            queries = [x for x in self.sliding_window(query_text, 500)]
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
    def __init__(self, model_path, connection_config, backend='redis'):
        """ Initializes NLP Model"""
        super(SapbertModelWrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gpu_avaialble = torch.cuda.is_available()
        if self.gpu_avaialble:
            self.model = AutoModel.from_pretrained(model_path).cuda(0)
        else:
            self.model = AutoModel.from_pretrained(model_path)
        if backend == 'redis':
            self.storage_client = RedisMemory(
                **connection_config
            )
        elif backend =="qdrant":
            self.storage_client = SAPQdrant(
                **connection_config
            )
        else:
            raise ValueError(f"Unsupported storage backend: {backend}")

    async def __call__(self, query_text, count=10, similarity="cosine", bl_type=""):
        """ Runs prediction on text"""
        toks = self.tokenizer.batch_encode_plus(
            [query_text], padding="max_length", max_length=25, truncation=True,
            return_tensors="pt")
        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.cuda(0) if self.gpu_avaialble else v
        output = self.model(**toks_cuda)
        cls_rep = output[0][:, 0, :]
        vector = cls_rep.cpu().detach().numpy().tolist()[0]
        logger.info(f"Calculated Vector of {len(vector)} dims,")
        logger.info("sending vector to elasticsearch")
        return await self.storage_client.search(
            query_vector=vector,
            top_n=count,
            bl_type=bl_type,
            algorithm=similarity
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
            logger.info("Model %s already in class skipping initialization",
                        name)
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
    async def query_model(model_name, query_text, query_count=1, **kwargs):
        if model_name not in ModelFactory.models:
            raise ModelNotFoundError(f"Model {model_name} not found")
        # since we have model as a callable class we can just treat it like a function
        return await ModelFactory.models[model_name](query_text, query_count, **kwargs)

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
            extra_params = {"connection_config": config[model_name].get('connectionParams', None),
                            "backend": config[model_name]["storage"]}

        ModelFactory.load_model(name=model_name, path=path, model_class=cls, extra_params=extra_params)

        logger.info(f"Loaded {cls} model from {path} as {model_name}")


def run_main():
    "pulling this into a function keeps the top-level namespace cleaner"
    model_path = "/models/medmentions-v0.1.nemo"
    ModelFactory.load_model('medmentions', path=model_path, model_class=TokenClassificationModelWrapper)
    text = """Scientific fraud: the McBride case--judgment. Dr W G McBride, who was a specialist obstetrician and gynaecologist and the first to publish on the teratogenicity of thalidomide, has been removed from the medical register after a four-year inquiry by the Medical Tribunal of New South Wales. Of the 44 medical practice allegations made against him by the Department of Health only one minor one was found proved but 24 of the medical research allegations were found proved. Of these latter, the most serious was that in 1982 he published a scientific journal, spurious results relating to laboratory experiments on pregnant rabbits dosed with scopolamine. Had Dr McBride used any of the many opportunities available to him to make an honest disclosure of his misdemeanour, his conduct would have been excused by the Tribunal. However, he persisted in denying his fraudulent conduct for several years, including the four years of the Inquiry. The Tribunal unanimously found Dr McBride not of good character in the context of fitness to practice medicine. The decision to deregister was taken by a majority of 3 to 1. Since research science is not organized as a profession, there are no formal sanctions which can be taken against his still engaging in such research. Scientific fraud: the McBride case--judgment. Aflatoxin exposure in Singapore: blood aflatoxin levels in normal subjects, hepatitis B virus carriers and primary hepatocellular carcinoma patients. Blood screening conducted on Singaporeans over 1991-1992 showed exposure to predominantly aflatoxin B1 and to a lesser extent G1. The extent of exposure to B1 among three groups of residents in Singapore, namely normal subjects (n = 423), hepatitis B virus carriers (n = 302) and primary hepatocellular carcinoma (PHC) patients (n = 58) were extensive as reflected by the positive rates of 15.1, 0.7 and 1.7 per cent respectively. However, the degree of individual exposure to this toxin among the three groups was considered low as shown by the low respective mean blood levels of 5.4 +/- 3.2 (range 3.0-17), 7.7 (range 7.5-7.9) and 7.5 picogrammes per ml of blood. It is not immediately clear whether or not such low levels would precipitate an undesirable health effect. The higher positive rate seen in normal subjects as compared with the other groups could be due to differences in dietary intake of aflatoxin B1, differences in metabolic patterns or both. About 70 per cent of PHC patients studied were carriers. The degree of aflatoxin B1 exposure among normal subjects in Singapore was a factor of 22.1 times less than that in Japan, 40.9 times less than that in Indonesia and 51.3 times less than that in the Philippines. Similarly, the extent of exposure among hepatitis B carriers in Singapore was a factor of 8.2 times, 39.6 times and 24.2 times less than those in the other three Asiatic countries respectively. The results reflected stringent Government control over the quality of food stuff imported into this country. As Singapore imports almost all of its dietary needs from elsewhere, it can afford to be selective at a cost. Aflatoxin M1, a metabolite of B1, was most commonly encountered in the liver tissues of deceased (n = 154) who died of causes other than sickness or disease in 1992-93, consistent with our blood findings of prevalence of aflatoxin B1. High performance liquid chromatography (HPLC) with fluorescence detection using one of the aflatoxins G2 or B2 as an internal standard was used for the detection and quantification of aflatoxins. The use of an internal standard structurally and chemically similar to those required to be quantified minimizes errors in quantifications. This is because differences in the quenching of fluorescence between specimen extracts and spiked-standard extracts were internally standardized and compensated for. The presence of an internal standard also helped to locate aflatoxins of interest more accurately.(ABSTRACT TRUNCATED AT 400 WORDS)"""
    result = ModelFactory.query_model('medmentions', text)
    print(result)

# test this factory by setting the model path
if __name__ == '__main__':
    run_main()

