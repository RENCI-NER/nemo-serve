import numpy as np
import pandas as pd
import logging
from functools import reduce
from transformers import AutoTokenizer, AutoModel
from nemo.collections.nlp.models import TokenClassificationModel
from scipy.spatial.distance import cdist
from nltk.tokenize import sent_tokenize


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
        sentences = sent_tokenize(text)
        window_end = False
        current_index = 0
        splitted = 0
        while not window_end:
            current_string = []
            for index, sentence in enumerate(sentences[current_index:]):
                if reduce(lambda x, y: x + len(y.split(" ")), current_string, 0) >= window_size:
                    yield " ".join(current_string)
                    current_index += index
                    break
                current_string.append(sentence)
                splitted += 1

            if splitted == len(sentences):
                window_end = True
                yield " ".join(current_string)

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
                span_end = len(' '.join(query[:j + 1]))
                span_start = 0 if j == 0 else span_end - len(word)

                label = ids_to_labels[preds[j]]

                if label != self.model._cfg.dataset.pad_label and label != '0':
                    if label.startswith('I-'):
                        denotations[-1]['span']['end'] = span_end
                        denotations[-1]['text'] += " " + word
                    else:
                        label = label.replace('B-', '')
                        denotation = {
                            'id': f'I',
                            'span': {
                                'begin': span_start,
                                'end': span_end
                            },
                            'obj': label,
                            'text': word
                        }
                        denotations.append(denotation)
        return {
            'text': ' '.join(q),
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
            output_file: file to save models predictions
        Returns:
            result: text with added entities
        """
        inferred = self.model._infer(queries, batch_size)
        return self._pubannotate(queries, inferred)

    @staticmethod
    def _merge_pubtator_annotations(annotations):
        result = {
            "text": "",
            "denotations": []
        }
        for index, a in enumerate(annotations):
            if index == 0:
                result = a
                continue
            offset = len(result['text']) + 1
            denotations = a['denotations']
            new_dennotations = [{
                'id': span['id'] + f'{index}',
                'span': {
                    'begin': span['span']['begin'] + offset,
                    'end': span['span']['end'] + offset
                }
            } for span in denotations]
            result['text'] += ' ' + a['text']
            result['denotations'] += new_dennotations
        return result

    def __call__(self, query_text):
        """ Runs prediction on text"""
        try:
            queries = [x for x in self.sliding_window(query_text, 300)]
            all_predictions = [self.__add_predictions([x]) for x in queries]
            return self._merge_pubtator_annotations(all_predictions)
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
    def __init__(self, model_path, all_reps_path, all_reps_ids_path):
        """ Initializes NLP Model"""
        super(SapbertModelWrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).cuda(0)
        self.all_reps_emb = np.load(all_reps_path)
        all_reps_name_ids = pd.read_csv(all_reps_ids_path, header=0, dtype=str)
        self.all_reps_names = all_reps_name_ids['Name']
        self.all_reps_ids = all_reps_name_ids['ID']

    def __call__(self, query_text):
        """ Runs prediction on text"""
        toks = self.tokenizer.batch_encode_plus([query_text], padding="max_length", max_length=25, truncation=True,
                                                return_tensors="pt")
        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.cuda(0)
        output = self.model(**toks_cuda)
        cls_rep = output[0][:, 0, :]
        dist = cdist(cls_rep.cpu().detach().numpy(), self.all_reps_emb)
        nn_index = np.argmin(dist)
        return [self.all_reps_names[nn_index], self.all_reps_ids[nn_index]]


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
    def load_model(name, path, model_class, ground_truth_data_path=None, ground_truth_data_ids_path=None):
        if name in ModelFactory.models.keys():
            logger.info(f"Model {name} already in class skipping initialization")
            return
        else:
            logger.info(f"Initializing model {name}")
            assert issubclass(model_class, ModelWrapper), "Error please provide a subclass type of ModelWrapper"
            # initializes model and makes its prediction a callable
            if ground_truth_data_path:
                ModelFactory.models[name] = model_class(path, ground_truth_data_path, ground_truth_data_ids_path)
            else:
                ModelFactory.models[name] = model_class(path)

    @staticmethod
    def query_model(model_name, query_text):
        if model_name not in ModelFactory.models:
            raise ModelNotFoundError(f"Model {model_name} not found")
        # since we have model as a callable class we can just treat it like a function
        return ModelFactory.models[model_name](query_text)

    @staticmethod
    def get_model_names():
        return list(ModelFactory.models.keys())

# test this factory by setting the model path
if __name__ == '__main__':
    model_path = "/models/medmentions-v0.1.nemo"
    ModelFactory.load_model('medmentions', path=model_path, model_class=TokenClassificationModelWrapper)
    text = """Scientific fraud: the McBride case--judgment. Dr W G McBride, who was a specialist obstetrician and gynaecologist and the first to publish on the teratogenicity of thalidomide, has been removed from the medical register after a four-year inquiry by the Medical Tribunal of New South Wales. Of the 44 medical practice allegations made against him by the Department of Health only one minor one was found proved but 24 of the medical research allegations were found proved. Of these latter, the most serious was that in 1982 he published a scientific journal, spurious results relating to laboratory experiments on pregnant rabbits dosed with scopolamine. Had Dr McBride used any of the many opportunities available to him to make an honest disclosure of his misdemeanour, his conduct would have been excused by the Tribunal. However, he persisted in denying his fraudulent conduct for several years, including the four years of the Inquiry. The Tribunal unanimously found Dr McBride not of good character in the context of fitness to practice medicine. The decision to deregister was taken by a majority of 3 to 1. Since research science is not organized as a profession, there are no formal sanctions which can be taken against his still engaging in such research. Scientific fraud: the McBride case--judgment. Aflatoxin exposure in Singapore: blood aflatoxin levels in normal subjects, hepatitis B virus carriers and primary hepatocellular carcinoma patients. Blood screening conducted on Singaporeans over 1991-1992 showed exposure to predominantly aflatoxin B1 and to a lesser extent G1. The extent of exposure to B1 among three groups of residents in Singapore, namely normal subjects (n = 423), hepatitis B virus carriers (n = 302) and primary hepatocellular carcinoma (PHC) patients (n = 58) were extensive as reflected by the positive rates of 15.1, 0.7 and 1.7 per cent respectively. However, the degree of individual exposure to this toxin among the three groups was considered low as shown by the low respective mean blood levels of 5.4 +/- 3.2 (range 3.0-17), 7.7 (range 7.5-7.9) and 7.5 picogrammes per ml of blood. It is not immediately clear whether or not such low levels would precipitate an undesirable health effect. The higher positive rate seen in normal subjects as compared with the other groups could be due to differences in dietary intake of aflatoxin B1, differences in metabolic patterns or both. About 70 per cent of PHC patients studied were carriers. The degree of aflatoxin B1 exposure among normal subjects in Singapore was a factor of 22.1 times less than that in Japan, 40.9 times less than that in Indonesia and 51.3 times less than that in the Philippines. Similarly, the extent of exposure among hepatitis B carriers in Singapore was a factor of 8.2 times, 39.6 times and 24.2 times less than those in the other three Asiatic countries respectively. The results reflected stringent Government control over the quality of food stuff imported into this country. As Singapore imports almost all of its dietary needs from elsewhere, it can afford to be selective at a cost. Aflatoxin M1, a metabolite of B1, was most commonly encountered in the liver tissues of deceased (n = 154) who died of causes other than sickness or disease in 1992-93, consistent with our blood findings of prevalence of aflatoxin B1. High performance liquid chromatography (HPLC) with fluorescence detection using one of the aflatoxins G2 or B2 as an internal standard was used for the detection and quantification of aflatoxins. The use of an internal standard structurally and chemically similar to those required to be quantified minimizes errors in quantifications. This is because differences in the quenching of fluorescence between specimen extracts and spiked-standard extracts were internally standardized and compensated for. The presence of an internal standard also helped to locate aflatoxins of interest more accurately.(ABSTRACT TRUNCATED AT 400 WORDS)"""
    result = ModelFactory.query_model('medmentions', text)
    print(result)
