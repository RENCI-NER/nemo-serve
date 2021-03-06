import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModel
from nemo.collections.nlp.models import TokenClassificationModel
from scipy.spatial.distance import cdist


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
        """ Initializes NLP Model"""
        super(TokenClassificationModelWrapper, self).__init__()
        self.model = TokenClassificationModel.restore_from(model_path)

    def __call__(self, query_text):
        """ Runs prediction on text"""
        return self.model.add_predictions([query_text])


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
    model_path = "/exp/trained_model.nemo"
    ModelFactory.load_model('medmentions', path=model_path, model_class=TokenClassificationModelWrapper)
    result = ModelFactory.query_model('medmentions', "asthma is a disease")
    print(result)
