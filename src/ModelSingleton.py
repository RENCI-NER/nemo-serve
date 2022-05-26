
import logging

from nemo.collections.nlp.models import TokenClassificationModel

logger = logging.Logger("model-loader")


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


class ModelFactory:
    models = {}

    def __init__(self):
        pass

    @staticmethod
    def load_model(name, path, model_class):
        if name in ModelFactory.models.keys():
            logger.info(f"Model {name} already in class skipping initialization")
            return
        else:
            logger.info(f"Initializing model {name}")
            assert issubclass(model_class, ModelWrapper), "Error please provide a subclass type of ModelWrapper"
            # initializes model and makes its prediction a callable
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
