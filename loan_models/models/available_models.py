import json
import os
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


def get_params(name_tag):
    """
    Return the parameters corresponding to the model name tag
    :param name_tag: name tag
    :return:
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    directory = "%s/model_parameters" % this_dir
    file_name = "%s/%s_params.json" % (directory, name_tag)
    return json.load(open(file_name, 'r'))


def random_forest_model(name):
    """
    Get the (untrained) Random Forest Model
    :param name: model name, will determine filename
    :return: model
    """
    params = get_params("random_forest")
    return H2ORandomForestEstimator(model_id=name, **params)


def gradient_boosting(name):
    """
    Get the Gradient Boosting Model
    :param name: model name, will determine filename
    :return:
    """
    params = get_params("gradient_boosting")
    return H2OGradientBoostingEstimator(model_id=name, **params)


def deep_learning(name):
    """
    Get the Deep Learning Model
    :param name: model name, will determine filename
    :return:
    """
    params = get_params("deep_learning")
    return H2ODeepLearningEstimator(model_id=name, **params)
