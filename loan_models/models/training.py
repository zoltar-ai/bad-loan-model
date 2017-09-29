import h2o
import json
import os

from .available_models import gradient_boosting
from .available_models import random_forest_model, deep_learning


def init_h2o():
    """
    Start up H2o
    """
    h2o.init(nthreads=-1)


def get_data():
    """
    Get the training and validation data
    :return:
    """
    loan_csv = "data/loan.csv"
    loans = h2o.import_file(loan_csv)

    print("Import approved and rejected loan requests...")

    loans["bad_loan_categorical"] = loans["bad_loan"].asfactor()
    train, valid, test = loans.split_frame([0.79, 0.2], seed=1234)
    return train, valid


def get_input_variables():
    """
    Get the input variables (predictors) used to train the model
    :return:
    """
    input_variables = ["loan_amnt", "longest_credit_length", "revol_util",
                       "emp_length", "home_ownership", "annual_inc",
                       "purpose", "addr_state", "dti", "delinq_2yrs",
                       "total_acc", "verification_status", "term"]

    return input_variables


def get_trained_model(train, valid, model_name, target_variable, model_type):
    """
    :param train: Training frame
    :param valid: Validation frame
    :param model_name: String, Name of model, determines name of file
    :param target_variable: String, Target variable to be predicted
    :return: Trained model
    """

    model = None

    if model_type == "random_forest":
        model = random_forest_model(model_name)
    elif model_type == "gradient_boosting":
        model = gradient_boosting(model_name)
    elif model_type == "deep_learning":
        model = deep_learning(model_name)
    else:
        raise ValueError("Unrecognized model_type: %s" % model_type)

    input_variables = get_input_variables()

    model.train(input_variables, target_variable,
                training_frame=train, validation_frame=valid)

    print(model)
    write_model_pojo(model)
    return model


def write_model_pojo(model):
    """
    Write the model as POJO
    :param model: trained model
    :return: None
    """

    # Relative path from code dir
    output_directory = "build"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    h2o.download_pojo(model, path=output_directory)


def get_gini(model):
    """
    Print out the Gini coefficient for binary classification models
    :param model: Trained H2o model
    """
    gini = model.gini(valid=True)
    print("Gini coefficient: %s" % gini)
    return gini


def create_outputs(model, model_name, model_type):
    output = {"model_name": model_name,
              "model_type": model_type}

    if model_name == "BadLoanModel":
        gini = get_gini(model)
        output['gini'] = gini

    return output


def write_outputs(model, model_name, model_type):
    output = create_outputs(model, model_name, model_type)
    out_file = 'build/model_output_data_%s_%s.json' % (model_name, model_type)
    json.dump(output, open(out_file, 'w'), indent=3, sort_keys=True)
