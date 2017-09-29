from loan_models.models.train_both_models import train_both_models
from loan_models.reporting.accuracy_curves import get_fallout_recall, calculate_gini
from loan_models.reporting.roc_plot import make_roc_plot

if __name__ == "__main__":
    model_type = "random_forest"
    # model_type = "gradient_boosting"
    # model_type = "deep_learning"
    bad_loan_model, interest_rate_model, valid = train_both_models(model_type)

    fallout, recall = get_fallout_recall(bad_loan_model, valid)
    gini = calculate_gini(fallout, recall)
    make_roc_plot(fallout, recall)
