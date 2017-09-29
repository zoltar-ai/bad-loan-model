import numpy as np
from scipy.integrate import trapz


def get_accuracy_curves(y_true, y_prob):
    """
    Calculate accuracy curves. Used for making ROC curve.
    :param y_true: Boolean for target variable
    :param y_prob: Corresponding score
    :return: dictionary of accuracy measures including confusion matrix,
        recall and fallout
    """
    n = len(y_true)
    assert len(y_prob) == n
    y_prob, n_true, n_false = count_true_false_by_prob(y_true, y_prob)

    so = np.argsort(-y_prob)
    thresh = y_prob[so]
    n_true = n_true[so]
    n_false = n_false[so]

    # Get confusion matrix for each threshold
    # The [::-1] indexing reverses the array

    true_pos = n_true.cumsum()
    false_pos = n_false.cumsum()
    true_neg = n_false[::-1].cumsum()[::-1] - n_false
    false_neg = n_true[::-1].cumsum()[::-1] - n_true

    recall = true_pos / (true_pos + false_neg)
    fallout = false_pos / (false_pos + true_neg)

    accuracy_measures = {"threshold": thresh,
                         "true_pos": true_pos,
                         "true_neg": true_neg,
                         "false_pos": false_pos,
                         "false_neg": false_neg,
                         "recall": recall,
                         "fallout": fallout}

    return accuracy_measures


def count_true_false_by_prob(y_true, y_prob):
    """
    Take a Boolean for target and score and return
    a unique score and counts (true and false).
    Required for making ROC curves when score is not
    necessarily unique.
    :param y_true: Boolean for target variable
    :param y_prob: Corresponding score
    :return: y_prob_unique, n_true, n_false
    """
    counts = {}
    for yt, yp in zip(y_true, y_prob):
        if yp not in counts:
            # initialize
            counts[yp] = [0, 0]

        counts[yp][0] += yt
        counts[yp][1] += (not yt)

    items = sorted(counts.items(), key=lambda x: x[0])
    y_prob_unique = [i[0] for i in items]
    n_true = [i[1][0] for i in items]
    n_false = [i[1][1] for i in items]

    # Make numpy arrays
    y_prob_unique = np.array(y_prob_unique)
    n_true = np.array(n_true)
    n_false = np.array(n_false)

    return y_prob_unique, n_true, n_false


def calculate_gini(fallout, recall):
    """
    Calculate the Gini score from fallout and recall
    :param fallout: numpy array
    :param recall: numpy array
    :return: Gini score
    """
    # Sort by fallout but use recall for tie breaker
    so = np.argsort(fallout + 1e-9 * recall)

    # Area under the curve
    auc = trapz(recall[so], fallout[so])
    gini = 2 * auc - 1
    print('Calculated Gini is: %s' % gini)
    return gini


def get_list_from_frame(frame):
    # TODO: This is awkward but it works
    data_frame = frame.as_data_frame()[1:]
    return [float(i[0]) for i in data_frame]


def get_score_from_model_and_frame(model, frame):
    prediction = model.predict(frame)['p1']
    score = get_list_from_frame(prediction)
    assert len(score) == len(frame)
    return score


def get_y_true_from_frame(frame):
    y_true = get_list_from_frame(frame['bad_loan'] == 1)
    return [int(i) for i in y_true]


def get_y_true_and_score_from_frame(model, frame):
    score = get_score_from_model_and_frame(model, frame)
    y_true = get_y_true_from_frame(frame)
    return np.array(y_true), np.array(score)


def get_fallout_recall(model, frame):
    y_true, score = get_y_true_and_score_from_frame(model, frame)
    accuracy_curves = get_accuracy_curves(y_true, score)
    return accuracy_curves['fallout'], accuracy_curves['recall']
