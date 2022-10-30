import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def boxplot(df, num_variable, cat_variable, rota=0):
    """
    Plot a boxplot
    :param df: dataframe
    :param num_variable: numerical variable
    :param cat_variable: categorical variable
    :param rota: rotation
    """
    bp = sns.boxplot(y=num_variable, x=cat_variable, data=df, width=0.5, palette="colorblind",
                     showfliers=False, showmeans=True,
                     meanprops={"marker": "o",
                                "markerfacecolor": "white",
                                "markeredgecolor": "black",
                                "markersize": "5"})
    lab = bp.get_xticklabels()
    bp.set_xticklabels(labels=lab, rotation=rota)
    plt.show()


def contingency_table(df, cat_variable_1, cat_variable_2):
    """
    Compute contingency table
    :param df: dataframe
    :param cat_variable_1: categorical variable 1
    :param cat_variable_2: categorical variable 2
    :return:
    """
    cont = df[[cat_variable_1, cat_variable_2]].pivot_table(index=cat_variable_1, columns=cat_variable_2,
                                                            aggfunc=len, margins=True, margins_name="Total")
    cont = cont.fillna(0)
    return cont


def heatmap(cont, data):
    """
    Plot heatmap from contingency table, color according to contribution to non-independence
    :param cont: contingency table
    :param data: dataframe
    """

    tx = cont.loc[:, ["Total"]]
    ty = cont.loc[["Total"], :]
    n = len(data)
    indep = tx.dot(ty) / n
    measure = (cont - indep) ** 2 / indep
    xi_n = measure.sum().sum()
    table = measure / xi_n
    sns.heatmap(table.iloc[:-1, :-1], annot=cont.iloc[:-1, :-1])
    plt.show()


def weighted_log_loss(y_true, y_pred, labels=None):
    """
    Compute weighted log loss
    :param y_true: true labels
    :param y_pred: predictions (probabilities)
    :param labels: labels or classes
    :return: weighted log loss
    """
    # Log loss is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p)).
    eps = 1e-15
    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # If y_pred is of single dimension, assume y_true to be binary and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)
    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    # Transform to list
    y_true = list(y_true)
    # Weights as mentionned
    weights = np.logspace(1, len(set(y_true)), len(set(y_true)))
    # label binarizer
    lb = LabelBinarizer()
    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)
    # Transform y_true
    transformed_labels = lb.transform(y_true)
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels, transformed_labels, axis=1)
    return np.average((-(transformed_labels * np.log(y_pred) * weights)).sum(axis=1))


def my_log_loss(y_true, y_pred, labels=None):
    """
    Compute weighted log loss
    :param y_true: true labels
    :param y_pred: predictions (probabilities)
    :param labels: labels or classes
    :return: weighted log loss
    """
    # Log loss is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p)).
    eps = 1e-15
    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # If y_pred is of single dimension, assume y_true to be binary and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)
    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    # Transform to list
    y_true = list(y_true)
    # label binarizer
    lb = LabelBinarizer()
    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)
    # Transform y_true
    transformed_labels = lb.transform(y_true)
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels, transformed_labels, axis=1)
    return np.average((-(transformed_labels * np.log(y_pred))).sum(axis=1))
