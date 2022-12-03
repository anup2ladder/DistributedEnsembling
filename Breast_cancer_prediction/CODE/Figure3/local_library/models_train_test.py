#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:15:23 2019
@author: anupt

A set of functions to make predictions, classifications and measure performance

Predict the class probability from themodel
Classify based on the probabilities
Measure the performance of these classifications
"""

# %% Import modules
import sklearn
import numpy as np
from local_library import initialize_outputs as io


# %%
def classify_prob(predict_prob, classification_threshold=0.5):
    """
    A function to classify predictions
    Using a default cutoff of 0.5 for probabilities, classify
    the predicted class probabilities as 1 and 0
    """
    predict_class = [
            1 if x >= classification_threshold else 0 for x in predict_prob
            ]
    return predict_class


# %%
def model_classwise_results(
        targets, predict_prob, predict_class,
        class_labels, perf_metrics,
        ):

    # %% Initialize class-wise and metric-wise template to store results
    model_results = io.oneshot_metrics(class_labels, perf_metrics)

    # %% Calculate the class-wise performance metrics from predictions
    for j_class in class_labels.keys():
        model_results[j_class]['acc'], \
            model_results[j_class]['prec'], \
            model_results[j_class]['recall'], \
            model_results[j_class]['f1'], \
            model_results[j_class]['mcc'], \
            model_results[j_class]['roc_auc'], \
            model_results[j_class]['PR_auc'], \
            = measure_prediction_performance(
                    targets,
                    predict_prob,
                    predict_class,
                    avgtype=class_labels[j_class]['avgtype'],
                    label=class_labels[j_class]['label']
                    )

    # Return results
    return model_results


# %%
def measure_prediction_performance(targets, model_prob, model_class,
                                   avgtype='macro', label=1
                                   ):
    """
    A function to calculate a number of model performance metrics
    based on input predicted probablities and classes from a model

    N.B. IT IS ASSUMED THE TARGETS IS A PANDAS SERIES with title "CLASS"
    """

    # All performance metrics
    acc = sklearn.metrics.accuracy_score(
            targets['CLASS'], model_class
            )
    prec = sklearn.metrics.precision_score(
            targets['CLASS'], model_class,
            pos_label=label, average=avgtype
            )
    recall = sklearn.metrics.recall_score(
            targets['CLASS'], model_class,
            pos_label=label, average=avgtype
            )
    f1score = sklearn.metrics.f1_score(
            targets['CLASS'], model_class,
            pos_label=label, average=avgtype
            )
    mcc = sklearn.metrics.matthews_corrcoef(
            targets['CLASS'], model_class
            )
    roc_fpr, roc_tpr, = sklearn.metrics.roc_curve(
            targets['CLASS'], model_prob
            )[0:2]
    auc_roc = sklearn.metrics.auc(
            roc_fpr, roc_tpr
            )
    PR_p, PR_r, = sklearn.metrics.precision_recall_curve(
            targets['CLASS'], model_prob
            )[0:2]
    auc_PR = sklearn.metrics.auc(
            PR_r, PR_p
            )

    return acc, prec, recall, f1score, mcc, auc_roc, auc_PR


# %%
def mean_stdev_iterations(iteration):
    '''A function to return a dictionary that contains
    the mean and standard deviation of a given list of
    iteration results (or any list, in general)'''
    result = {
        'mean': np.mean(iteration),
        'stdev': np.std(iteration)
    }
    return result
