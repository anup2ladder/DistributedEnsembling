#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:29:53 2019
@author: anupt

The dummy models: random_predictions and majority_predictions
"""

# %% Import modules
import numpy as np
from local_library import models_train_test


# %% random_predicitions
def random_predictions(num_test, prevalence, targets,
                       class_labels, perf_metrics,
                       ):
    '''Generate random predictions based on
    the prevalence of the majority class'''

    # Random predictions of probabilities
    rand_predict_prob = np.random.random_sample(num_test)

    # Classify the predictions as True[1] or False[0]
    # based on the predicted probabilities.
    # Default cutoff is 0.5
    # However, set cutoff at prevalence so the random predictor's predictions
    # are similar to that of the data distribution
    # i.e. set cutoff as prevalence of False class (prevalence['c0'])
    rand_predict_class = models_train_test.classify_prob(
            rand_predict_prob, prevalence['majority_prev'])

    # Results
    rand_model = models_train_test.model_classwise_results(
            targets, rand_predict_prob, rand_predict_class,
            class_labels, perf_metrics
            )

    return rand_model


# %% majority_predictions
def majority_predictions(num_test, prevalence, targets,
                         class_labels, perf_metrics,
                         ):
    '''Results from predicting the majority class'''

    # Predict only the majority class
    major_predict_prob = [ prevalence['majority_class'] ]*num_test

    # Classify the predictions as True[1] or False[0]
    # based on the predicted probabilities.
    # Use default threshold of 0.5
    major_predict_class = models_train_test.classify_prob(major_predict_prob)

    # Get results
    major_model = models_train_test.model_classwise_results(
            targets, major_predict_prob, major_predict_class,
            class_labels, perf_metrics
            )

    return major_model
