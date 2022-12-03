#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:26:14 2019
@author: anupt

A set of functions to train support vector machines with distributed learning
"""


# %% Import modules
import copy
import numpy as np
import sklearn
import sklearn.svm

from local_library import models_train_test
from local_library import distributed_models


# %%
def GEC_SVM(
        gec_type,
        kernel,
        central_features, central_targets,
        num_agents,
        agent_features, agent_targets,
        test_features, test_targets, scram_test_features,
        class_labels, perf_metrics,
        poly_degree=3
        ):
    """
    Global ensemble classifier using SVM to train agents
    Requires a specification of type of GEC to do (gec_type) as string
        Accepted inputs:
            1. "average_probability"
            2. "majority_vote"
    """

    # %% Perform central training
    # Get all 7 returned values
    central_SVM, central_results, _ignore_scram, \
        central_predict_prob, _ignore_scram, \
        central_predict_class, _ignore_scram = distributed_SVM(
                kernel, poly_degree,
                num_agents['central'], central_features, central_targets,
                test_features, test_targets, scram_test_features,
                class_labels, perf_metrics
                )

    # %% Perform distributed training
    # Get all 7 returned values
    agent_SVMs, avg_agent_results, scram_avg_agent_results, \
        agent_predict_prob, scram_predict_prob, \
        agent_predict_class, scram_predict_class = distributed_SVM(
                kernel, poly_degree,
                num_agents['dist'], agent_features, agent_targets,
                test_features, test_targets, scram_test_features,
                class_labels, perf_metrics
            )

    # %% Now ensemble the individual agents
    # with type of ensemble specified by the caller, with "gec_type"

    # Ensemble all agents, unweighted
    ensemble_all = distributed_models.GEC_main(
            gec_type,
            agent_predict_prob,
            agent_predict_class,
            test_targets, class_labels, perf_metrics)

    # %% Return average agent and ensemble results
    return central_results, \
        avg_agent_results,  \
        ensemble_all


# %%
def distributed_SVM(
        kernel, poly_degree,
        num_agents, agent_features, agent_targets,
        test_features, test_targets, scram_test_features,
        class_labels, perf_metrics
        ):
    """
    Perform distributed training of a model using independent datasets
    Returns a list of trained weights, average agent performance (+scram),
    predicted probalilities and classifications (+scram)
    """

    # %% Train the SVM agents
    agent_SVMs = train_agents_SVM(
            kernel, poly_degree,
            num_agents, agent_features, agent_targets,
            )

    # %% Individual agent Test predictions
    # Probabilities and classifications

    # Test features
    agent_predict_prob, agent_predict_class = individual_agent_predictions_SVM(
                agent_SVMs, num_agents,
                test_features, classification_threshold=0.5
                )

    # Scrambled Test features
    scram_predict_prob, scram_predict_class = individual_agent_predictions_SVM(
                agent_SVMs, num_agents,
                scram_test_features, classification_threshold=0.5
                )

    # %% Individual agent Test perforamnce
    # List of class-wise performance metrics for each agent

    # Predictions from test features
    indv_agent_results = distributed_models.individual_agent_performance(
            test_targets, num_agents,
            class_labels, perf_metrics,
            agent_predict_prob, agent_predict_class
            )

    # Predictions from scrambled test features
    indv_scram_results = distributed_models.individual_agent_performance(
            test_targets, num_agents,
            class_labels, perf_metrics,
            scram_predict_prob, scram_predict_class
            )

    # %% Average Test performance of individual agents
    # Linear average of individual agent performance

    # Predictions from Test features
    avg_agent_results = distributed_models.average_agent_performance(
            num_agents, class_labels, perf_metrics, indv_agent_results
            )

    # Predictions from scrambled test features
    scram_avg_agent_results = distributed_models.average_agent_performance(
            num_agents, class_labels, perf_metrics, indv_scram_results
            )

    # %% Return agent weights, average performances and individual predictions
    # A long tuple of length = 7
    # For GEC majority vote, you would want to capture all of these
    # For GEC average probability, only first 5 are sufficient i.e. [0:5]
    return agent_SVMs, avg_agent_results, scram_avg_agent_results, \
        agent_predict_prob, scram_predict_prob, \
        agent_predict_class, scram_predict_class


# %%
def train_agents_SVM(
        kernel, polydegree,
        num_agents, agent_features, agent_targets
        ):
    '''
    A function to train a set of indepdendent distributed agents that
    each have their own independent training dataset.

    Returns a list of trained SVMs, one per agent
    '''

    # %% Initialize list to store the trained weights
    agent_SVMs = [
            sklearn.svm.SVC(kernel=kernel, degree=polydegree, probability=True)
            for i in range(0, num_agents)
            ]

    # %% Train agents
    # np.ravel(targets) is necessary because SVM asks for it that way (shrug)
    for n_agent in range(0, num_agents):
        agent_SVMs[n_agent].fit(
                agent_features[n_agent],
                np.ravel(agent_targets[n_agent]))

    # %% Return trained SVMs for all agents
    return agent_SVMs


# %%
def individual_agent_predictions_SVM(
        agent_SVMs, num_agents,
        features, classification_threshold=0.5
        ):
    # Make predictions for each agent
    # Calculate probabilities and classifications using the inputted features
    # whether they are training or test features, or scrambled features

    # %% Initialize lists to store the predictions
    agent_predict_prob = [[] for i in range(0, num_agents)]
    agent_predict_class = copy.deepcopy(agent_predict_prob)

    # %% Make predictions
    for n_agent in range(0, num_agents):
        agent_predict_prob[n_agent], agent_predict_class[n_agent] = \
             model_predictions_SVM(agent_SVMs[n_agent], features)

    # %% Return predicted probabilities and classifications
    return agent_predict_prob, agent_predict_class


# %%
def model_predictions_SVM(SVM_model, features):
    """
    Predict probabilities and classifications from SVM(SVC) model
    The class predictions use SVC.predict, and are cool
    The probabilities are hacky and limited to binary
        This is because I have to figure out which column of probabilities
        correspond to the classifications made with SVC.predict().
        This is becaise I can't pin down necessarily which column corresponds
        to the predictions of SVC.predict(), as this seems to vary sometimes

        What I essentially do then manually check which column to use
    """

    # Make normal SVM predictions
    predict_class = SVM_model.predict(features)

    # Make "probability" predictions. The output has 2 columns for binary
    svc_proba = SVM_model.predict_proba(features)

    # Make sure these probability predictions are within the normal 0-1 range,
    # and sum to 1
    # Get row sum
    row_sum = np.sum(svc_proba, axis=1)
    # Scale each row by its sum
    for i in range(len(row_sum)):
        svc_proba[i, :] = svc_proba[i, :] / row_sum[i]

    # Now figure out which column of probabilities corresponds to the
    # classification. To do this, classify the first column of the first row
    # of predicted probabilities, and compare this to the first row of
    # the classifications. Using this, assign the probabilities

    # Calculate the classifications based on the second column
    svc_proba_class1 = models_train_test.classify_prob(svc_proba[:, 1])
    correspondance = sklearn.metrics.accuracy_score(
            predict_class, svc_proba_class1)
    if correspondance >= 0.5:
        # Case where second column is representative of probabilities
        # NB: Modification so if it is exactly 0.5, defaults to second column
        predict_prob = svc_proba[:, 1]
    elif correspondance < 0.5:
        # Case where first column is representative of probabilities
        predict_prob = svc_proba[:, 0]
    else:
        assert False, "ERROR in predicted probabilities from SVM"

    return predict_prob, predict_class
