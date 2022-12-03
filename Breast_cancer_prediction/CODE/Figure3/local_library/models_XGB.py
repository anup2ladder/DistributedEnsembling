#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:26:14 2019
@author: anupt

A set of functions to train XGBoost with distributed learning
"""


# %% Import modules
import copy
import numpy as np
import sklearn
import xgboost

from local_library import models_train_test
from local_library import distributed_models


# %%
def GEC_XGB(
        gec_type,
        XGB_param,
        central_features, central_targets,
        num_agents, training_c0, training_c1,
        agent_features, agent_targets,
        test_features, test_targets, scram_test_features,
        class_labels, perf_metrics
        ):
    """
    Global ensemble classifier using XGBoost to train agents
    Requires a specification of type of GEC to do (gec_type) as string
        Accepted inputs:
            1. "average_probability"
            2. "majority_vote"
    """

    # %% Perform central training
    # Get all 7 returned values
    central_XGB, central_results, _ignore_scram, \
        central_predict_prob, _ignore_scram, \
        central_predict_class, _ignore_scram = distributed_XGB(
                XGB_param,
                num_agents['central'], central_features, central_targets,
                test_features, test_targets, scram_test_features,
                class_labels, perf_metrics
                )

    # %% Perform distributed training
    # Get all 7 returned values
    agent_XGBs, avg_agent_results, scram_avg_agent_results, \
        agent_predict_prob, scram_predict_prob, \
        agent_predict_class, scram_predict_class = distributed_XGB(
                XGB_param,
                num_agents['total'], agent_features, agent_targets,
                test_features, test_targets, scram_test_features,
                class_labels, perf_metrics
            )

    # %% Get results for average majority class 0 and majority class 1 agents
    avg_major_c0_results, avg_major_c1_results = \
        distributed_models.average_g1_g2_performance(
                agent_predict_prob, agent_predict_class,
                num_agents, 'major_c0', 'minor_c0',
                test_targets, class_labels, perf_metrics
                )

    # %% Now ensemble the individual agents
    # with type of ensemble specified by the caller, with "gec_type"

    # Ensemble majority class 0 agents
    ensemble_major_c0 = distributed_models.GEC_main(
            gec_type,
            agent_predict_prob[0:num_agents['major_c0']],
            agent_predict_class[0:num_agents['major_c0']],
            test_targets, class_labels, perf_metrics)

    # Ensemble majority class 1 agents
    ensemble_major_c1 = distributed_models.GEC_main(
            gec_type,
            agent_predict_prob[num_agents['major_c0']:num_agents['total']],
            agent_predict_class[num_agents['major_c0']:num_agents['total']],
            test_targets, class_labels, perf_metrics)

    # Ensemble all agents, unweighted
    ensemble_all = distributed_models.GEC_main(
            gec_type,
            agent_predict_prob,
            agent_predict_class,
            test_targets, class_labels, perf_metrics)

    # %% Return average agent and ensemble results
    return central_results, \
        avg_agent_results, avg_major_c0_results, avg_major_c1_results, \
        ensemble_all, ensemble_major_c0, ensemble_major_c1


# %%
def distributed_XGB(
        XGB_param,
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
    agent_XGBs = train_agents_XGB(
            XGB_param,
            num_agents, agent_features, agent_targets,
            )

    # %% Individual agent Test predictions
    # Probabilities and classifications

    # Test features
    agent_predict_prob, agent_predict_class = individual_agent_predictions_XGB(
                agent_XGBs, num_agents,
                test_features, classification_threshold=0.5
                )

    # Scrambled Test features
    scram_predict_prob, scram_predict_class = individual_agent_predictions_XGB(
                agent_XGBs, num_agents,
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
    return agent_XGBs, avg_agent_results, scram_avg_agent_results, \
        agent_predict_prob, scram_predict_prob, \
        agent_predict_class, scram_predict_class


# %%
def train_agents_XGB(
        XGB_param,
        num_agents, agent_features, agent_targets
        ):
    '''
    A function to train a set of indepdendent distributed agents that
    each have their own independent training dataset.
    XGB_param can be specified, but
    for now, the code doesn't do anything with it

    Returns a list of trained XGBs, one per agent
    '''

    # %% Initialize list to store the trained weights
    agent_XGBs = [
            xgboost.XGBClassifier()
            for i in range(0, num_agents)
            ]

    # %% Train agents
    # np.ravel(targets) is necessary because RF asks for it that way (shrug)
    for n_agent in range(0, num_agents):
        agent_XGBs[n_agent].fit(
                agent_features[n_agent],
                agent_targets[n_agent])

    # %% Return trained random forests for all agents
    return agent_XGBs


# %%
def individual_agent_predictions_XGB(
        agent_XGBs, num_agents,
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
             model_predictions_XGB(agent_XGBs[n_agent], features)

    # %% Return predicted probabilities and classifications
    return agent_predict_prob, agent_predict_class


# %%
def model_predictions_XGB(XGB_model, features):
    """
    Predict probabilities and classifications from XGBoost model
    The class predictions use XGB.predict, and are cool
    The probabilities are hacky and limited to binary
        This is because I have to figure out which column of probabilities
        correspond to the classifications made with XGB.predict().
        This is becaise I can't pin down necessarily which column corresponds
        to the predictions of XGB.predict(), as this seems to vary sometimes

        What I essentially do then manually check which column to use
    """

    # Make normal XGB predictions
    predict_class = XGB_model.predict(features)

    # Make "probability" predictions. The output has 2 columns for binary
    xgb_proba = XGB_model.predict_proba(features)

    # Make sure these probability predictions are within the normal 0-1 range,
    # and sum to 1
    # Get row sum
    row_sum = np.sum(xgb_proba, axis=1)
    # Scale each row by its sum
    for i in range(len(row_sum)):
        xgb_proba[i, :] = xgb_proba[i, :] / row_sum[i]

    # Now figure out which column of probabilities corresponds to the
    # classification. To do this, classify the first column of the first row
    # of predicted probabilities, and compare this to the first row of
    # the classifications. Using this, assign the probabilities

    # Calculate the classifications based on the second column
    xgb_proba_class1 = models_train_test.classify_prob(xgb_proba[:, 1])
    correspondance = sklearn.metrics.accuracy_score(
            predict_class, xgb_proba_class1)
    if correspondance >= 0.5:
        # Case where second column is representative of probabilities
        # NB: Modification so if it is exactly 0.5, defaults to second column
        predict_prob = xgb_proba[:, 1]
    elif correspondance < 0.5:
        # Case where first column is representative of probabilities
        predict_prob = xgb_proba[:, 0]
    else:
        assert False, "ERROR in predicted probabilities from XGB"

    return predict_prob, predict_class
