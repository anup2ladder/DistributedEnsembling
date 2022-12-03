#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:26:14 2019
@author: anupt

A set of functions to train neural nets with distributed learning
"""


# %% Import modules
import numpy as np

from local_library import models_train_test as mtt
from local_library import initialize_outputs as io


# %%
def individual_agent_performance(
        targets, num_agents,
        class_labels, perf_metrics,
        agent_predict_prob, agent_predict_class,
        ):
    """
    Calculate the class-wise and performance of each individual agent
    using predicted probabilities and classification, vs prediction targets
    """

    # %% Initialize agentclass-wise and metric-wise template to store results
    # Each individual agent
    # A list of floats with len()=num_agents per class/metric
    indv_agent_results = io.keywise_template(
            io.keywise_template([0.]*num_agents, perf_metrics),
            class_labels
            )

    # %% Calculate each individual agent's performance
    for n_agent in range(0, num_agents):
        # Get class-wise performance metrics for each agent
        result = mtt.model_classwise_results(
                targets,
                agent_predict_prob[n_agent], agent_predict_class[n_agent],
                class_labels, perf_metrics,
            )

        # %% Assign the class-wise performance metrics for the agent
        for j_class in class_labels.keys():
            for k_perf in perf_metrics:
                indv_agent_results[j_class][k_perf][n_agent] = \
                    result[j_class][k_perf]

    # %% Return prediction performance of each individual agent
    return indv_agent_results


# %%
def average_agent_performance(
        num_agents, class_labels, perf_metrics, indv_agent_results
        ):
    """
    Get the class-wise average performance of the agents as a linear average
    of individual agent performances
    individual agent performances
    """

    # %% Initialize class-wise and metric-wise template to store results
    # Average agent - A single float per class/metric
    avg_agent_results = io.oneshot_metrics(class_labels, perf_metrics)

    # %% Calculate average performance of individual agents
    # Class-wise
    for j_class in class_labels.keys():
        # Metric-wise
        for k_perf in perf_metrics.keys():
            avg_agent_results[j_class][k_perf] = np.mean(
                    indv_agent_results[j_class][k_perf]
                    )

    # %% Return the averaged results
    return avg_agent_results


# %%
def average_large_small_performance(
        agent_predict_prob, agent_predict_class,
        num_agents_dict, targets, class_labels, perf_metrics
        ):
    """Get average agent results for large and small agents"""
    # %%
    agent_keys = ['large', 'small']  # This order matters and must be kept

    # Dictionary to store output
    indv_results = dict()
    avg_results = dict()

    # Track where in the list to get agent predicted probs and classes from
    i_start = 0
    i_end = 0

    for k in agent_keys:
        i_end += num_agents_dict[k]

        indv_results[k] = individual_agent_performance(
                targets, num_agents_dict[k],
                class_labels, perf_metrics,
                agent_predict_prob[i_start:i_end],
                agent_predict_class[i_start:i_end]
                )

        avg_results[k] = average_agent_performance(
                num_agents_dict[k], class_labels, perf_metrics,
                indv_results[k])

        i_start += num_agents_dict[k]
        # Fix this from here

    return avg_results['large'], avg_results['small']


# %%
def GEC_main(
        gec_type,
        agent_predict_prob, agent_predict_class,
        targets, class_labels, perf_metrics):
    """
    Main Global Ensemble Classifier (GEC) function
    Requires a specification of type of GEC to do (gec_type) as string
        Accepted inputs:
            1. "average_probability"
            2. "majority_vote"
    Takes the predicted agent probabilities and classes and performs
    the specified GEC
    """

    # A set of if statements to decide which GEC to do
    # And then get the GEC ensemble predicted probability and class
    if gec_type == "average_probability":
        ensemble_prob, ensemble_class = global_ensemble_classifier(
                agent_predict_prob)
    elif gec_type == "majority_vote":
        ensemble_prob, ensemble_class = global_ensemble_classifier(
                agent_predict_class)
    else:
        assert False, "gec_type is incorrectly specified!"

    # %% Measure performance of the GEC
    ensemble_results = mtt.model_classwise_results(
            targets, ensemble_prob, ensemble_class,
            class_labels, perf_metrics)

    return ensemble_results

# %%
def GEC_weighted(
        gec_type, weights,
        agent_predict_prob, agent_predict_class,
        targets, class_labels, perf_metrics):
    """
    Main Global Ensemble Classifier (GEC) function
    Requires a specification of type of GEC to do (gec_type) as string
        Accepted inputs:
            1. "average_probability"
            2. "majority_vote"
    Takes the predicted agent probabilities and classes and performs
    the specified GEC
    """

    # A set of if statements to decide which GEC to do
    # And then get the GEC ensemble predicted probability and class
    if gec_type == "average_probability":
        ensemble_prob, ensemble_class = global_ensemble_classifier(
                agent_predict_prob, weights)
    elif gec_type == "majority_vote":
        ensemble_prob, ensemble_class = global_ensemble_classifier(
                agent_predict_class, weights)
    else:
        assert False, "gec_type is incorrectly specified!"

    # %% Measure performance of the GEC
    ensemble_results = mtt.model_classwise_results(
            targets, ensemble_prob, ensemble_class,
            class_labels, perf_metrics)

    return ensemble_results


# %%
def global_ensemble_classifier(agent_predictions, weights=None):
    """
    Simple function to ensemble predictions from a list of agent predictions
    and classify these predictions

    The input is what determines what is being ensembled
    For average_probability, the input are the predicted probabilities
    For majority_vote, the input are the predicted classes

    Updated function takes weighted average based on a list of weights
    An easy weight list is the number of examples each agent is trained on
    Default = None, so it is a simple arthimetic mean
    """
    ensemble_predicted_prob = np.average(
            agent_predictions,
            axis=0,
            weights=weights)

    # %% Classify the ensemble's predictions
    ensemble_predicted_class = mtt.classify_prob(ensemble_predicted_prob)

    return ensemble_predicted_prob, ensemble_predicted_class
