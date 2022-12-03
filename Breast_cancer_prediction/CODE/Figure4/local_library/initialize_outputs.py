#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:55:37 2019

@author: anupt
"""

# %% Import modules
import copy
import numpy as np


# %%
def keywise_template(template, keys_from_dict):
    '''A function to copy a template data structure into a new dictionary
    that stores it based on the keys from an given dictionary

    For example, the dictionary could be a list of models (the keys)
    or a list of classes (the keys)
    create a class-wise template from
    of a given template'''

    template_keywise = {}

    for k_key in keys_from_dict.keys():
        template_keywise[k_key] = copy.deepcopy(template)
    return template_keywise


# %%
def oneshot_metrics(class_labels, perf_metrics):
    '''
    A function to store results of a single model, single iterations results
    This is not a matrix or a list, but a single value (float)
    Stored in a nested fashion as dict['class']['metric']
    '''

    # Create dictionary to store a float for each metric
    template_oneshot = {}
    for m_metric in perf_metrics.keys():
        template_oneshot[m_metric] = 0.

    # Create the class-wise template
    classwise_oneshot = keywise_template(template_oneshot, class_labels)

    return classwise_oneshot


# %%
def iteration_metrics(num_iterations, model_titles, dist_titles,
                      class_labels, perf_metrics):
    '''A function to initialize a dictionary to store
    the iteration-by-iteration metrics of a model'''

    # Create a template for a single model and classes iteration results
    template_iteration_metrics = {}
    for m_metric in perf_metrics.keys():
        # Loop through the metrics and create a list of placeholder zeros
        # the length of the number of iterations
        template_iteration_metrics[m_metric] = [0.]*num_iterations

    # Now, create a model and classwise list to store the iteration results
    # For this, I will call the keywise_template in a nested fashion
    # The inner call uses the class_labels to create dictionary of copies of
    # the template_results_matrix using the keys from class_labels
    # The outer (first) call then copies this to each model, using
    # the keys from model_titles
    modelclasswise_iteration_metrics = keywise_template(
            keywise_template(
                    keywise_template(template_iteration_metrics, class_labels),
                    dist_titles
                    ),
            model_titles
            )
    return modelclasswise_iteration_metrics


# %%
def iteration_metrics_array(num_iterations, num_cond,
                            model_titles, dist_titles,
                            class_labels, perf_metrics):
    '''A function to initialize a dictionary to store
    the iteration-by-iteration metrics of a model'''

    # Create a template for a single model and classes iteration results
    template_iteration_metrics = {}
    for m_metric in perf_metrics.keys():
        # Loop through the metrics and create a list of placeholder zeros
        # the length of the number of iterations
        template_iteration_metrics[m_metric] = np.zeros(
                (num_iterations, num_cond)
                )

    # Now, create a model and classwise list to store the iteration results
    # For this, I will call the keywise_template in a nested fashion
    # The inner call uses the class_labels to create dictionary of copies of
    # the template_results_matrix using the keys from class_labels
    # The outer (first) call then copies this to each model, using
    # the keys from model_titles
    modelclasswise_iteration_metrics = keywise_template(
            keywise_template(
                    keywise_template(template_iteration_metrics, class_labels),
                    dist_titles
                    ),
            model_titles
            )
    return modelclasswise_iteration_metrics


# %%
def results_singular(model_titles, class_labels, perf_metrics):
    '''
    Initialize a dictionary to store the
    mean and standard deviation of various metrics
    from a single set of conditions
    '''

    # First create the template to copy, for the list of metrics
    template_summary_metrics = {}
    for m_metric in perf_metrics.keys():
        template_summary_metrics[m_metric] = {'mean': 0, 'stdev': 0}

    # Now, create a model and classwise matrix to store the results
    # For this, I will call the keywise_template in a nested fashion
    # The inner call uses the class_labels to create dictionary of copies of
    # the template_results_matrix using the keys from class_labels
    # The outer (first) call then copies this to each model, using
    # the keys from model_titles
    modelclasswise_summary = keywise_template(
            keywise_template(template_summary_metrics, class_labels),
            model_titles
            )
    return modelclasswise_summary


# %%
def results_matrix(num_rows, num_columns,
                   model_titles, dist_titles, class_labels, perf_metrics):
    '''
    Initialize a model- and class-wise dictionary to store a matrix
    mean and standard deviation of various metrics of a 2 condition
    '''

    # Create the template for a single model and classes results matrix
    template_results_matrix = {}
    for m_metric in perf_metrics.keys():
        # Loop through the metrics and for each, create a dictionary
        # with 'mean' and 'stdev', which are both zero arrays of row x column
        template_results_matrix[m_metric] = {
                'mean': np.zeros([num_rows, num_columns]),
                'stdev': np.zeros([num_rows, num_columns])
        }

    # Now, create a model and classwise matrix to store the results
    # For this, I will call the keywise_template in a nested fashion
    # The inner call uses the class_labels to create dictionary of copies of
    # the template_results_matrix using the keys from class_labels
    # The outer (first) call then copies this to each model, using
    # the keys from model_titles
    modelclasswise_results_matrix = keywise_template(
            keywise_template(
                    keywise_template(template_results_matrix, class_labels),
                    dist_titles
                    ),
            model_titles
            )

    return modelclasswise_results_matrix
