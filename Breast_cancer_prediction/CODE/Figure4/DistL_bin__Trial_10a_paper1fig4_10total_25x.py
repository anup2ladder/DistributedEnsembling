#!/usr/bin/env python
# coding: utf-8
'''
General Purpose: Compare different machine learning models on their performance
in distributed learning with global ensemble classification.
Test some different parameters for each model type

Models:
    - Neural nets [nn] with 2 layers and a varying number of neurons per layer
        - 16 neurons per layer
        - 128 neurons per layer
        - 1024 neurons per layer
    - Support vector machines [svm] with different kernels
        - Linear
        - Polynomial degree 2 and 3
        - Gaussian
        - Sigmoid
    - Random Forest [rf] with varying number of trees
        - 1 tree
        - 10 trees
        - 100 trees
        - 1000 trees
        - 10000 trees
    - XGBoost [xgb]. Currently with just base parameters, no changes

Here, care is taken to ensure classes are balanced, so that for the
limiting case (2 examples per agent), there is at least one of each class

    The conditions I test are:
        1. Types of models (rows)
        2. Number of agents / number of examples per pagent (columns)

N.B. This script is set-up to read a file in the "DATA" parent folder
called "DATA_FILENAME.txt" and retrieve filename for the csv datafile
from there. Thus, it doesn't require modification for a specific dataset

You just need to specify the conditions filename and number of iterations
(And in general, you'll just set these and keep it the same across datasets)

Trial 9: Update to Trial 7 that uses a more-conssitent set of conditions
across datasets, for Figure 2 in Paper 1

Also, calculates and returns GEC for both Average Probability (AP)
and Majority Vote (MV)

'''


# %%Import system modules and libraries
# System libraries
import keras as K
from tensorflow.python.client import device_lib
import pandas as pd
import os
import shutil
import datetime
import time
import gc
import warnings

# My libraries for this script
from local_library import manage_keras_memory
from local_library import data_in_out
from local_library import models_train_test
from local_library import assign_the_data
from local_library import initialize_outputs
from local_library import models_dumb
from local_library import models_NN
from local_library import models_SVM
from local_library import models_RF
from local_library import models_XGB
from local_library import plot_results

# %% Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %% Define key run options

# Point to the datafile
data_foldername = "../../DATA"  # File stored in master DATA folder
# datafolder = "./" #File stored in same directory as script
data_filename = data_in_out.get_data_filename(
        'DATA_FILENAME.txt', data_foldername)

# The column identifying the targets to classify
# target_name = 'CLASS'
# FORGET THIS. Just ensure the target is labeled as CLASS

# Point to the CSV defining the conditions
conditions_agents_filename = "conditions_10total_9large.csv"
conditions_foldername = "."  # File stored in same directory as script
# N.B. The number of training examples is defined in the first entry
# I.e. The first line of conditions_agents.csv is central

# Define number of iterations
num_iterations = 25

# Flag whether or not to save the output
save_output = True


# %% Define what models are to be tested

# A dictionary with key names to reference| the models
# and model titles, for printing
model_titles = {
#    'rand_model': 'Random',
#    'major_model': 'Majority Class',
    'nn_low': 'NN (2L x 16N)',
    'nn_med': 'NN (2L x 128N)',
    'nn_high': 'NN (2L x 1024N)',
    'svm_lin': 'SVM (Linear)',
    'svm_poly2': 'SVM (Polynomial 2)',
    'svm_poly3': 'SVM (Polynomial 3)',
    'svm_rbf': 'SVM (Gaussian)',
    'svm_sig': 'SVM (Sigmoid)',
    'rf_1': 'RF (1 trees)',
    'rf_10': 'RF (10 trees)',
    'rf_100': 'RF (100 trees)',
    'rf_1000': 'RF (1000 trees)',
    'rf_10000': 'RF (10000 trees)',
    'xgb_base': 'XGBoost (Default)',
}

# %% Define sub-conditions for each model

sub_titles = {
    'central': "Central",
    'ag_all': "Average All Agents",
    'ag_large': "Average Large Agents",
    'ag_small': "Average Small Agents",
    'GEC_large': "GEC Large Agents",
    'GEC_small': "GEC Small Agents",
    'GEC_all_uw': "GEC All Agents (unweighted)",
    'GEC_all_W': "GEC All Agents (Weighted by # Training)",
}

output_order = ['central', 'ag_all', 'ag_large', 'ag_small',
                'GEC_large', 'GEC_small', 'GEC_all_uw', 'GEC_all_W']


# %% Define classification classes

# A list of strings of classes for metrics, to allow for easy looping
# If non-binary, this would need to be modified
# NEW: import this from the root of "CODE"
class_labels = data_in_out.pickle_in_variable(
        'class_labels.pickle',
        data_foldername)


# %% A list of strings of performance metrics, to allow easy looping
perf_metrics = dict(
    acc='Accuracy',
    prec='Precision',
    recall='Recall',
    f1='F1-score',
    mcc='MCC',
    roc_auc='ROC AUC',
    PR_auc='PR AUC'
)


# %% Define neural net model design and training info

# Store as a dictionary
NN_model_info = dict(
    epochs=100,
    batch_size=64,
    loss='binary_crossentropy',
    optimizer='adam',
    model_metrics=[K.metrics.binary_accuracy],
    l1reg_lambda=0.0001,
    # l1reg_kernel = K.regularizers.l1(l1reg_lambda),
    activation_hidden='relu',
    activation_output='sigmoid',
    number_hidden_layers=2,
    neurons_per_hidden=None,  # Will need to set default value
    verbose=0,
    shuffle_batches=True,
    GEC_init_same=False,
    unity_init_same=True
)

# %% Limit Tensorflow GPU memory use to only the necessary amount
# Initialize Tensorflow/Keras
# sess is the session
sess = manage_keras_memory.initialize(config=manage_keras_memory.limitGPUmem)

# %% Check whether we are using the CPU or GPU
print(device_lib.list_local_devices())


# %% Import the CSV with number of agents to train

# Import the module


# Import and process the conditions
# cond_agents: Number of agents
# cond_trainEXs: Examples per training agent
cond_agents, cond_trainEXs = data_in_out.import_conditions(
        os.path.join(conditions_foldername, conditions_agents_filename)
        )

# %% Define default values for agents and training examples
num_agents = dict(
        central=cond_agents['values'][0],
        total=cond_agents['values'][4],
        large=cond_agents['values'][2],
        small=cond_agents['values'][3]
        )
#num_agents = cond_agents['values'][0]

num_train = dict(
        central=cond_trainEXs['values'][0],
        total=cond_trainEXs['values'][4],
        large=cond_trainEXs['values'][2],
        small=cond_trainEXs['values'][3]
        )

# List of number of training examples for each agent
# As a combination of the large and small
training_guide = [num_train['large']]*num_agents['large'] + \
    [num_train['small']]*num_agents['small']


# %% Import the raw data and retrieve the number of examples and features

# The number of training examples is the first entry in Examples per agent
# I.e. the first entry is for 1 agent AKA CENTRAL
#num_train_total = int(cond_trainEXs['values'][0])
#num_train_class = int(num_train_total/2)


# Import the raw data
raw_data, num_examples, num_columns = data_in_out.import_the_data(
        os.path.join(data_foldername, data_filename)
        )

# Define number of test examples
num_test = num_examples - num_train['total']

# Calculate the number of features
# Here, only 1 of the columns are related to the target
# Thus n-1 are the features
num_features = num_columns - 1

# Get the prevalence of the two classes
prevalence = assign_the_data.get_binary_prevalence(raw_data['CLASS'])


# %% Get ready
#model = models_NN.define_model(
#        NN_model_info, num_features
#        )
## Get the optimizer configuration
#opt_config = model.optimizer.get_config()

# %% Instantiate the neural network models
# LOW
# Set the number of neurons per hidden layer
NN_model_info['neurons_per_hidden'] = 16
# Instantiate the model for the current configuration
model_low = models_NN.define_model(
        NN_model_info, num_features
        )
# Get the optimizer configuration
optimizer_config_low = model_low.optimizer.get_config()

# MEDIUM
# Set the number of neurons per hidden layer
NN_model_info['neurons_per_hidden'] = 128
# Instantiate the model for the current configuration
model_med = models_NN.define_model(
        NN_model_info, num_features
        )
# Get the optimizer configuration
optimizer_config_med = model_med.optimizer.get_config()

# HIGH
# Set the number of neurons per hidden layer
NN_model_info['neurons_per_hidden'] = 1024
# Instantiate the model for the current configuration
model_high = models_NN.define_model(
        NN_model_info, num_features
        )
# Get the optimizer configuration
optimizer_config_high = model_high.optimizer.get_config()


# %% Begin calculations
# Train and evaluate all model types for [iterations] number of iterations

# Record the start time of evaluation
start_time = time.time()
# Now, loop through conditions


# %% Intialize the results variable
# This is stored as a series of nested dictionaries, going by:
# model, class, performance metric, mean or stdev, array (cond1 x cond2)
results = initialize_outputs.results_matrix(
        1, 1,
        model_titles, sub_titles, class_labels, perf_metrics
        )

# %% Initialize dictionary to store results from iterations
#iterations = initialize_outputs.iteration_metrics_array(
#        num_iterations, cond_agents['num'],
#        model_titles, sub_titles,
#        class_labels, perf_metrics
#        )

# %%
iterations = initialize_outputs.iteration_metrics(
        num_iterations,
        model_titles, sub_titles,
        class_labels, perf_metrics
        )

# %% Loop 1: Iterations (Shuffle of main data)
for i_it in range(0, num_iterations):
    # %% Assign the data
    # Randomly assign training and test data
    train_c0, train_c1,\
        test_features, test_targets,\
        scram_test_features = assign_the_data.balanced_shuffle(
                raw_data, num_train['total'], num_test
                )

    # %% Create a central dataset from shuffled training set
    central_features, central_targets = assign_the_data.balanced_central(
            train_c0, train_c1
            )

    # %% Assign agent training datasets
    agent_features, agent_targets = \
        assign_the_data.assign_uneven_balanced_agents(
                num_agents['total'], training_guide,
                train_c0, train_c1
                )

    # %% Loop 2: Now repeatedly test this set of conditions
#    for ag in range(cond_agents['num']):
#        # Set the number of agents
#        num_agents = cond_agents['values'].iloc[ag]
#        # This also requires setting the number of examples per agent
#        num_train_agent = cond_trainEXs['values'].iloc[ag]
#
#        # Assign agent training features and targets
#        agent_features, agent_targets = assign_the_data.assign_balanced_agents(
#                num_agents, num_train_agent,
#                train_c0, train_c1
#                )

#            # Assign the data for unbalanced classification (e.g. 1 example)
#             Randomly assign training and test data
#            train_features, train_targets,\
#                test_features, test_targets,\
#                scram_valid_features = assign_the_data.shuffle_the_data(
#                        raw_data, num_train_total, num_test
#                        )
#
#            # Assign agent training features and targets
#            agent_features, agent_targets = assign_the_data.agents(
#                    num_agents, num_train_agent,
#                    train_features, train_targets
#                    )

    # %% Prepare dict to store results from this iteration
#    print(cond_agents['name']+":", cond_agents['values'].iloc[ag])
    print("Iteration:", i_it+1)
    # this_it stores this specific iterations results (as floats)
    # (Re-)Initialize it
    this_iteration = {}

        # %% Random model
#        local_start = time.time()
#        this_iteration['rand_model'] = {}
#        # Assign dumb predictions to 'agent'
#        this_iteration['rand_model']['agent'] = \
#            models_dumb.random_predictions(
#                    num_test, prevalence, test_targets,
#                    class_labels, perf_metrics,
#                    )
#        # Copy these over to 'GEC'
#        this_iteration['rand_model']['GEC_AP'] = \
#            this_iteration['rand_model']['agent']
#        this_iteration['rand_model']['GEC_MV'] = \
#            this_iteration['rand_model']['agent']
#        local_end = time.time()
#        local_run_sec = round((local_end - local_start), 3)
#        print('\tDumb Random:', local_run_sec, 'sec')

        # %% Majority class model
#        local_start = time.time()
#        this_iteration['major_model'] = {}
#        this_iteration['major_model']['agent'] = \
#            models_dumb.majority_predictions(
#                    num_test, prevalence, test_targets,
#                    class_labels, perf_metrics,
#                    )
#        # Copy these over to 'GEC'
#        this_iteration['major_model']['GEC_AP'] = \
#            this_iteration['major_model']['agent']
#        this_iteration['major_model']['GEC_MV'] = \
#            this_iteration['major_model']['agent']
#        local_end = time.time()
#        local_run_sec = round((local_end - local_start), 3)
#        print('\tDumb Majority:', local_run_sec, 'sec')

    # %% Distributed GEC Neural Nets Low
    local_start = time.time()

    # Fit the model
    output = models_NN.GEC_NN(
            'average_probability',
            model_low, NN_model_info, optimizer_config_low,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'nn_low'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tNeural Net (16 x 2):', local_run_sec, 'sec')

    # %% Distributed GEC Neural Nets Medium
    local_start = time.time()

    # Fit the model
    output = models_NN.GEC_NN(
            'average_probability',
            model_med, NN_model_info, optimizer_config_med,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'nn_med'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tNeural Net (128 x 2):', local_run_sec, 'sec')

    # %% Distributed GEC Neural Nets High
    local_start = time.time()

    # Fit the model
    output = models_NN.GEC_NN(
            'average_probability',
            model_high, NN_model_info, optimizer_config_high,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'nn_high'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tNeural Net (1024 x 2):', local_run_sec, 'sec')

    # %% Distributed GEC SVM Linear
    kernel = 'linear'

    local_start = time.time()

    # Fit the model
    output = models_SVM.GEC_SVM(
            'average_probability',
            kernel,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'svm_lin'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tSVM Linear:', local_run_sec, 'sec')

    # %% Distributed GEC SVM Poly(2)
    kernel = 'poly'

    local_start = time.time()

    # Fit the model
    output = models_SVM.GEC_SVM(
            'average_probability',
            kernel,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            poly_degree=2
            )

    # Assign the output
    key = 'svm_poly2'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tSVM Polynomial (2):', local_run_sec, 'sec')

    # %% Distributed GEC SVM Poly(3)
    kernel = 'poly'

    local_start = time.time()

    # Fit the model
    output = models_SVM.GEC_SVM(
            'average_probability',
            kernel,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            poly_degree=3
            )

    # Assign the output
    key = 'svm_poly3'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tSVM Polynomial (3):', local_run_sec, 'sec')

    # %% Distributed GEC SVM Gaussian
    kernel = 'rbf'

    local_start = time.time()

    # Fit the model
    output = models_SVM.GEC_SVM(
            'average_probability',
            kernel,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'svm_rbf'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tSVM Gaussian:', local_run_sec, 'sec')

    # %% Distributed GEC SVM Sigmoid
    kernel = 'sigmoid'

    local_start = time.time()

    # Fit the model
    output = models_SVM.GEC_SVM(
            'average_probability',
            kernel,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'svm_sig'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tSVM Sigmoid:', local_run_sec, 'sec')

    # %% Distributed GEC RF (1 trees)
    n_trees = 1

    local_start = time.time()

    # Fit the model
    output = models_RF.GEC_RF(
            'average_probability',
            n_trees,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'rf_1'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tRF (1 trees):', local_run_sec, 'sec')

    # %% Distributed GEC RF (10 trees)
    n_trees = 10

    local_start = time.time()

    # Fit the model
    output = models_RF.GEC_RF(
            'average_probability',
            n_trees,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'rf_10'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tRF (10 trees):', local_run_sec, 'sec')

    # %% Distributed GEC RF (500 trees)
    n_trees = 100

    local_start = time.time()

    # Fit the model
    output = models_RF.GEC_RF(
            'average_probability',
            n_trees,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'rf_100'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tRF (100 trees):', local_run_sec, 'sec')

    # %% Distributed GEC RF (1000 trees)
    n_trees = 1000

    local_start = time.time()

    # Fit the model
    output = models_RF.GEC_RF(
            'average_probability',
            n_trees,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'rf_1000'  # The key that should match to model_titles dictionary
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tRF (1000 trees):', local_run_sec, 'sec')

    # %% Distributed GEC RF (10000 trees)
    n_trees = 10000

    local_start = time.time()

    # Fit the model
    output = models_RF.GEC_RF(
            'average_probability',
            n_trees,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'rf_10000'  # The key that should match to model_titles dict
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tRF (10000 trees):', local_run_sec, 'sec')

    # %% Distributed XGB (Base parameters)
    XGB_parameter = None

    local_start = time.time()

    # Fit the model
    output = models_XGB.GEC_XGB(
            'average_probability',
            XGB_parameter,
            central_features, central_targets,
            num_agents, training_guide,
            agent_features, agent_targets,
            test_features, test_targets, scram_test_features,
            class_labels, perf_metrics,
            )

    # Assign the output
    key = 'xgb_base'  # The key that should match to model_titles dict
    this_iteration[key] = assign_the_data.assign_output_dict(
            output, output_order
            )

#        local_start = time.time()
#        this_iteration['xgb_base'] = {}
#        this_iteration['xgb_base']['agent'], _, \
#            this_iteration['xgb_base']['GEC_AP'], _, \
#            this_iteration['xgb_base']['GEC_MV'], _ = \
#            models_XGB.GEC_XGB(
#                    'both AP and MV (this does not do anything)',
#                    XGB_parameter,
#                    num_agents, agent_features, agent_targets,
#                    test_features, test_targets, scram_test_features,
#                    class_labels, perf_metrics
#                )
    local_end = time.time()
    local_run_sec = round((local_end - local_start), 3)
    print('\tXGBoost (Default):', local_run_sec, 'sec')

    # %% Store this iteration's results in "iterations"
    # Per model, class, and performance metric
    # Transfer the result from this_it into the appropriate index
    # in iterations, matching for model, class, and metric
    for m_model in this_iteration.keys():
        for d_dist in this_iteration[m_model].keys():
            for j_class in class_labels.keys():
                for k_perf in perf_metrics:
                    iterations[m_model][d_dist][j_class][k_perf][i_it]\
                        = this_iteration[m_model][d_dist][j_class][k_perf]

    # Delete variables that I am done with
    gc.collect()

    # %% End Loop 2
    # Re-run the same set of shuffled data on a new set of conditions
    # e.g. a different number of agents

# %% End of Loop 1
# Done with this data (train/test) split (i.e. this iteration)
# Re-shuffle the data and repeat the train/testing for all sets of agents


# %% Get summary statistics
# This occurs after finishing out of Iterations
# Per model, class, and performance metric
# Calculate the results (mean and standard deviation)
# And store in the appropriate location the results matrix

# For each number of agents, which are stored in the order they are listed
#for ag in range(cond_agents['num']):
for i_model in iterations.keys():
    for d_dist in iterations[i_model].keys():
        for j_class in class_labels.keys():
            for k_perf in perf_metrics:
                iter_result = models_train_test.mean_stdev_iterations(
                        iterations[i_model][d_dist][j_class][k_perf][:]
                        )
            # iter_result is a dict with keys 'mean' and 'stdev'

                for r in iter_result.keys():
                    results[i_model][d_dist][j_class][k_perf][r][:] = \
                        iter_result[r]

# Delete variables that I am done with
gc.collect()


# %% Garbage the instantiated model design
# Destroy the current model architecture, because
# the next iteration of Loop 1 will use
# a different number of neurons per layer
# Thus, the model architecture is different
manage_keras_memory.reset(manage_keras_memory.limitGPUmem)
del model_med, optimizer_config_med
gc.collect()


# %% End calculations
# Record and print the end time of the evaluation
end_time = time.time()

run_time = dict()
run_time['sec'] = round(end_time - start_time, 2)
run_time['min'] = round(run_time['sec']/60, 2)
run_time['hour'] = round(run_time['sec']/60, 2)

print('\n\nThe run time for the loop was', run_time['min'], 'minutes\n\n')


# %% Summarize the parameters

summary_parameters = [
    ['Number of iterations (N)', num_iterations],
    ['Number of features', num_features],
    ['Number of of training examples', num_train['total']],
    ['Number of test examples', num_test],
    ['', ''],
    ['GEC initialize with same weights', NN_model_info['GEC_init_same']],
    ['Unity initialize with same weights', NN_model_info['unity_init_same']],
    ['Epochs', NN_model_info['epochs']],
    ['Batch size', NN_model_info['batch_size']],
    ['Shuffle Batches', NN_model_info['shuffle_batches']],
    ['Loss model', NN_model_info['loss']],
    ['Optimizer', NN_model_info['optimizer']],
    ['Activation function for hidden:', NN_model_info['activation_hidden']],
    ['Activation function for output:', NN_model_info['activation_output']],
    ['L1 regularization rate', NN_model_info['l1reg_lambda']]
]

summary_parameters = pd.DataFrame(
        summary_parameters, columns=['Parameter', 'Value']
        )


# %% CODE TO STORE RESULTS
# IF save_output = True

if save_output is True:
    # %% Experiment master directory
    # This is the directory to store all the results from this Trial
    # The name is thesame as the folder from which the code is run
    master_relpath = os.path.join(
            "../../RESULTS/",
            os.path.basename(os.getcwd())
            )

    # If this folder doesn't exist, create it
    if os.path.exists(master_relpath) is False:
        os.mkdir(master_relpath)

    # %% Results directory
    # Folder name
    # Format: timestamp + space + conditions_name + space + number iterations
    # timestamp YYYYMMDDHHMMSS
    # E.g. 199912311159 conditions_small 9999x
    folder_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + " " \
        + os.path.splitext(conditions_agents_filename)[0] + " " \
        + str(num_iterations) + "x"

    # Define Relative path to the results folder
    results_folder_relpath = os.path.join(master_relpath, folder_name)

    # Make the folder for the results
    os.mkdir(results_folder_relpath)

    # Inform user
    print("\n\nResult files stored in:\n\t",
          os.path.abspath(results_folder_relpath)
          )

    # %% Save (pickle) the results and additional dictionaries

    # results.pickle
    data_in_out.pickle_out_variable(
            results, 'results.pickle', results_folder_relpath
            )

    # perf_metrics.pickle
    data_in_out.pickle_out_variable(
            perf_metrics, 'perf_metrics.pickle', results_folder_relpath
            )

    # model_titles.pickle
    data_in_out.pickle_out_variable(
            model_titles, 'model_titles.pickle', results_folder_relpath
            )

    # class_labels.pickle
    data_in_out.pickle_out_variable(
            class_labels, 'class_labels.pickle', results_folder_relpath
            )

    # dist_titles.pickle
    data_in_out.pickle_out_variable(
            sub_titles, 'sub_titles.pickle', results_folder_relpath
            )

    # Neural net model info
    data_in_out.pickle_out_variable(
            NN_model_info, 'NN_model_info.pickle', results_folder_relpath
            )

    # %% Save the parameters used
    # parameters.csv
    summary_parameters.to_csv(
            os.path.join(results_folder_relpath, "parameters.csv"),
            index=False
            )

    # %% Save the conditions tested
    # conditions.csv
    shutil.copy2(
            os.path.join(conditions_foldername, conditions_agents_filename),
            os.path.join(results_folder_relpath, 'conditions.csv')
            )

    # %% Call function to plot
    plot_results.main(
            results, cond_agents, cond_trainEXs, output_order,
            model_titles, perf_metrics, class_labels, sub_titles,
            num_train['total'], results_folder_relpath)

# %% Results not saved
elif save_output is False:
    print("/n/n/nNOTE: RESULTS ARE NOT SAVED")

# %% Error in save_output
else:
    assert False, "save_output is improperly defined"

# %% End of script
