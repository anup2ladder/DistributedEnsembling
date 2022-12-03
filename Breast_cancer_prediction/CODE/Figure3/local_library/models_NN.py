#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:26:14 2019
@author: anupt

A set of functions to train neural nets with distributed learning
"""


# %% Import modules
import copy
import numpy as np
import keras as K
import gc


from local_library import models_train_test
from local_library import distributed_models


# %%
def GEC_NN(
        gec_type,
        model, model_info, opt_config,
        central_features, central_targets,
        num_agents,
        agent_features, agent_targets,
        test_features, test_targets, scram_test_features,
        class_labels, perf_metrics
        ):
    """
    Global ensemble classifier using Neural Nets to train agents
    Requires a specification of type of GEC to do (gec_type) as string
        Accepted inputs:
            1. "average_probability"
            2. "majority_vote"
    """

    # %% Perform central training
    # Get all 7 returned values
    central_weights, central_results, _ignore_scram, \
        central_predict_prob, _ignore_scram, \
        central_predict_class, _ignore_scram = distributed_NN(
                model, model_info, opt_config, model_info['GEC_init_same'],
                num_agents['central'], central_features, central_targets,
                test_features, test_targets, scram_test_features,
                class_labels, perf_metrics
                )

    # %% Perform distributed training
    # Get all 7 returned values
    agent_weights, avg_agent_results, scram_avg_agent_results, \
        agent_predict_prob, scram_predict_prob, \
        agent_predict_class, scram_predict_class = distributed_NN(
                model, model_info, opt_config, model_info['GEC_init_same'],
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
        avg_agent_results, \
        ensemble_all


# %%
def distributed_NN(
        model, model_info, opt_config, weight_initialization,
        num_agents, agent_features, agent_targets,
        test_features, test_targets, scram_test_features,
        class_labels, perf_metrics
        ):
    """
    Perform distributed training of a model using independent datasets
    Returns a list of trained weights, average agent performance (+scram),
    predicted probalilities and classifications (+scram)
    """

    # %% Set the weights

    # If you want all agents initialized with the same starting weight
    if weight_initialization is True:
        # reset the model weights and get them back
        # N.B. This will both get a list of newly initialized weights
        # and reset the weights in the global model
        unified_initial_weights = reset_weights(model)

    # If you want all the agents starting weights independently initialized
    elif weight_initialization is False:
        # Set to False, which will cause the train_agents function
        # to set unique initial weights for each agent
        unified_initial_weights = False

    # To catch errors in this parameter
    else:
        # For now this is broken, but this should catch the case
        # Where the function is passed in a set of weights
        assert False, "Weight initialization parameter invalid"
        unified_initial_weights = weight_initialization

    # %% Train the SVM agents
    agent_weights = train_agents_NN(
            model, model_info, opt_config, unified_initial_weights,
            num_agents, agent_features, agent_targets,
            valid_features=None, valid_targets=None,
            )

    # %% Individual agent Test predictions
    # Probabilities and classifications

    # Test features
    agent_predict_prob, agent_predict_class = individual_agent_predictions_NN(
                model, agent_weights, num_agents,
                test_features, classification_threshold=0.5
                )

    # Scrambled Test features
    scram_predict_prob, scram_predict_class = individual_agent_predictions_NN(
                model, agent_weights, num_agents,
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
    return agent_weights, avg_agent_results, scram_avg_agent_results, \
        agent_predict_prob, scram_predict_prob, \
        agent_predict_class, scram_predict_class


# %%
def train_agents_NN(
        model, model_info, opt_config, weights_given,
        num_agents, agent_features, agent_targets,
        valid_features, valid_targets
        ):
    '''
    A function to train a set of indepdendent distributed agents with
    neural nets that each have their own independent training dataset.

    The weights can be given (weights_given) so that it is same for all agents
    or set as "False", and thus uniquely initialized for each agent

    Returns a list with the trained weights for each agent
    '''

    # %% Initialize list to store the trained weights
    agent_weights = [[] for i in range(0, num_agents)]

    # %% Train agents
    for n_agent in range(0, num_agents):

        # %% Set the initial weights for the agent
        if weights_given is False:
            # %% Each agent has unique initial weights
            # If no initial weights given
            # i.e. weights_given=False or not specified
            # Then initial weights for each agent are randomly initialized
            initial_weights = reset_weights(model)
        else:
            # %% Each agent has same initial weights
            # Initial weights are passed in, then set each agent's
            # initial training weigots to this passed in value
            initial_weights = weights_given

        # %% Train the agent
        # Just in case, set the model to the current initial weights
        # This is redundant, but just in case
#        model.set_weights(initial_weights)

        # Now train the agents using initial_weights specified at some point
        agent_weights[n_agent] = train_model_NN(
                model, model_info, opt_config, initial_weights,
                agent_features[n_agent], agent_targets[n_agent],
                valid_features, valid_targets
                )

    # %% Return trained weights for all agents
    return agent_weights


# %%
def individual_agent_predictions_NN(
        model, agent_weights, num_agents,
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
             model_predictions_NN(model, agent_weights[n_agent], features)

    # %% Return predicted probabilities and classifications
    return agent_predict_prob, agent_predict_class


# %%
def model_predictions_NN(model, test_weights, features,
                         classification_threshold=0.5
                         ):
    """
    A function to make predictions on a model using test weights on given
    features, and returns predicted probabilities and classes using
    a classification threshold (default=0.5)
    """
    # Set Keras session
    sess = K.backend.get_session()
    K.backend.tensorflow_backend.set_session(sess)

    # Set the model weights
    model.set_weights(test_weights)

    # Predict probabilities of model, with test_weights, from features
    predict_prob = model.predict(features, verbose=0)

    # Classify these predicted probabilities using classification_threshold
    # Default value for classification_threshold is 0.5 (50%)
    predict_class = models_train_test.classify_prob(
            predict_prob, classification_threshold
            )

    # Return predictions
    return predict_prob, predict_class












# %% Necessary extra code for keras

# %%
def define_model(model_info, num_features):
    '''
    Define a sequential neural network model.
    Do not compile or train

    The model dimensions are taken in from model_info
    and dynamically generated.
    A minimum of 1 hidden layer is required.
    '''

    # Set Keras session
    sess = K.backend.get_session()
    K.backend.tensorflow_backend.set_session(sess)

    # Input layer
    input_layer = K.layers.Input(shape=(num_features,), name='Input')

    # First hidden layer
    hidden_layer = K.layers.Dense(
            units=model_info['neurons_per_hidden'],
            activation=model_info['activation_hidden'],
            kernel_regularizer=K.regularizers.l1(
                    model_info['l1reg_lambda']
                    ),
            name='Hidden_1__'+model_info['activation_hidden']
            )(input_layer)

    # Create the remaining n+1 hidden layers
    for i in range(1, model_info['number_hidden_layers']):
        hidden_layer = K.layers.Dense(
                units=model_info['neurons_per_hidden'],
                activation=model_info['activation_hidden'],
                kernel_regularizer=K.regularizers.l1(
                        model_info['l1reg_lambda']
                        ),
                name='Hidden_'+str(i+1)+'__'+model_info['activation_hidden']
                )(hidden_layer)

    # Output/classification layer
    output_layer = K.layers.Dense(
            units=1,
            activation=model_info['activation_output'],
            name='Output__'+model_info['activation_output']
            )(hidden_layer)

    # Instantiate the model
    model = K.models.Model(
            inputs=input_layer,
            outputs=output_layer
            )

    # Compile the model
    model.compile(loss=model_info['loss'],
                  optimizer=model_info['optimizer'],
                  metrics=model_info['model_metrics']
                  )

    return model


# %%
def train_model_NN(model, model_info, opt_config, initial_weights,
                   train_features, train_targets,
                   valid_features, valid_targets):
    '''
    Train an instantiated model using a pre-determined set of weights.
    Returns the trained weights
    '''

    # %% Set Keras session
    sess = K.backend.get_session()
    K.backend.tensorflow_backend.set_session(sess)

    # Constantly re-compiling the model seems to fuck up the training time
    # Creates a new gunk of compiled models?
    # Yet I should find a way to reset the model comipiler state to
    # the initial state
    # Re-compile the model
#    model.reset_states()
#    model.compile(loss=model_info['loss'],
#                  optimizer=model_info['optimizer'],
#                  metrics=model_info['model_metrics']
#                  )

    # %%Set the weights
    # This requires an initial set of weights to be passed in
    # Thus, the onus is on the caller to pass in weights
    # Whether it be the models current weights, or another model's weights
    # OR a newly initialized set of weights
    model.set_weights(initial_weights)

    # %% reset optimizer weights
    opt_weights = model.optimizer.get_weights()
    # If the model has been trained before, then reset optimizer weights
    if len(opt_weights) > 0:
        # Set iterations to 0
        opt_weights[0] = np.int64(0)
        # reset rest of the weights to 0
        for i in range(1, len(opt_weights)):
            opt_weights[i] = np.zeros(
                    opt_weights[i].shape,
                    dtype='float32'
                    )
    model.optimizer.set_weights(opt_weights)

    # %% Set optimizer vavlue to initial config, JIC
    # Dirty
    K.backend.set_value(model.optimizer.lr, opt_config['lr'])
    K.backend.set_value(model.optimizer.beta_1, opt_config['beta_1'])
    K.backend.set_value(model.optimizer.beta_2, opt_config['beta_2'])
    K.backend.set_value(model.optimizer.decay, opt_config['decay'])
#    K.backend.set_value(model.optimizer.epsilon, opt_config['epsilon'])
#    K.backend.set_value(model.optimizer.amsgrad, opt_config['amsgrad'])

    # %% If validation data is not given, then set this to None
    # This happens if either valid_features or valid_targets is set to None
    if valid_features is None or valid_targets is None:
        valid_data = None
    else:
        valid_data = (valid_features, valid_targets)

    # %% Train the model
    history = model.fit(
            train_features, train_targets,
            epochs=model_info['epochs'],
            verbose=model_info['verbose'],
            batch_size=model_info['batch_size'],
            validation_data=valid_data,
            shuffle=model_info['shuffle_batches'],
            # initial_epoch=0
            )

    # %% Retrieve the trained weights
    trained_weights = model.get_weights()

    # %% Clear the history file, just in case
    del history
    gc.collect()

    # %% return trained_weights
    return trained_weights


# %%
def reset_weights(model):
    # Set Keras session
    sess = K.backend.get_session()
    K.backend.tensorflow_backend.set_session(sess)

    # This will use whatever initializer the model has been compiled for
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=sess)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=sess)

    weights_to_return = model.get_weights()

    return weights_to_return




# %% Legacy code for Unity

# %%
def Unity(
        model, model_info, opt_config,
        num_agents, agent_features, agent_targets,
        test_features, test_targets, scram_test_features,
        class_labels, perf_metrics
        ):
    """
    Unified model with linear average of weights after end of training
    """

    # %% Perform distributed training
    # We actually only need the first 3 results [0:3]
    # As we only need the agent weights and average agent results
    # IF unity_init_same = True: all the agents have same starting weights
    # AND this starting weight is randomly initialized (reset) before training
    # IF unity_init_same = False: all agents have different starting weights
    # AND these are randomly initialized (reset) before training
    # If a set of weights are passed in instead of a True/False value:
    # THEN the models are trained with the same starting weights, which are
    # the passed in weights rather than randomly initialized weights
    agent_weights, avg_agent_results, scram_avg_agent_results, \
        agent_predict_prob, scram_predict_prob, \
        agent_predict_class, scram_predict_class = distributed_NN(
                model, model_info, opt_config, model_info['unity_init_same'],
                num_agents, agent_features, agent_targets,
                test_features, test_targets, scram_test_features,
                class_labels, perf_metrics
            )

    # %% Unify the model
    unity_weights = average_agent_weights(agent_weights, num_agents)

    # %% Make predictions with Unity

    # Validation features
    unity_predict_prob, unity_predict_class = model_predictions_NN(
            model, unity_weights, test_features
            )

    # Scrambled validation features
    scraminity_predict_prob, scraminity_predict_class = model_predictions_NN(
            model, unity_weights, scram_test_features
            )

    # %% Calculate Unity's prediction performance

    # Validation features
    unity_results = models_train_test.model_classwise_results(
            test_targets, unity_predict_prob, unity_predict_class,
            class_labels, perf_metrics,
            )

    # Scrambled Validation features
    scraminity_results = models_train_test.model_classwise_results(
            test_targets, scraminity_predict_prob, scraminity_predict_class,
            class_labels, perf_metrics,
            )

    # %% Return average agent and Unity results
    return avg_agent_results, scram_avg_agent_results, \
        unity_results, scraminity_results


# %%
def average_agent_weights(
        agent_weights, num_agents
        ):
    """
    Linear unweighted average of weights of agents
    I currently don't know a smart way to do this with numpy
    So I am literally going to add all the weights and then
    divided by the number of agents
    """

    # %% Initialize avg_weights list
    avg_weights = list()

    # %%  Loop through the weights by each position in the weights list
    for w_weights in range(0, len(agent_weights[0])):
        # %% Initialize sum_weights array equal to this layer's weights
        sum_weights = np.zeros(agent_weights[0][w_weights].shape)

        # %% Sum this layer's weights across agents
        for n_agent in range(0, num_agents):
            sum_weights += agent_weights[n_agent][w_weights]

        # %% Calculate the average weight from the sum
        avg_weights.append(sum_weights / num_agents)

    # Return the average weights
    return avg_weights





# %% OLD CODE
"""OUTDATED OLD CODE"""
## %%
#def distributed_training(
#        model, model_info, opt_config,
#        num_agents, weight_initialization,
#        class_labels, perf_metrics,
#        agent_features, agent_targets,
#        valid_features, valid_targets,
#        scram_valid_features=False
#        ):
#    """
#    Perform distributed training of a model using independent datasets
#    Returns a list of trained weights, average agent performance (+scram),
#    predicted probalilities and classifications (+scram)
#    """
#
#    # %% Set the weights
#
#    # If you want all agents initialized with the same starting weight
#    if weight_initialization is True:
#        # reset the model weights and get them back
#        # N.B. This will both get a list of newly initialized weights
#        # and reset the weights in the global model
#        unified_initial_weights = reset_weights(model)
#
#    # If you want all the agents starting weights independently initialized
#    elif weight_initialization is False:
#        # Set to False, which will cause the train_agents function
#        # to set unique initial weights for each agent
#        unified_initial_weights = False
#
#    # To catch errors in this parameter
#    else:
#        # For now this is broken, but this should catch the case
#        # Where the function is passed in a set of weights
#        assert False, "Weight initialization parameter invalid"
#        unified_initial_weights = weight_initialization
#
#    # %% Train the agents
#    agent_weights = train_agents(
#            model, model_info, opt_config,
#            num_agents,
#            agent_features, agent_targets,
#            valid_features, valid_targets,
#            weights_given=unified_initial_weights
#            )
#
#    # %% Individual agent Validation predictions
#    # Probabilities and classifications
#
#    # Validation features
#    agent_predict_prob, agent_predict_class = individual_agent_predictions(
#                model, agent_weights, num_agents,
#                valid_features, classification_threshold=0.5
#                )
#
#    # Scrambled Validation features
#    scram_predict_prob, scram_predict_class = individual_agent_predictions(
#                model, agent_weights, num_agents,
#                scram_valid_features, classification_threshold=0.5
#                )
#
#    # %% Individual agent Validation perforamnce
#    # List of class-wise performance metrics for each agent
#
#    # Predictions from validation features
#    indv_agent_results = individual_agent_performance(
#            valid_targets, num_agents,
#            class_labels, perf_metrics,
#            agent_predict_prob, agent_predict_class
#            )
#
#    # Predictions from scrambled validation features
#    indv_scram_results = individual_agent_performance(
#            valid_targets, num_agents,
#            class_labels, perf_metrics,
#            scram_predict_prob, scram_predict_class
#            )
#
#    # %% Average validation performance of individual agents
#    # Linear average of individual agent performance
#
#    # Predictions from validation features
#    avg_agent_results = average_individual_agent_performance(
#            num_agents, class_labels, perf_metrics, indv_agent_results
#            )
#
#    # Predictions from scrambled validation features
#    scram_avg_agent_results = average_individual_agent_performance(
#            num_agents, class_labels, perf_metrics, indv_scram_results
#            )
#
#    # %% Return agent weights, average performances and individual predictions
#    # A long tuple of length = 7
#    # For GEC majority vote, you would want to capture all of these
#    # For GEC average probability, only first 5 are sufficient i.e. [0:5]
#    # For Unity, only interesetd in first 3 (i.e. [0:3])
#    return agent_weights, avg_agent_results, scram_avg_agent_results, \
#        agent_predict_prob, scram_predict_prob, \
#        agent_predict_class, scram_predict_class
#
#
## %% train_agents
#def train_agents(
#        model, model_info, opt_config,
#        num_agents,
#        agent_features, agent_targets,
#        valid_features, valid_targets,
#        weights_given=False
#        ):
#    '''
#    A function to train a set of indepdendent distributed agents that
#    each have their own independent training dataset.
#
#    Returns a list with the trained weights for each agent
#    '''
#
#    # %% Initialize list to store the trained weights
#    agent_weights = [[] for i in range(0, num_agents)]
#
#    # %% Train agents
#    for n_agent in range(0, num_agents):
#
#        # %% Set the initial weights for the agent
#        if weights_given is False:
#            # %% Each agent has unique initial weights
#            # If no initial weights given
#            # i.e. weights_given=False or not specified
#            # Then initial weights for each agent are randomly initialized
#            initial_weights = reset_weights(model)
#        else:
#            # %% Each agent has same initial weights
#            # Initial weights are passed in, then set each agent's
#            # initial training weigots to this passed in value
#            initial_weights = weights_given
#
#        # %% Train the agent
#        # Just in case, set the model to the current initial weights
#        # This is redundant, but just in case
##        model.set_weights(initial_weights)
#
#        # Now train the agents using initial_weights specified at some point
#        agent_weights[n_agent] = train_model(
#                model, initial_weights,
#                model_info, opt_config,
#                agent_features[n_agent], agent_targets[n_agent],
#                valid_features, valid_targets
#                )
#
#    # %% Return trained weights for all agents
#    return agent_weights
#
#
## %% individual_agent_predictions
#def individual_agent_predictions(
#        model, agent_weights, num_agents,
#        features, classification_threshold=0.5
#        ):
#    # Make predictions for each agent
#    # Calculate probabilities and classifications using the inputted features
#    # whether they are training or validation features, or scrambled features
#
#    # %% Initialize lists to store the predictions
#    agent_predict_prob = [[] for i in range(0, num_agents)]
#    agent_predict_class = copy.deepcopy(agent_predict_prob)
#
#    # %% Make predictions
#    for n_agent in range(0, num_agents):
#        agent_predict_prob[n_agent], agent_predict_class[n_agent] = \
#            model_predictions(
#                    model, agent_weights[n_agent], features
#                    )
#
#    # %% Return predicted probabilities and classifications
#    return agent_predict_prob, agent_predict_class
#
#
## %% individual_agent_performance
#def individual_agent_performance(
#        targets, num_agents,
#        class_labels, perf_metrics,
#        agent_predict_prob, agent_predict_class,
#        ):
#    """
#    Calculate the class-wise and performance of each individual agent
#    using predicted probabilities and classification, vs prediction targets
#    """
#
#    # %% Initialize agentclass-wise and metric-wise template to store results
#    # Each individual agent
#    # A list of floats with len()=num_agents per class/metric
#    indv_agent_results = io.keywise_template(
#            io.keywise_template([0.]*num_agents, perf_metrics),
#            class_labels
#            )
#
#    # %% Calculate each individual agent's performance
#    for n_agent in range(0, num_agents):
#        # Get class-wise performance metrics for each agent
#        result = mtt.model_classwise_results(
#                targets, class_labels, perf_metrics,
#                agent_predict_prob[n_agent],
#                agent_predict_class[n_agent]
#            )
#
#        # %% Assign the class-wise performance metrics for the agent
#        for j_class in class_labels.keys():
#            for k_perf in perf_metrics:
#                indv_agent_results[j_class][k_perf][n_agent] = \
#                    result[j_class][k_perf]
#
#    # %% Return prediction performance of each individual agent
#    return indv_agent_results
#
#
## %% average_individual_agent_performance
#def average_individual_agent_performance(
#        num_agents, class_labels, perf_metrics, indv_agent_results
#        ):
#    """
#    Get the class-wise average performance of the agents as a linear average
#    of individual agent performances
#    individual agent performances
#    """
#
#    # %% Initialize class-wise and metric-wise template to store results
#    # Average agent - A single float per class/metric
#    avg_agent_results = io.oneshot_metrics(class_labels, perf_metrics)
#
#    # %% Calculate average performance of individual agents
#    # Class-wise
#    for j_class in class_labels.keys():
#        # Metric-wise
#        for k_perf in perf_metrics.keys():
#            avg_agent_results[j_class][k_perf] = np.mean(
#                    indv_agent_results[j_class][k_perf]
#                    )
#
#    # %% Return the averaged results
#    return avg_agent_results
#
#
## %% global_ensemble_classifier
#def global_ensemble_classifier(
#        targets, class_labels, perf_metrics,
#        ensemble_prob
#        ):
#
#    # %% Classify the ensemble's predictions
#    ensemble_class = mtt.classify_prob(ensemble_prob)
#
#    # %% Initialize class-wise and metric-wise template
#    ensemble_results = io.oneshot_metrics(class_labels, perf_metrics)
#
#    # %% Calculate the performance of the predictions against the targets
#    for j_class in class_labels.keys():
#        ensemble_results[j_class]['acc'], \
#            ensemble_results[j_class]['prec'], \
#            ensemble_results[j_class]['recall'], \
#            ensemble_results[j_class]['f1'], \
#            ensemble_results[j_class]['mcc'], \
#            ensemble_results[j_class]['roc_auc'], \
#            ensemble_results[j_class]['PR_auc'], \
#            = mtt.measure_prediction_performance(
#                    targets,
#                    ensemble_prob,
#                    ensemble_class,
#                    avgtype=class_labels[j_class]['avgtype'],
#                    label=class_labels[j_class]['label']
#                    )
#
#    # %% Return ensemble results
#    return ensemble_results
#
#
## %% global_ensemble_predictor
#def global_ensemble_predictor(
#        agent_predictions
#        ):
#    """
#    Simple function to ensemble predictions from a list of agent predictions
#
#    The input is what determines what is being ensembled
#    For average_probability, the input are the predicted probabilities
#    For majority_vote, the input are the predicted classes
#    """
#    ensemble_predicted_prob = np.mean(agent_predictions, axis=0)
#
#    return ensemble_predicted_prob
#
#
## %% GEC_average_probability
#def GEC_average_probability(
#        model, model_info, opt_config,
#        class_labels, perf_metrics,
#        num_agents,
#        agent_features, agent_targets,
#        valid_features, valid_targets,
#        scram_valid_features
#        ):
#    """
#    Global ensemble classifier with average probability ensembling
#    """
#
#    # %% Perform distributed training
#    # Only ask for first 5 values for return
#    # As we only need the predicted probabilities
#    agent_weights, avg_agent_results, scram_avg_agent_results, \
#        agent_predict_prob, scram_predict_prob \
#        = distributed_training(
#                model, model_info, opt_config,
#                num_agents, model_info['GEC_init_same'],
#                class_labels, perf_metrics,
#                agent_features, agent_targets,
#                valid_features, valid_targets,
#                scram_valid_features
#            )[0:5]
#
#    # %% Ensemble the probabilities with average_probability
#    # Take the average probability of agent predictions on each example
#    # and treated this as the ensemble's predicted probabilities
#
#    # Validation predicted probabilites
#    ensemble_avgprob_prob = global_ensemble_predictor(agent_predict_prob)
#
#    # Scrambled predicted probabilites
#    scramble_avgprob_prob = global_ensemble_predictor(scram_predict_prob)
#
#    # %% Get the ensemble performance from the predictions
#    # Classify based on the averaged predicted binary class probability
#
#    # Validation predictions
#    ensemble_results = global_ensemble_classifier(
#            valid_targets, class_labels, perf_metrics,
#            ensemble_avgprob_prob
#            )
#
#    # Scrambled validation predictions
#    scramble_results = global_ensemble_classifier(
#            valid_targets, class_labels, perf_metrics,
#            scramble_avgprob_prob
#            )
#
#    # %% Return average agent and ensemble results
#    return avg_agent_results, scram_avg_agent_results, \
#        ensemble_results, scramble_results
#
#
## %% GEC_majority_vote
#def GEC_majority_vote(
#        model, model_info, opt_config,
#        class_labels, perf_metrics,
#        num_agents,
#        agent_features, agent_targets,
#        valid_features, valid_targets,
#        scram_valid_features
#        ):
#    """
#    Global ensemble classifier with average probability ensembling
#    """
#
#    # %% Perform distributed training
#    # Get all 7 returned values
#    agent_weights, avg_agent_results, scram_avg_agent_results, \
#        agent_predict_prob, scram_predict_prob, \
#        agent_predict_class, scram_predict_class = distributed_training(
#                model, model_info, opt_config,
#                num_agents, model_info['GEC_init_same'],
#                class_labels, perf_metrics,
#                agent_features, agent_targets,
#                valid_features, valid_targets,
#                scram_valid_features
#            )
#
#    # Remove the unused predicted probabilities
#    del agent_predict_prob, scram_predict_prob
#
#    # %% Ensemble the classifications with majority vote
#    # Average the binary classifications and treat these as probabilities
#
#    # Validation predicted probabilites
#    ensemble_majvote_prob = global_ensemble_predictor(agent_predict_class)
#
#    # Scrambled predicted probabilites
#    scramble_majvote_prob = global_ensemble_predictor(scram_predict_class)
#
#    # %% Get the ensemble performance from the predictions
#    # Classify the majority vote using the averaged binary classification
#
#    # Validation predictions
#    ensemble_results = global_ensemble_classifier(
#            valid_targets, class_labels, perf_metrics,
#            ensemble_majvote_prob
#            )
#
#    # Scrambled validation predictions
#    scramble_results = global_ensemble_classifier(
#            valid_targets, class_labels, perf_metrics,
#            scramble_majvote_prob
#            )
#
#    # %% Return average agent and ensemble results
#    return avg_agent_results, scram_avg_agent_results, \
#        ensemble_results, scramble_results
#
#
