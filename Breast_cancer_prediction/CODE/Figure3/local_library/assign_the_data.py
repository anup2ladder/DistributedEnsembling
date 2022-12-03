#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:22:39 2019
@author: anupt

Functions to shuffle and split the data into train and tests sets
"""

# %% Import modules
import sklearn
import pandas as pd


# %% Function to get class distribution in the binary case
def get_binary_prevalence(target_series):
    """
    Determine the prevalence of the binary classes, as a fraction

    Input: Pandas series for the target (binary:0,1)
    Output: prevalence (dictionary), with:
        - c0: Prevalence for class 0
        - c1: Prevalence for class 1
        - majority_class = ID for which is the majority_class
        - majority_prev = prevalence for majority class
    """

    # Count the number of values in class 0 and class 1
    count_c0_all, count_c1_all = target_series.value_counts().sort_index()

    # Calculate the prevalence = # in class / # total
    prevalence = dict()
    prevalence['c0'] = count_c0_all / (count_c1_all+count_c0_all)
    prevalence['c1'] = count_c1_all / (count_c1_all+count_c0_all)

    # Determine which class is majority and assign
    if count_c0_all > count_c1_all:
        # Class 0 is majority
        prevalence['majority_class'] = 0
        prevalence['majority_prev'] = prevalence['c0']

    elif count_c0_all < count_c1_all:
        # Class 1 is majority
        prevalence['majority_class'] = 1
        prevalence['majority_prev'] = prevalence['c1']

    else:
        # They are equal. Set class 1 as majority
        print('N.B.: The prevalences are equal. Set majority class to CLASS_1')
        prevalence['majority_class'] = 1
        prevalence['majority'] = prevalence['c1']

    return prevalence


# %% shuffle_the_data
def shuffle_the_data(raw, num_train, num_valid):
    '''A function to separate the data to features and targets
    based on the defined nuber of training and validation examples.

    A scrambled validation feature set is also created and returned
    for the purposes of permutation testing
    '''
    # Shuffle the data and seperate out into features and targets
    shuffled_data = sklearn.utils.shuffle(raw)
    features = shuffled_data.drop(axis='columns', labels=['CLASS'])
    targets = shuffled_data[['CLASS']]

    # Seperate the shuffled features and targets into training and validation
    training_features = features.head(num_train)
    training_targets = targets.head(num_train)
    validation_features = features.tail(num_valid)
    validation_targets = targets.tail(num_valid)

    # Create a scrambled validation feature set
    scrambled_validation_features = sklearn.utils.shuffle(validation_features)

    return training_features, training_targets, validation_features,\
        validation_targets, scrambled_validation_features


# %% assign_agent_training_data
def assign_agents(
        num_agents, num_train_agent,
        training_features, training_targets
        ):
    '''A function to assign independent training data for each agent'''

    # Initialize
    features_agent = []
    targets_agent = []

    # Loop through the # of agents and append a dataframe
    # to each position that refers to that agents
    # feature sand targets
    for i in range(0, num_agents):
        features_agent.append(
                training_features.iloc[
                        i*num_train_agent:(i+1)*num_train_agent
                        ]
                )
        targets_agent.append(
                training_targets.iloc[
                        i*num_train_agent:(i+1)*num_train_agent
                        ]
                )

    return features_agent, targets_agent


# %%
def assign_balanced_agents(
        num_agents, num_train_agent,
        train_c0, train_c1
        ):
    """
    Assign agent data in a balanced manner, with an equal number of examples
    from each class, for each agent
    """

    # Number of training examples in each class per agent
    num_train_class = int(num_train_agent/2)

    # Initialize the list that will store the agent-wise training data
    # The training data for each class comes as a shuffled dataframe
    # that contains both features and targets
    # Since the data will be shuffled after combining the classwise data
    # I will keep the features and targets together before appending ..?
    features_agent = []
    targets_agent = []

    # Loop through the # of agents and append a dataframe
    # to each position that refers to that agent's features and targets
    # Do this in a balanced manner, so that the number of examples
    # from each class for each agent is equal
    for i in range(0, num_agents):
        # For this specific agent, extract this agent's training data
        # from each class's raw training data
        raw_agent_c0 = train_c0.iloc[i*num_train_class:(i+1)*num_train_class]
        raw_agent_c1 = train_c1.iloc[i*num_train_class:(i+1)*num_train_class]

        # Combine the class-wise agent raw data into a single dataframe
        raw_agent = pd.concat([raw_agent_c0, raw_agent_c1], ignore_index=False)
        # ...and then shuffle the combined dataframe
        raw_agent = sklearn.utils.shuffle(raw_agent)

        # Separate to features and targets, and append to the list
        # that stores these on a per-agent basis
        features_agent.append(raw_agent.drop(axis='columns', labels=['CLASS']))
        targets_agent.append(raw_agent[['CLASS']])

    return features_agent, targets_agent


# %%
def assign_uneven_balanced_agents(
        num_agents_total, num_train_agent,
        train_c0, train_c1
        ):
    """
    Assign uneven agents (i.e. agents are not all trained with the same
    number of training examples) that are balanced (equal number of examples
    from each class)
    """

    # Initialize the list that will store the agent-wise training data
    # The training data for each class comes as a shuffled dataframe
    # that contains both features and targets
    # Since the data will be shuffled after combining the classwise data
    # I will keep the features and targets together before appending ..?
    features_agent = []
    targets_agent = []

    # Loop through the # of agents and append a dataframe
    # to each position that refers to that agent's features and targets
    # Do this in a balanced manner, so that the number of examples
    # from each class for each agent is equal
    start_i = 0  # Counter to keep track of position within raw data
    for i in range(0, num_agents_total):
        # Number of training examples in each class per agent
        num_train_class = int(num_train_agent[i]/2)
        # For this specific agent, extract this agent's training data
        # from each class's raw training data
        raw_agent_c0 = train_c0.iloc[start_i:start_i+num_train_class]
        raw_agent_c1 = train_c1.iloc[start_i:start_i+num_train_class]

        # Move up counter
        start_i += num_train_class

        # Combine the class-wise agent raw data into a single dataframe
        raw_agent = pd.concat([raw_agent_c0, raw_agent_c1], ignore_index=False)
        # ...and then shuffle the combined dataframe
        raw_agent = sklearn.utils.shuffle(raw_agent)

        # Separate to features and targets, and append to the list
        # that stores these on a per-agent basis
        features_agent.append(raw_agent.drop(axis='columns', labels=['CLASS']))
        targets_agent.append(raw_agent[['CLASS']])

    return features_agent, targets_agent


# %%
def assign_even_imbalanced_agents(
        num_agents_total, num_train_c0, num_train_c1,
        train_c0, train_c1
        ):
    """
    Assign uneven agents (i.e. agents are all trained with the same number of
    training examples) that are imbalanced (not 50/50 class distribution
    for each agent)
    """

    # Initialize the list that will store the agent-wise training data
    # The training data for each class comes as a shuffled dataframe
    # that contains both features and targets
    # Since the data will be shuffled after combining the classwise data
    # I will keep the features and targets together before appending ..?
    features_agent = []
    targets_agent = []

    # Loop through the # of agents and append a dataframe
    # to each position that refers to that agent's features and targets
    # Do this in a balanced manner, so that the number of examples
    # from each class for each agent is equal
    start_c0 = 0  # Counter to keep track of position within raw data
    start_c1 = 0  # Counter to keep track of position within raw data
    for i in range(0, num_agents_total):
        # For this specific agent, extract this agent's training data
        # from each class's raw training data
        end_c0 = start_c0 + num_train_c0[i]
        end_c1 = start_c1 + num_train_c1[i]

        raw_agent_c0 = train_c0.iloc[start_c0:end_c0]
        raw_agent_c1 = train_c1.iloc[start_c1:end_c1]

        # Move up counter
        start_c0 += num_train_c0[i]
        start_c1 += num_train_c1[i]

        # Combine the class-wise agent raw data into a single dataframe
        raw_agent = pd.concat([raw_agent_c0, raw_agent_c1], ignore_index=False)
        # ...and then shuffle the combined dataframe
        raw_agent = sklearn.utils.shuffle(raw_agent)

        # Separate to features and targets, and append to the list
        # that stores these on a per-agent basis
        features_agent.append(raw_agent.drop(axis='columns', labels=['CLASS']))
        targets_agent.append(raw_agent[['CLASS']])

    return features_agent, targets_agent


# %%
def balanced_shuffle(
        raw_data, num_train_total, num_valid
        ):
    """
    Shuffle and split the data into training and test sets
    BUT with an equal number of examples from each class
    in the training set.

    For the training set, it returns a class-wise raw data set
    where features and targets (the "CLASS" column) are in one dataframe

    The test data is split into features and targets, along with
    scrambled test features
    """

    # Number of training examples per class
    num_train_class = int(num_train_total/2)

    # Split the raw data into class-wise data
    raw_c0 = raw_data[raw_data.CLASS == 0]
    raw_c1 = raw_data[raw_data.CLASS == 1]

    # Shuffle the classwise data
    raw_c0 = sklearn.utils.shuffle(raw_c0)
    raw_c1 = sklearn.utils.shuffle(raw_c1)

    train_c0 = raw_c0.head(num_train_class)
    test_c0 = raw_c0.tail(raw_c0.shape[0]-num_train_class)
    train_c1 = raw_c1.head(num_train_class)
    test_c1 = raw_c1.tail(raw_c1.shape[0]-num_train_class)

    # Test set
    raw_test = pd.concat([test_c0, test_c1], ignore_index=True)
    raw_test = sklearn.utils.shuffle(raw_test)
    test_features = raw_test.drop(axis='columns', labels=['CLASS'])
    test_targets = raw_test[['CLASS']]
    scram_test_features = sklearn.utils.shuffle(test_features)

    # Full training set
    # Not used, but save code for later
#    raw_train = pd.concat([train_c0, train_c1], ignore_index=True)
#    raw_train = sklearn.utils.shuffle(raw_train)
#    train_features = raw_train.drop(axis='columns', labels=['CLASS'])
#    train_targets = raw_train[['CLASS']]

    return train_c0, train_c1, test_features, test_targets, scram_test_features


# %%
def balanced_central(raw_c0, raw_c1):
    """
    Using balanced training data, create a balanced central learning dataset
    Take in class-wise training data (raw, so features and targets)
    Output combined and shuffled features and targets
    """
    features = list()
    targets = list()
    central_raw = pd.concat([raw_c0, raw_c1], ignore_index=False)
    central_raw = sklearn.utils.shuffle(central_raw)
    features.append(central_raw.drop(axis='columns', labels=['CLASS']))
    targets.append(central_raw[['CLASS']])
    return features, targets


# %%
def assign_output_dict(output_tuple, key_list):
    """
    Take a tuple of function returns (output_tuple) and
    map it to a dictionary using keys from (key_list)
    """
    assigned_output = dict()
    for i in range(len(output_tuple)):
        assigned_output[key_list[i]] = output_tuple[i]
    return assigned_output
