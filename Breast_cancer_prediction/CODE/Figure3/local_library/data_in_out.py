#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:06:03 2019

@author: anupt
"""

# %% Import modules
import pandas as pd
import pickle
import os


# %%
def pickle_out_variable(variable, filename, folder_path='.'):
    """
    Simple function to pickle out a single python variable
    given the filename (filename.pickle), as a string
    and a foldername ('.' for current directory) as a string
    """
    # Open the file, call it "results.pickle"
    pickle_out = open(os.path.join(folder_path, filename), 'wb')
    # Write out the results to defined file
    pickle.dump(variable, pickle_out)
    # Close the file
    pickle_out.close()


# %%
def pickle_in_variable(filename, folder_path='.'):
    """
    Simple function to pickle in a single python variable
    given the filename (filename.pickle), as a string
    and a foldername ('.' for current directory) as a string
    """
    # Open the file
    pickle_in = open(os.path.join(folder_path, filename), 'rb')
    # Read in the variable
    variable = pickle.load(pickle_in)
    # Close the file
    pickle_in.close()

    # Return the variable
    return variable


# %%
def get_data_filename(source_file, source_directory):
    """
    Read the first line of a text file and get the data filename
    stored within.
    Only reads the first line.
    The datafile should be in the same directory as well
    """
    # Open the file
    f = open(os.path.join(source_directory, source_file))
    # Read the line
    data_filename = f.readline()
    # Close the file
    f.close()
    # Remove the newline
    data_filename = data_filename.strip('\n')
    return data_filename


# %%
def import_the_data(filepath):
    """
    Import the entire dataset into a pandas dataframe

    Input: Filepath, either as a relative or absolute path
    Output: Pandas dataframe of the data
    as well as information on the number of exmaples and the number of columns
    """

    # Import the raw data
    raw_data = pd.read_csv(filepath)

    # Retrieve the number of examples and number of features
    (num_examples, num_columns) = raw_data.shape

    return raw_data, num_examples, num_columns


# %%
def import_2conditions(filepath):
    """
    Import and process the conditions file that the script will loop through

    Input: Filepath for conditions file (2 conditions)
    Output: condition1 and condition2 dictionaries, which both contain:
        - 'name': The title of the condition
        - 'values' :
    """

    # Import the raw conditions information
    conditions_raw = pd.read_csv(filepath)

    # Initialize the dictionaries for condition1 and condition2
    cond1 = {}
    cond2 = {}

    # Extract the column names (i.e. the condition names)
    cond1['name'], cond2['name'] = conditions_raw.columns.values

    # Remove empty values and (in this case) ensure they are integers (int8)
    cond1['values'] = conditions_raw[cond1['name']].dropna().astype('int32')
    cond2['values'] = conditions_raw[cond2['name']].dropna().astype('int32')

    # Retrieve the number of parameters in each condition
    # i.e. the length of the list of values to test in each condition
    cond1['num'] = cond1['values'].size
    cond2['num'] = cond2['values'].size

    return cond1, cond2


# %%
def import_3conditions(filepath):
    """
    Import and process the conditions file that the script will loop through

    Input: Filepath for conditions file (2 conditions)
    Output: condition1 and condition2 dictionaries, which both contain:
        - 'name': The title of the condition
        - 'values' :
    """

    # Import the raw conditions information
    conditions_raw = pd.read_csv(filepath)

    # Initialize the dictionaries for condition1 and condition2
    cond1 = {}
    cond2 = {}
    cond3 = {}

    # Extract the column names (i.e. the condition names)
    cond1['name'], cond2['name'], cond3['name'] = conditions_raw.columns.values

    # Remove empty values and (in this case) ensure they are integers (int8)
    cond1['values'] = conditions_raw[cond1['name']].dropna().astype('int32')
    cond2['values'] = conditions_raw[cond2['name']].dropna().astype('int32')
    cond3['values'] = conditions_raw[cond3['name']].dropna().astype('int32')

    # Retrieve the number of parameters in each condition
    # i.e. the length of the list of values to test in each condition
    cond1['num'] = cond1['values'].size
    cond2['num'] = cond2['values'].size
    cond3['num'] = cond3['values'].size

    return cond1, cond2, cond3


# %%
def import_4conditions(filepath):
    """
    Import and process the conditions file that the script will loop through

    Input: Filepath for conditions file (2 conditions)
    Output: condition1 and condition2 dictionaries, which both contain:
        - 'name': The title of the condition
        - 'values' :
    """

    # Import the raw conditions information
    conditions_raw = pd.read_csv(filepath)

    # Initialize the dictionaries for condition1 and condition2
    cond1 = {}
    cond2 = {}
    cond3 = {}
    cond4 = {}

    # Extract the column names (i.e. the condition names)
    cond1['name'], cond2['name'], cond3['name'], cond4['name'] = \
        conditions_raw.columns.values

    # Remove empty values and (in this case) ensure they are integers (int8)
    cond1['values'] = conditions_raw[cond1['name']].dropna().astype('int32')
    cond2['values'] = conditions_raw[cond2['name']].dropna().astype('int32')
    cond3['values'] = conditions_raw[cond3['name']].dropna().astype('int32')
    cond4['values'] = conditions_raw[cond4['name']].dropna().astype('int32')

    # Retrieve the number of parameters in each condition
    # i.e. the length of the list of values to test in each condition
    cond1['num'] = cond1['values'].size
    cond2['num'] = cond2['values'].size
    cond3['num'] = cond3['values'].size
    cond4['num'] = cond4['values'].size

    return cond1, cond2, cond3, cond4
