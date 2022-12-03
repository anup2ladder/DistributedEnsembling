#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:20:24 2019
@author: anupt

Functions to manage memory usage in Keras/Tensorflow

"""

# %% Import modules
import keras as K
#import gc


# %% Configuration options to limit GPU memory usage to as necessary, 
# rather than pre-allocate entire memory
limitGPUmem = K.backend.tf.ConfigProto()
limitGPUmem.gpu_options.allow_growth = True


# %% Limit Tensorflow GPU memory use to only the necessary amount
def initialize(config=None):
    """
    Initialize a Keras session
    By default, it will initialize to use the entire GPU memory
    Pass in config=limitGPUmem to use only what is necessary
    """
    sess = K.backend.tf.Session(config=config)
    K.backend.tensorflow_backend.set_session(sess)
    sess = K.backend.get_session()
    return sess


# %% Reset Tensorflow/Keras
def reset(config=None):
    """
    "Reset" Kera/Tensorflow's memory usage.
    This closes and clears the current active graph in Keras
    and then initializes a new session

    By default, it will initialize to use the entire GPU memory
    Pass in config=limitGPUmem to use only what is necessary
    """
    oldsess = K.backend.get_session()
    oldsess.close()
    K.backend.clear_session()
#    gc.collect()

    newsess = K.backend.tf.Session(config=config)
    K.backend.tensorflow_backend.set_session(newsess)

    return newsess
