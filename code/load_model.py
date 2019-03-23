# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 00:39:48 2019

@author: grosati
"""



# load the model
from keras.models import load_model
from numpy.testing import assert_allclose

new_model = load_model(checkpoint_path)
assert_allclose(model.predict(x),
                new_model.predict(x),
                1e-5)