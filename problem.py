"""Black-box problem definition for the wind farm layout assignment.

The objective functions and the constraint below are taken *unchanged* from the
provided notebook (1BM120_assignment3.ipynb). The assignment is explicit that we
must treat the surrogate as a black box and not touch what happens inside these
functions, so the bodies are the original code. The only thing added here is a
small amount of glue so the ensemble can be loaded when this file is imported as
a normal module instead of being run inside the notebook.

Everyone on the team imports the objectives from here, so we all measure the
exact same problem.
"""

import os
import pickle

import numpy as np
import xgboost as xgb
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist


# --- problem constants ------------------------------------------------------
# These describe the search space and the physical wind farm. They are handy to
# have as named values so the rest of the framework does not hard-code numbers.
N_TURBINES = 5
DIM = 2 * N_TURBINES          # 10 decision variables, an (x, y) pair per turbine
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

# Original names used inside the given objective2 / constraint1 code. Keep the
# exact names and values so those functions stay byte-for-byte the assignment's.
rotor_diameter = 126          # in meters
farm_length = 333.33 * 5      # 1666.65 meters, the [0, 1] box maps onto this

# Aliases in the framework's naming style.
ROTOR_DIAMETER = rotor_diameter
FARM_LENGTH = farm_length

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_FILE = os.path.join(_HERE, "Ensemble.pkl")


# Define median-based ensemble of surrogates
class medClassifier:
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def predict(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            try:
                self.predictions_.append(classifier.predict(X)) #used for the random forest that is part of the ensemble
            except:
                X = xgb.DMatrix(X)
                self.predictions_.append(classifier.predict(X)) #used for the XGBoost models that are part of the ensemble
        med1 = np.median(self.predictions_, axis=0) #median of predictions
        mean1 = np.mean(self.predictions_, axis=0) #mean of predictions
        out = med1 + np.random.rand()*np.abs(med1-mean1) #add more noise if median is far from mean, indicating more uncertainty, also all noise is positive to focus on minimizing parts with more certainty
        return out


class _EnsembleUnpickler(pickle.Unpickler):
    """The pickle was saved from the notebook, so medClassifier lived in __main__
    at that point. Redirect that reference to the class defined right above, so a
    plain ``import problem`` works instead of blowing up on medClassifier."""

    def find_class(self, module, name):
        if name == "medClassifier":
            return medClassifier
        return super().find_class(module, name)


def _load_ensemble(path=_ENSEMBLE_FILE):
    with open(path, "rb") as file:
        return _EnsembleUnpickler(file).load()


# Load the ensemble once, the same way the notebook does it.
Ensemble = _load_ensemble()


# This is the first objective function
def objective1(x):
    x = np.array([x])
    pred = Ensemble.predict(x)
    return pred[0]


def objective2(x):

    x = np.array([x])
    coords = np.resize(x,(2,5)) # wind turbine coordinates

    # Use a Monte Carlo simulation for the birds, who all fly from top to bottom at an x-location with a normal distribution. The mean of the normal distribution is far to the left of the wind farm.
    bird_mean = -25000 #width of bird corridor is 50km
    x_sigma = 4 # assume this many sigma of birds to stay within the planned corridor
    bird_std = (25000/x_sigma)
    #simulate birds (location in meters)
    nr_birds = 1000 # number of birds
    birds = truncnorm.rvs(x_sigma,x_sigma+farm_length/bird_std,loc=bird_mean,scale=bird_std,size=nr_birds) # Uses a truncated normal distribution, only sampling in the wind farm, not the entire bird corridor

    #check how many birds are close to a wind turbine (everything right of the leftmost turbine - rotor_diameter is dangerous area)
    leftmost = np.min(coords[0]) #location of leftmost turbine, in [0,1] units
    leftmost = leftmost*farm_length #change to meters
    threshold = leftmost-rotor_diameter #threshold of where the dangerous area starts (from left to right)

    close_birds = np.sum(birds >= threshold)/nr_birds #check how many birds fly to the right of the threshold
    return close_birds


def constraint1(x):
    coords = np.resize(x,(5,2))
    min_dist = 999999 #minimum distance between turbines (Euclidean)

    for turb in range(4):
        dists = cdist([coords[turb]],coords[turb+1:])
        next_min = np.min(dists)
        if next_min < min_dist:
            min_dist = next_min

    if min_dist*farm_length < 2*rotor_diameter:
        constr = 0 # constraint not satisfied, wind turbines are too close to each other
    else:
        constr = 1 # constraint satisfied
    return constr
