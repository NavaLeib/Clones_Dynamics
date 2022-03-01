import math
import numpy as np
import random

from clones_dynamic_functions import *
from analysis import Analysis

# model = Model(number_of_species=10**3, mutant_percent=0.1, number_of_mutations=2,
#               net_growth_rates_array=[math.log(2)/16,-math.log(0.8)/16,
#                                       math.log(2)/16, -math.log(0.9791)/16,
#                                       math.log(2)/16,-math.log(0.9)/16],
#               interaction=0, plating_efficiency=1)
#
# X_X=model.time_propagation(number_of_steps=10**5)

X_X = np.load('sample.npy')

mean_field()

analysis = Analysis(X_X=X_X,number_of_species=100, mutant_percent=0.1)

analysis.figure_hist_many_mutants(number_of_mutations=2,generations_to_plot=[4,7])



