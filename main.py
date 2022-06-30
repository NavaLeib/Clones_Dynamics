import math
import numpy as np
import random

from clones_dynamic_functions import *
from analysis import Analysis

for p in [3, 4, 5]:
    m=10
    model = Model(number_of_species=int( 10 **4  / m), mutant_percent=0.1, number_of_mutations=2,
                  net_growth_rates_array=[math.log(2) / 16, -math.log(0.8) / 16,
                                          math.log(2) / 16, -math.log(0.9791) / 16,
                                          math.log(2) / 16, -math.log(0.9) / 16],
                  interaction=10 ** (-6), plating_efficiency=1, initial_mean_clone_size=m)

    # X_X=model.time_propagation_pass_every_gen(number_of_steps=5*10**5,passaging = 5) #pqssaging every M generations

    X_X = model.time_propagation_pass_every_time(number_of_steps= 10 ** 9, passaging= p ,
                                                 percent_of_pass=10)  # passaging every p days

    np.save('passging days' + str(p) + ".npy", X_X)
    X_X = np.load('initial_mean_clone_size' + str(m) + ".npy")

    # mean_field()

    analysis = Analysis(X_X=X_X, number_of_species= int( 10 ** 4 / m), mutant_percent=0.1)

    analysis.figure_number_clones(passaging=5, initial_mean_clone_size=m)

    #analysis.figure_hist_many_mutants(number_of_mutations=2,generations_to_plot=[0,3])
    #
    #analysis.figure_hist_many_mutants_all_pass(number_of_mutations=2,generations_to_plot=[0,3,6,9])
    #

