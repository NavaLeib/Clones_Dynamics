import math
import numpy as np
import random

from clones_dynamic_functions import *
from analysis import Analysis


for cells in [10 ** 5]:#[10**4, 10 ** 5,10**6]

    for m in [10]:# [10, 10 ** 2, 10 ** 3]: #mean initial clone size

        if int(cells / m) < 10**3: #taking only (cells, m) which provides at least 10^3 clones initially
            break

        for p in [3]:#[3,5]:  #dayes to passage

            for y in [5]:  #paercent to passage

                # model = Model(number_of_species=int(cells / m), mutant_percent=0.1, number_of_mutations=2,
                #               net_growth_rates_array=[math.log(2) / 16, -math.log(0.8) / 16,
                #                                       math.log(2) / 16, -math.log(0.9791) / 16,
                #                                       math.log(2) / 16, -math.log(0.9) / 16],
                #               interaction=0, plating_efficiency=1, initial_mean_clone_size=m)
                # #
                # # # X_X=model.time_propagation_pass_every_gen(number_of_steps=5*10**5,passaging = 5) #pqssaging every M generations
                # #
                # X_X = model.time_propagation_pass_every_time(number_of_steps=10 ** 9, passaging=p,
                #                                              percent_of_pass=y)  # passaging every p days.
                #
                # np.save('Cells' + str(cells) +
                #         '_MeanInitClone' + str(m)+
                #         '_Days' + str(p) +
                #         '_PercentY' + str(y)+ ".npy", X_X)
                #
                # np.save('initial_mean_clone_size' + str(m) + ".npy", X_X)
                X_X = np.load('Cells' + str(cells) + '_MeanInitClone' + str(m)+ '_Days' + str(p) + '_PercentY' + str(y)+ ".npy")


                analysis = Analysis(X_X=X_X, number_of_species= int( cells / m), mutant_percent=0.1)



                #analysis.figure_number_clones(passaging=5, initial_mean_clone_size=m)

                #analysis.figure_hist_many_mutants(number_of_mutations=2,generations_to_plot=[0,3])
                #
                analysis.figure_hist_many_mutants_all_pass(number_of_mutations=2,generations_to_plot=[0,3,6,9],
                                                           percent_of_pass =y, initial_mean_clone_size =m , cells = cells)
                #

                # mean_field()

                print('Cells' + str(cells) + '_MeanInitClone' + str(m)+ '_Days' + str(p) + '_PercentY' + str(y)+ ".npy")

                #analysis = Analysis(X_X=X_X, number_of_species= int( cells/ m), mutant_percent=0.1)

                #analysis.figure_hist_many_mutants_all_pass(number_of_mutations=2, generations_to_plot=[0,5,10,15],percent_of_pass=y,
                                                           #initial_mean_clone_size=m, cells=cells)
                #

