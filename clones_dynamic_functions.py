from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import math
import numpy as np

class Model:
    """
    Simulation (modelling) of the clones dynamics.

    :argument:
    number_of_species = number of species / clones (int)
    mutant_percent  # percentage of mutation initially (float in [0,1])
    number_of_mutations # number of mutations
    self.net_growth_rates_array = net_growth_rates_array  # array (dimension : 2*number of species )
                                  of birth and death rates [rP1, rM1, rP2, rM2...] =
                                        [birth_normal, death_normal, birth_mut1, death_mut1, birth_mut2, death_mut2,...]
    self.interaction = interaction  # scalar of the interaction strength
    self.plating_efficiency = plating_efficiency  # scalar (>=1) captures the plating efficiency of mutations

    Methods:
    time_propagation(self, number_of_steps=10**6 , passaging = 4)
    """

    def __init__(self, number_of_species, mutant_percent, number_of_mutations, net_growth_rates_array, interaction,
                 plating_efficiency,initial_mean_clone_size):
        self.number_of_species = number_of_species  # Number of Species (clones)
        self.mutant_percent = mutant_percent  # percentage of mutation initially
        self.number_of_mutations = number_of_mutations # number of mutations
        self.net_growth_rates_array = net_growth_rates_array  # array of birth and death rates [rP1, rM1, rP2, rM2...]
        self.interaction = interaction  # scalar of the interaction strength
        self.plating_efficiency = plating_efficiency  # scalar (>=1) captures the plating efficiency of mutations
        self.initial_mean_clone_size = initial_mean_clone_size

    def time_propagation_pass_every_gen(self, number_of_steps=10**6 , passaging = 4):
        """
        Propagate the model with time; simulate and saved the clones compositions in every generation

        :param self: the parameters of the model
        :param number_of_steps: the total number of steps/reactions (int)
        :param passaging: the number of generations before sampling and passaging

        :return: clones compositions in every generation
        """

        rP = np.zeros(self.number_of_species)
        rM = np.zeros(self.number_of_species)
        rates_array= self.net_growth_rates_array

        # growth rates:
        rP[:int(self.number_of_species * self.mutant_percent)] = rates_array[0] * self.plating_efficiency
        rP[int(self.number_of_species * self.mutant_percent):] = rates_array[0]

        #death rate:
        rM[int(self.number_of_species * self.mutant_percent / self.number_of_mutations):] = rates_array[1]
        for i in range(self.number_of_mutations):
             rM[int(i * self.number_of_species * self.mutant_percent / self.number_of_mutations)
                :int((i+1) * self.number_of_species * self.mutant_percent / self.number_of_mutations)] = rates_array[i+3]

        X = np.ones(self.number_of_species)  # starting with one individual

        X = np.array([int(i) for i  in np.random.exponential(self.initial_mean_clone_size, self.number_of_species)])  #starting with m individuals ; m expo. distrubted

        initial_number_of_cells = sum(X)
        print('initial number of cells', initial_number_of_cells)

        # for interaction (affect only on normal cells - suppressed by mutants
        normal_indicator = np.zeros(self.number_of_species)
        normal_indicator[int(self.number_of_species * self.mutant_percent):] = 1

        interaction = self.interaction

        X_X = np.zeros((100, self.number_of_species))
        X_X[0, :] = X
        time_gen = np.zeros(100)
        t = 0
        flag=0
        rng = np.random.default_rng()

        for n in range(number_of_steps):

            RxPlus = np.array(rP) * np.array(X)
            RxMinus = np.array(rM) * np.array(X) + \
                      interaction * (np.array(X[:int(self.number_of_species * self.mutant_percent)]).sum()) * \
                      np.array(normal_indicator) * np.array(X)
            Rtotal = sum(RxPlus + RxMinus)

            rand = np.random.uniform()
            dt = -math.log(rand) / Rtotal
            t = t + dt

            r = np.random.uniform()

            RR = (np.concatenate((np.array([0]), (np.cumsum(np.concatenate((RxPlus, RxMinus))))))) / Rtotal

            # i = list(abs(RR - r)).index(min(abs(RR - r)))
            temp1 = np.where(r >= np.array(RR))
            temp2 = np.where(r < np.array(RR))
            s = temp1[0][-1]

            if temp1[0][-1] != (temp2[0][0] - 1):
                print('error')

            # print(s)
            X[s % self.number_of_species] = X[s % self.number_of_species] \
                                            + (self.number_of_species >= s) - (self.number_of_species < s)

            # for s in range(2*NumberOfSpecies):
            #    if r>RR[s] and r<RR[s+1]:
            #        X[s % NumberOfSpecies] = X[s % NumberOfSpecies] + (NumberOfSpecies >= s) - (NumberOfSpecies < s)
            #        break

            #print((math.log2(sum(X) / initial_number_of_cells)) % (1) )

            if (math.log2(sum(X) / initial_number_of_cells)) % (1) == 0 and (math.log2(sum(X) / initial_number_of_cells)) > 0:
                X_X[int(math.log2(sum(X) / initial_number_of_cells)) + flag, :] = X
                print(math.log10(n), int(math.log2(sum(X) / initial_number_of_cells)) + flag,
                       X[:int(self.number_of_species / 100)].sum(axis=0) / (X.sum(axis=0)))
                time_gen[int(math.log2(sum(X) / initial_number_of_cells)) + flag] = t
                if (math.log2(sum(X) / initial_number_of_cells)) % (passaging) == 0:
                    X = rng.multinomial(initial_number_of_cells,
                                        X / (X.sum()))  # THAT'S NOT EXACT SAMPLING _ CAN SAMPLE MORE THEN HAVE
                    print('passing every ' + str(passaging) + ' generations, ')
                    flag = flag + passaging

            if int(math.log2(sum(X) / initial_number_of_cells)) + flag == 99:  # no more than 100 generations to keep
                break

        print('saving data')
        np.save("sample.npy", X_X)

        return X_X

    def time_propagation_pass_every_time(self, number_of_steps=10**6 , passaging = 4, percent_of_pass = 10):
        """
        Propagate the model with time; simulate and saved the clones compositions in every generation

        :param self: the parameters of the model
        :param number_of_steps: the total number of steps/reactions (int)
        :param passaging: the number of days before sampling and passaging
        :param percent_of_pass: the % of passaging

        :return: clones compositions in every generation
        """

        rP = np.zeros(self.number_of_species)
        rM = np.zeros(self.number_of_species)
        rates_array= self.net_growth_rates_array

        # growth rates:
        rP[:int(self.number_of_species * self.mutant_percent)] = rates_array[0] * self.plating_efficiency
        rP[int(self.number_of_species * self.mutant_percent):] = rates_array[0]

        #death rate:
        rM[int(self.number_of_species * self.mutant_percent / self.number_of_mutations):] = rates_array[1]
        for i in range(self.number_of_mutations):
             rM[int(i * self.number_of_species * self.mutant_percent / self.number_of_mutations)
                :int((i+1) * self.number_of_species * self.mutant_percent / self.number_of_mutations)] = rates_array[i+3]

        X = np.ones(self.number_of_species)  # starting with one individual

        X = np.array([int(i) for i  in np.random.exponential(self.initial_mean_clone_size, self.number_of_species)])  #starting with m individuals ; m expo. distrubted

        initial_number_of_cells = sum(X)
        print('initial number of cells', initial_number_of_cells)

        # for interaction (affect only on normal cells - suppressed by mutants
        normal_indicator = np.zeros(self.number_of_species)
        normal_indicator[int(self.number_of_species * self.mutant_percent):] = 1

        interaction = self.interaction

        X_X = np.zeros((100, self.number_of_species))
        X_X[0, :] = X
        time_gen = np.zeros(100)
        t = 0
        flag=0
        rng = np.random.default_rng()

        for n in range(number_of_steps):

            RxPlus = np.array(rP) * np.array(X)
            RxMinus = np.array(rM) * np.array(X) + \
                      interaction * (np.array(X[:int(self.number_of_species * self.mutant_percent)]).sum()) * \
                      np.array(normal_indicator) * np.array(X)
            Rtotal = sum(RxPlus + RxMinus)

            rand = np.random.uniform()
            dt = -math.log(rand) / Rtotal
            t = t + dt

            r = np.random.uniform()

            RR = (np.concatenate((np.array([0]), (np.cumsum(np.concatenate((RxPlus, RxMinus))))))) / Rtotal

            # i = list(abs(RR - r)).index(min(abs(RR - r)))
            temp1 = np.where(r >= np.array(RR))
            temp2 = np.where(r < np.array(RR))
            s = temp1[0][-1]

            if temp1[0][-1] != (temp2[0][0] - 1):
                print('error')

            # print(s)
            X[s % self.number_of_species] = X[s % self.number_of_species] \
                                            + (self.number_of_species >= s) - (self.number_of_species < s)

            # for s in range(2*NumberOfSpecies):
            #    if r>RR[s] and r<RR[s+1]:
            #        X[s % NumberOfSpecies] = X[s % NumberOfSpecies] + (NumberOfSpecies >= s) - (NumberOfSpecies < s)
            #        break

            #print((math.log2(sum(X) / initial_number_of_cells)) % (1) )



            X_X[int(t / 24),:] = X
            # if t>24:
            #     print('days = ', int(t/24),
            #        X[:int(self.number_of_species / 100)].sum(axis=0) / (X.sum(axis=0)))

            if t > 24 * passaging * (flag+1) :

                print('passing every ' + str(passaging) + ' days \t t=', t/24)
                print('number of cells before pass' + str(X.sum(axis=0)))
                num_to_pass=  (X.sum()) * percent_of_pass /100
                X = rng.multinomial(num_to_pass,
                                    X / (X.sum()))  # THAT'S NOT EXACT SAMPLING _ CAN SAMPLE MORE THEN HAVE
                print('number of cells after pass' + str(X.sum(axis=0)))
                flag = flag + 1

            # if int(t/24) == 99:  # no more than 100 generations to keep
            #     break

            if int(t / (passaging*24)) == 10:  # no more than 10 passaging to keep

                break

        print('saving data')
        np.save("sample.npy", X_X)

        return X_X

def mean_field():
    """
    generate and plot the mean field (deterministic) solution.
    :argument:
    :var:

    :return:
        """

    Int_opt = np.zeros(8)
    alpha_opt = np.zeros(8)

    np.seterr(divide='ignore', invalid='ignore')

    mutant_percent=0.1
    m0=mutant_percent
    g1 = (math.log(2) / 16)-(-math.log(0.9791) / 16)
    g0 = (math.log(2) / 16)-(-math.log(0.8) / 16)
    N0 = 1

    Int=10**-5 #no interactions
    alpha=1 #no plating efficiency


    def func(i, g1, Int, mt_perc, Nt, time):
        # res = least_squares(f, x0=70, bounds = ((0, 10**4)),args=(g1,g0,mt,Nt,Int))
        # print(Int,res.x,i)
        # dt=res.x
        dt = fsolve(f, x0=60, args=(g1, g0, mt_perc, Nt, Int, time))
        tt = time + dt
        try:
            X = m0 / (N0 - m0) * np.exp(Int * (1 / g1) * (np.exp(g1 * tt) - 1)) * np.exp((g1 - g0) * tt)
            F = X / (X + 1)
        except OverflowError:
            F = 1
        # return (m0)*math.exp(g1*tt)#/(m0*math.exp(g1*tt)+(N0-m0)*math.exp(g0*tt)*(math.exp(-Int*m0*math.exp(g1*tt)+1)))
        return F


    def f(t, g1, g0, mt_perc, Nt, Int, time):
        try:
            ff = 16 - np.exp(g1 * t) - (Nt - mt_perc) * np.exp(g0 * t) * (
                        np.exp(-(Int / g1) * np.exp(g1 * time) * (np.exp(g1 * t) - 1)) - np.exp((g1 - g0) * t))
        except OverflowError:
            ff = float(60)
            print(Int, mt)
        return ff


    time = np.zeros([100])
    fit = np.zeros([100])

    mt_perc = m0
    for i in range(25):
        Nt = 1
        print(time[i])
        time[i + 1] = time[i] + fsolve(f, x0=60, args=(g1, g0, mt_perc, Nt, Int, time[i]))
        fit[i+1]=func(i,g1,0,mt_perc,Nt,time[i])
        # print('function',1 if pd.isna(func(i,Int,mt,Nt,time[I,i])) else func(i,Int,mt,Nt,time[I,i]) )
        mt_perc= fit[i]


    import matplotlib.pyplot as plt

    plt.plot(fit[:25],':',label='model')
    print(fit[:25])

    NumberOfSpecies = 10 ** 4

    X_X = np.load('sample.npy')

    generations=np.where((X_X.sum(axis=1))>0)

    gen_num=generations[0][-1]

    ratio=[X_X[i,:int(NumberOfSpecies*0.1)].sum(axis=0)/(X_X[i,:].sum(axis=0)) for i in range(gen_num+1)]

    plt.plot(ratio,'o',label='simulations')
    plt.xlabel('# passaging')
    plt.ylabel('ratio')
    plt.legend()
    plt.show()

