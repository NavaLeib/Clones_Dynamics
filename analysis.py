

import  numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy import stats

from clones_dynamic_functions import Model


class Analysis(Model):
    """
    generate figures from the simulated data

    :argument:
    X_X = the simulated data
    """
    def __init__(self, X_X,number_of_species=100, mutant_percent=0.1):
        self.X_X = X_X
        self.gen_num = np.where((X_X.sum(axis=1)) > 0)[0][-1]
        self.number_of_species = number_of_species  # Number of Species (clones)
        self.mutant_percent = mutant_percent  # percentage of mutation initially

    def figure_ratio_generation(self):
        """
        generate from data plot of ratio = all mutants / all vs generation
        :return:
        """
        ratio = [self.X_X[i, :int(self.number_of_species * self.mutant_percent)].sum(axis=0)
                 / (self.X_X[i, :].sum(axis=0)) for i in range(self.gen_num + 1)]
        plt.plot(ratio, 'o')
        plt.xlabel('generation')
        plt.ylabel('ratio')
        plt.show()
        pass

    def figure_ratio_passaging(self,passaging):
        """
        generate from data plot of ratio = all mutants / all vs passaging
        :param passaging: int
        :return:
        """
        ratio = [self.X_X[i, :int(self.number_of_species * self.mutant_percent)].sum(axis=0)
                 / (self.X_X[i, :].sum(axis=0)) for i in range(self.gen_num + 1)]
        plt.plot(ratio[::passaging], 'o')
        plt.xlabel('# passaging')
        plt.ylabel('ratio')
        plt.show()
        pass

    def figure_ratio_generation_many_mutants(self, number_of_mutations):
        """
        generate from data plot of ratio = every mutants / all vs generation
        :return:
        """
        ratio=[]
        for mut in range(number_of_mutations):
            ratio.append([self.X_X[i, int(mut * self.number_of_species * self.mutant_percent):
                                    int((mut+1) * self.number_of_species * self.mutant_percent)].sum(axis=0)
                          / (self.X_X[i, :].sum(axis=0)) for i in range(self.gen_num + 1)])
            plt.plot(ratio[mut], 'o', label= 'mutant ' + str(int(mut)+1))
        plt.xlabel('generation')
        plt.ylabel('ratio')
        plt.legend()
        plt.show()
        pass

    def figure_hist_many_mutants(self, number_of_mutations, generations_to_plot):
        """
        generate from data stacked histogram for different generation
        :argument:
        number_of_mutation = number of mutations specified in teh model (int)
        generations_to_plot = array of generations to plot [gen_1,gen_2,....] (int, int,.. ) --> currently [gen1, gen2]
        :return:
        """

        ratio=[]
        for mut in range(number_of_mutations):
            ratio.append([self.X_X[i, int(mut * self.number_of_species * self.mutant_percent):
                                    int((mut+1) * self.number_of_species * self.mutant_percent)].sum(axis=0)
                          / (self.X_X[i, :].sum(axis=0)) for i in range(self.gen_num + 1)])

        fig, axs = plt.subplots(1,len(generations_to_plot))
        fig.supxlabel('clone size (fraction of total cells)',fontsize=20)
        fig.supylabel('clones counts',fontsize=20)

        for (index,gen_num) in enumerate(generations_to_plot):
            location=[0,1]
            (i)= location[index]

            Data = self.X_X[gen_num,:]/(self.X_X[gen_num,:].sum())
            Data = Data[Data>0]

            P = ss.expon.fit(Data)
            rX = np.linspace(0,max(Data), 1000)
            rP = ss.expon.pdf(rX, *P)
            axs[i].plot(rX, rP, color='blue')

            bins = 10**(np.arange(-4.5,-1.5,0.1))
            counts, bins, bars = axs[i].hist([Data[:int(self.number_of_species * self.mutant_percent/2)],
                                                Data[int(self.number_of_species * self.mutant_percent/2):
                                                     int(self.number_of_species * self.mutant_percent)],
                                                Data[int(self.number_of_species * self.mutant_percent):]], density=True,
                                               bins=10**(np.arange(-4.5,-1.5,0.1)),alpha=0.5 ,histtype='barstacked',
                                               color=('orange','saddlebrown','blue'))
            counts, bins, bars = axs[i].hist(Data, density=True,bins=10**(np.arange(-4.5,-1.5,0.1)),alpha=0.0)
            #plt.plot(bins[1:]-np.diff(bins)/2,counts ,'.')
            axs[i].set_yscale('log')
            axs[i].set_xscale('log')
            axs[i].set_title("$\\bf{1,12,17,20}$: "+str(round(ratio[0][gen_num],2)) + "; $\\bf{1q}$: "
                             +str(round(ratio[0][gen_num],2)), fontsize=20)

            #plt.show()

            from scipy.optimize import curve_fit

            x = bins[1:]-np.diff(bins)/2

            def MLE(params,*args):
                y= np.array(args,dtype=float)
                a, k1, k2 = params
                yPred = (1-a)*k1*np.exp(-k1*y) + a*k2*np.exp(-k2*y)
                negLL = -np.sum(np.log(yPred))
                return negLL

            y = counts
            import scipy.optimize
            from scipy.optimize import minimize
            bnd = ((0, 1), (10**2,10**5), (10**2, 10**5))
            guess=np.array([1,1000,1000])
            results = scipy.optimize.minimize(MLE, guess, args=Data,bounds=bnd,method= 'L-BFGS-B')

            print(results.x) # This contains your three best fit parameters
            a, k1, k2 = results.x
            curvey = (1-a)*k1*np.exp(-k1*x) + a*k2*np.exp(-k2*x) # This is your y axis fit-line

            axs[i].plot(x, curvey,color='darkorange')
            axs[i].scatter(x,y,color='black')
            axs[i].set_ylim(10**(-2),10**(5))
            axs[i].set_xlim(bins[0], 10**(-2.5))
            plt.legend(loc='best')
            plt.setp(axs[i].get_xticklabels(), fontsize=20)
            plt.setp(axs[i].get_yticklabels(), fontsize=20)
        plt.gca().legend(('_nolegend_','_nolegend_','_nolegend_','1,12,17,20','1q','normal'),fontsize=16)
        plt.show()

        [gen1,gen2] = generations_to_plot
        Data1 = self.X_X[gen1, :] / (self.X_X[gen1, :].sum())
        Data1 = Data1[Data1 > 0]
        Data2 = self.X_X[gen2, :] / (self.X_X[gen2, :].sum())
        Data2 = Data2[Data2 > 0]

        print(stats.ks_2samp(Data1, Data2))

        pass
