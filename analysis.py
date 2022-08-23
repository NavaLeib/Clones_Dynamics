

import  numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy import stats

from clones_dynamic_functions import Model

from decimal import *


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
        #plt.show()
        plt.savefig('figure_ratio_generation_many_mutants.pdf')
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
                             +str(round(ratio[1][gen_num],2)), fontsize=20)

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
        #plt.show()
        plt.savefig('figure_hist_many_mutants.pdf')

        [gen1,gen2] = generations_to_plot
        Data1 = self.X_X[gen1, :] / (self.X_X[gen1, :].sum())
        Data1 = Data1[Data1 > 0]
        Data2 = self.X_X[gen2, :] / (self.X_X[gen2, :].sum())
        Data2 = Data2[Data2 > 0]

        print(stats.ks_2samp(Data1, Data2))

        print(stats.ks_2samp(rP, curvey))

        pass


    def figure_hist_many_mutants_all_pass(self, number_of_mutations, generations_to_plot,percent_of_pass,initial_mean_clone_size, cells):

        """
        generate from data stacked histogram for different generation
        :argument:
        number_of_mutation = number of mutations specified in teh model (int)
        generations_to_plot = array of generations to plot [gen_1,gen_2,....] (int, int,.. ) --> currently [gen1, gen2]
        :return:
        """


        ratio = []
        for mut in range(number_of_mutations):
            ratio.append([self.X_X[i, int(mut * self.number_of_species * self.mutant_percent):
                                      int((mut + 1) * self.number_of_species * self.mutant_percent)].sum(axis=0)
                          / (self.X_X[i, :].sum(axis=0)) for i in range(self.gen_num + 1)])

        fig, axs = plt.subplots(2, 2)
        fig.supxlabel('clone size (fraction of total cells)', fontsize=10)
        fig.supylabel('clones counts', fontsize=10)

        for (index, gen_num) in enumerate(generations_to_plot):
            location = [(0, 0), (0, 1), (1, 0), (1, 1)]
            (i, j) = location[index]

            Data = self.X_X[gen_num, :] / (self.X_X[gen_num, :].sum())
            Data = Data[Data > 0]

            min_data = min(Data)
            max_data = max(Data)
            print(min_data,max_data)

            bins = 10 ** (np.arange(np.log10(min_data), np.log10(max_data), 0.1))
            counts, bins, bars = axs[i, j].hist([Data[:int(self.number_of_species * self.mutant_percent / 2)],
                                                 Data[int(self.number_of_species * self.mutant_percent / 2):
                                                      int(self.number_of_species * self.mutant_percent)],
                                                 Data[int(self.number_of_species * self.mutant_percent):]],
                                                density=True,
                                                bins= bins, alpha=0.5,
                                                histtype='barstacked',
                                                color=('orange', 'saddlebrown', 'blue'))
            counts, bins, bars = axs[i, j].hist(Data, density=True, bins=bins, alpha=0.0)
            # plt.plot(bins[1:]-np.diff(bins)/2,counts ,'.')
            axs[i, j].set_yscale('log')
            axs[i, j].set_xscale('log')
            axs[i, j].set_title("$\\bf{1,12,17,20}$: " + str(round(ratio[0][gen_num], 2)) + "; $\\bf{1q}$: "
                                + str(round(ratio[1][gen_num], 2)), fontsize=10)

            from scipy.optimize import curve_fit

            x = bins[1:] - np.diff(bins) / 2

            P = ss.expon.fit(Data)
            #rX = np.linspace(min(Data), max(Data), 1000)
            rP = ss.expon.pdf(x, *P)
            axs[i, j].plot(x, rP, color='blue')


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

            axs[i,j].plot(x, curvey,color='darkorange')
            axs[i,j].scatter(x,y,color='black')
            axs[i,j].set_ylim(10**(-2),10**(5))
            axs[i,j].set_xlim(bins[0], 10**(-2.5))
            plt.legend(loc='best')
            plt.setp(axs[i,j].get_xticklabels(),fontsize=10)
            plt.setp(axs[i,j].get_yticklabels(),fontsize=10)


            gen2 = gen_num
            Data1 = self.X_X[1, :] / (self.X_X[1, :].sum())
            Data1 = Data1[Data1 > 0]
            Data2 = self.X_X[gen2, :] / (self.X_X[gen2, :].sum())
            Data2 = Data2[Data2 > 0]

            #(ks, p) = stats.ks_2samp(Data1, Data2)

            # axs[i, j].set_title('H0: PDF(gen=1) = PDF (gen=' + str(gen2) + '); p_value = ' + f"{Decimal(str(p)):.2E} \n " +
            #                     f"$\\bf{1,12,17,20}$: " + str(round(ratio[0][gen_num], 2)) + "; $\\bf{1q}$: "
            #                     + str(round(ratio[1][gen_num], 2)))

            P = ss.expon.fit(Data2)
            rX = np.linspace(min(Data2), max(Data2), 1000)
            rP = ss.expon.cdf(rX, *P)
            (ks,p) = stats.kstest(Data2,rP,N=10**5)
            axs[i, j].set_title(
                #'H0: PDF (day=' + str(gen2) + ') = exp. dist; p_value = ' + f"{Decimal(str(p)):.3E} \n " +
                'PDF (day=' + str(gen2) + ')\n   ' + f"$\\bf{1, 12, 17, 20}$: " + str(round(ratio[0][gen_num], 2)) + "; $\\bf{1q}$: " +
                 str(round(ratio[1][gen_num], 2)),fontsize=10)

            #print('between two measuremnts', stats.ks_2samp(Data1, Data2), 'gen=', gen2)

            #print('between two fitting curves', stats.ks_2samp(rP, curvey), 'gen=', gen2)

            print('between two fit exp curve to data ', (ks,p), 'gen=', gen2)

        plt.gca().legend(('_nolegend_', '_nolegend_', '_nolegend_', '1,12,17,20', '1q', 'normal'), fontsize=10)
        plt.suptitle('percent_of_pass=' +str(percent_of_pass) + '; initial_mean_clone_size=' +str(initial_mean_clone_size)
                     + '; cells='  +str(cells), fontsize=10)

        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.9,
                            top=0.85,
                            wspace=0.25,
                            hspace=0.4)

        plt.show()
        plt.savefig('figure_hist_many_mutants' +'percent_of_pass' +str(percent_of_pass) + '_initial_mean_clone_size' +str(initial_mean_clone_size)
                     + '_cells'  +str(cells)+ '_day5.pdf')

        pass

    def figure_number_clones(self, passaging, initial_mean_clone_size, percent_of_pass, cells):

        (ks,p)=stats.ks_2samp(Data1, Data2)

        axs[i, j].set_title('H0: PDF(gen=1) = PDF (gen='+str(gen2)+'); p_value = ' +str(p))
        print('between two measuremnts', stats.ks_2samp(Data1, Data2), 'gen=', gen2)

        print('between two fitting curves', stats.ks_2samp(rP, curvey),'gen=', gen2)

        plt.gca().legend(('_nolegend_','_nolegend_','_nolegend_','1,12,17,20','1q','normal'),fontsize=10)
        plt.show()
        #plt.savefig('figure_hist_many_mutants.pdf')
        pass

    def figure_number_clones(self, passaging, initial_mean_clone_size):

        """
        generate from data plot the number of clones in each passaging
        :param passaging: int
        :return:
        """
        num_non_zero = [((self.X_X[i, :])!=0).sum(axis=0) for i in range(self.gen_num + 1)]
        print(num_non_zero)
        mutant_percent = self.mutant_percent
        number_of_species =  self.number_of_species
        percent_of_pass = mutant_percent
        cells = number_of_species * initial_mean_clone_size

        plt.plot(num_non_zero[::passaging], 'o', label = str(percent_of_pass) )
        plt.xlabel('# passaging')
        plt.legend(title='percent to pass')
        plt.ylabel('# unique barcodes = # clones')

        plt.title('cells=' + f"{Decimal(str(cells)):.1E};   " 'init clone size=' +
                  f"{Decimal(str(initial_mean_clone_size)):.1E};    "+ '  passaging=' + str(passaging) + 'days'
                   )
        plt.show()
        # print('figure_number_clones_init_cells_per_clone' + str(initial_mean_clone_size) + '.pdf')
        plt.savefig('figure_number_clones_' + 'cells' + str(cells)+ '_init clone size' +
                  str(initial_mean_clone_size)+ '_passaging' + str(passaging) + '_Vary_PercentY.pdf')


        pass
