#!/usr/local/bin/python

# import matplotlib
# matplotlib.use('TkAgg') #Uncomment this to get Tk plots


import numpy as np
import random
from prototype_ebm.file_readers import adniFileReader
from prototype_ebm.mcmc import MCMC, bootstrapMCMC
from prototype_ebm.mixture_model import MixtureModel
from prototype_ebm.distribution import Distribution
from prototype_ebm.plotting import uncertaintyMatrix, plotMixtureModels, plotAllPatientStages, plotAllPatientStages_new
from matplotlib import pyplot as plt
from builtins import range

def main(doBootstrap=False):
    allData, allLabels, bioHeaders = adniFileReader()
    nBioMarkers = allData.shape[1]
    allModels = []
    for i in range(nBioMarkers):
        hModel = Distribution()
        dModel = Distribution()
        bioMixture = MixtureModel(healthyModel=hModel,
                                  diseaseModel=dModel)
        bioMixture.fit(allData[:, i], allLabels)
        allModels.append(bioMixture)

    fig1, ax1 = plotMixtureModels(allData, allLabels, bioHeaders, allModels)
    fig1.show()
    finalOrders = MCMC(allData, allLabels, allModels)
    bestSeq = np.array(finalOrders[0][1])

    fig2, ax2 = uncertaintyMatrix(finalOrders, bioHeaders, bestSeq=bestSeq)
    fig2.show()
    fig3, ax3 = plotAllPatientStages_new(allData, allLabels, allModels,
                                     bestSeq)
    fig3.show()
    if(doBootstrap):
        finalOrders = bootstrapMCMC(allData, allLabels, allModels,
                                    nBootstraps=5, plot=False)
        fig4, ax4 = uncertaintyMatrix(finalOrders, bioHeaders,
                                      bestSeq=bestSeq, bootstrap=True)
        fig4.show()
    plt.show()

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main(False)
