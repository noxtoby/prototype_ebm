import numpy as np
from matplotlib import pyplot as plt
import math
from prototype_ebm.mixture_model import getProbMatrix, patientStaging, patientStaging_new
from builtins import range

def uncertaintyMatrix(finalOrders, bioHeaders, bestSeq=None, bootstrap=False):
    """Plots a uncertainty matrix from the output of the MCMC
    algorithm.

    Parameters
    ----------
    finalOrders : array-like, shape(numMCIters, 2)
        Description
    bioHeaders : array-like, shape(nBiomarkers)
        Description
    bestSeq : array-like, shape(nBiomarkers)
        Description

    Returns
    -------
    fig : Figure
        Matplotlib figure object with a single subplot
    ax : AxesSubplot
        Matplotlib axis oject with the uncertainty matrix plot
        for the MCMC samples.
    """

    if(bestSeq is None):
        bestSeq = np.array(finalOrders[0][1])
        finalOrders = finalOrders[1:]
    nBioMarkers = len(bestSeq)
    _, finalOrders = zip(*finalOrders)
    finalOrders = np.array(finalOrders)

    confusionMat = np.zeros((nBioMarkers, nBioMarkers))
    for i in range(nBioMarkers):
        confusionMat[i, :] = np.sum(finalOrders == bestSeq[i], axis=0)

    fig, ax = plt.subplots()

    ax.imshow(confusionMat, interpolation='nearest', cmap=plt.cm.Blues)
    if(bootstrap):
        ax.set_title('Bootstrapped EBM Output')
    else:
        ax.set_title('EBM Output')
    tick_marks = np.arange(nBioMarkers)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(range(1, nBioMarkers+1), rotation=45)

    ax.set_yticks(tick_marks)
    ax.set_yticklabels(np.array(bioHeaders, dtype='object')[bestSeq],
                       rotation=30, ha='right',
                       rotation_mode='anchor')

    ax.set_ylabel('Biomarker Name')
    ax.set_xlabel('Event Order')
    return fig, ax


def greedyAscentTrace(likelihoodMat):
    """Plots the trace of each of the random start points used
    in the greey ascent algorithm.

    Parameters
    ----------
    likelihoodMat : array-like, shape(nStartPoints, nGreedyIt)
        The likelihood matrix generted by the greedy ascent algorithm.

    Returns
    -------
    fig : Figure
        Matplotlib figure object with a single subplot
    ax : AxesSubplot
        Matplotlib axis oject with the trace plot for the greedy
        ascent algorithm.
    """
    fig, ax = plt.subplots(1, 1)
    for i in range(likelihoodMat.shape[0]):
        ax.plot(np.arange(likelihoodMat.shape[1]), likelihoodMat[i])
    ax.set_ylabel('Likelihood')
    ax.set_xlabel('Iteration number')
    return fig, ax


def plotMixtureModels(inData, healthyDiseaseLabels, bioHeaders, mixtureModels):
    """Plots a figure which contains each of the biomarker distributions and the
    fitted mixture models for each biomarker.

    Parameters
    ----------
    inData : array-like, shape(nPatients, nBiomarkers)
        All patient-all biomarker measurements.
    labels : array-like, shape(nPatients,)
        Diagnosis labels for each of the patients.
    bioHeaders : array-like, shape(nBiomarkers,)
        List of the names of the biomarkers
    mixtureModels : array-like, shape(nBiomarkers,)
        List of fit mixture models for each of the biomarkers

    Returns
    -------
    None
    """
    nBioMarkers = len(bioHeaders)
    numY, numX = (int(math.ceil(math.sqrt(nBioMarkers))),
                  int(round(math.sqrt(nBioMarkers))))
    fig, ax = plt.subplots(numX, numY)

    for i in range(nBioMarkers):
        mixtureMod = mixtureModels[i]
        indivBM = inData[healthyDiseaseLabels != 2, i]

        histDat = [inData[healthyDiseaseLabels == 0, i],
                   inData[healthyDiseaseLabels == 1, i]]
        barColors = ['g', 'b']
        histDat = [x[~np.isnan(x)] for x in histDat]

        ax[math.floor((i)/numY), i % numY].hist(histDat, histtype='bar',
                                                stacked=True, fill=True,
                                                color=barColors, bins=15)

        indivBM = indivBM[~np.isnan(indivBM)]

        xSpace = np.linspace(min(indivBM), max(indivBM), 500).reshape(-1, 1)

        hGauss = mixtureMod.hModel.getPDF(xSpace)
        hGauss *= (mixtureMod.mix * len(indivBM)/3) / max(hGauss)

        dGauss = mixtureMod.dModel.getPDF(xSpace)
        dGauss *= ((1-mixtureMod.mix) * len(indivBM)/3) / max(dGauss)

        ax[math.floor((i)/numY), i % numY].plot(xSpace, hGauss, 'y-')
        ax[math.floor((i)/numY), i % numY].plot(xSpace, dGauss, 'r-')
        ax[math.floor((i)/numY), i % numY].set_title(bioHeaders[i])
    plt.draw()
    return fig, ax


def plotAllPatientStages(inData, healthyDiseaseLabels, allModels,
                         sequence):
    """Plots a bar graph for the event stages, based on the best ordering
    from MCMC sampling. Patients with label  0 are assumed to be healthy, 1
    are diseased and 2 are other data to be staged.

    Parameters
    ----------
    inData : array-like, shape(nPatients, nBiomarkers)
        All patient-all biomarker measurements.
    healthyDiseaseLabels : array-like, shape(nPatients,)
        Diagnosis labels for each of the patients.
    allModels : array-like, shape(nBiomarkers,)
        List of fit mixture models for each of the biomarkers
    sequence : array-like, shape(nBiomarkers,)
        Event ordering for the biomarkers.

    Returns
    -------
    None
    """

    probMat = getProbMatrix(inData, allModels, healthyDiseaseLabels,
                            cleanData=False)
    nPatients = probMat.shape[0]
    patientStages = np.zeros(nPatients)
    patientStages, _ = patientStaging(probMat, sequence)

    patientStages = patientStages.astype(int)

    healthyStages = patientStages[healthyDiseaseLabels == 0]
    diseaseStages = patientStages[healthyDiseaseLabels == 1]
    unlabelledStages = patientStages[healthyDiseaseLabels == 2]

    numBins = sequence.shape[0]+1
    healthyStages = np.bincount(healthyStages, minlength=numBins)
    diseaseStages = np.bincount(diseaseStages, minlength=numBins)
    if(len(unlabelledStages)):
        unlabelledStages = np.bincount(unlabelledStages, minlength=numBins)
        maxEle = max([max(healthyStages), max(diseaseStages),
                      max(unlabelledStages)])
    else:
        unlabelledStages = []
        maxEle = max([max(healthyStages), max(diseaseStages)])

    healthyStages = healthyStages/float(sum(healthyStages))
    diseaseStages = diseaseStages/float(sum(diseaseStages))
    if(len(unlabelledStages)):
        unlabelledStages = unlabelledStages/float(sum(unlabelledStages))
        maxTick = max([max(healthyStages), max(diseaseStages),
                       max(unlabelledStages)])
    else:
        maxTick = max([max(healthyStages), max(diseaseStages)])
    healthyStages = [math.ceil(x*maxEle) for x in healthyStages]
    diseaseStages = [math.ceil(x*maxEle) for x in diseaseStages]
    if(len(unlabelledStages)):
        unlabelledStages = [math.ceil(x*maxEle) for x in unlabelledStages]
    idxs = np.array(range(len(healthyStages)))
    width = 0.25
    fig, ax = plt.subplots()
    ax.bar(idxs, healthyStages, width, color='b', label='HC')
    if(len(unlabelledStages)):
        ax.bar(idxs+width, unlabelledStages, width, color='k', label='other (MCI)')
    ax.bar(idxs+2*width, diseaseStages, width, color='r', label='AD')

    ax.set_ylabel('Proportion')
    ax.set_title('Patient stages')
    ax.set_xticks(idxs+width)
    ax.set_xticklabels([str(x) for x in idxs])

    y_labels = [str(x) for x in np.arange(0, maxTick+0.1, 0.1)]
    y_ticks = np.linspace(0, maxEle, len(y_labels))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    return fig, ax

def plotAllPatientStages_new(inData, healthyDiseaseLabels, allModels,
                         sequence, option=1):
    """Calls patientStaging_new() instead of patientStaging()
    
    See plotAllPatientStages()
    
    """

    probMat = getProbMatrix(inData, allModels, healthyDiseaseLabels,
                            cleanData=False)
    nPatients = probMat.shape[0]
    patientStages = np.zeros(nPatients)
    _,_,patientStages = patientStaging_new(probMat, sequence, option=option)

    patientStages = patientStages.astype(int)

    healthyStages = patientStages[healthyDiseaseLabels == 0]
    diseaseStages = patientStages[healthyDiseaseLabels == 1]
    unlabelledStages = patientStages[healthyDiseaseLabels == 2]

    numBins = sequence.shape[0]+1
    healthyStages = np.bincount(healthyStages, minlength=numBins)
    diseaseStages = np.bincount(diseaseStages, minlength=numBins)
    if(len(unlabelledStages)):
        unlabelledStages = np.bincount(unlabelledStages, minlength=numBins)
        maxEle = max([max(healthyStages), max(diseaseStages),
                      max(unlabelledStages)])
    else:
        unlabelledStages = []
        maxEle = max([max(healthyStages), max(diseaseStages)])

    healthyStages = healthyStages/float(sum(healthyStages))
    diseaseStages = diseaseStages/float(sum(diseaseStages))
    if(len(unlabelledStages)):
        unlabelledStages = unlabelledStages/float(sum(unlabelledStages))
        maxTick = max([max(healthyStages), max(diseaseStages),
                       max(unlabelledStages)])
    else:
        maxTick = max([max(healthyStages), max(diseaseStages)])
    healthyStages = [math.ceil(x*maxEle) for x in healthyStages]
    diseaseStages = [math.ceil(x*maxEle) for x in diseaseStages]
    if(len(unlabelledStages)):
        unlabelledStages = [math.ceil(x*maxEle) for x in unlabelledStages]
    idxs = np.array(range(len(healthyStages)))
    width = 0.25
    fig, ax = plt.subplots()
    ax.bar(idxs, healthyStages, width, color='b', label='HC')
    if(len(unlabelledStages)):
        ax.bar(idxs+width, unlabelledStages, width, color='k', label='other (MCI)')
    ax.bar(idxs+2*width, diseaseStages, width, color='r', label='AD')

    ax.set_ylabel('Proportion')
    ax.set_title('Patient stages')
    ax.set_xticks(idxs+width)
    ax.set_xticklabels([str(x) for x in idxs])

    y_labels = [str(x) for x in np.arange(0, maxTick+0.1, 0.1)]
    y_ticks = np.linspace(0, maxEle, len(y_labels))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    return fig, ax


def plot_pvd_new(finalOrders, bioHeaders, bestSeq=None, bootstrap=False, option='cumulative'):
    """Plots the MCMC positional mass diagram, a.k.a. positional variance diagram,
    in different ways:
    
    option = 'cumulative'
      plots the cumulative sum along each row as a separate curve
    
    algorithm.

    Returns
    -------
    fig,ax : figure and axis objects from matplotlib
    pmm : positional mass matrix
    """

    if(bestSeq is None):
        bestSeq = np.array(finalOrders[0][1])
        finalOrders = finalOrders[1:]
    nBioMarkers = len(bestSeq)
    _, finalOrders = zip(*finalOrders)
    finalOrders = np.array(finalOrders)

    pmm = np.zeros((nBioMarkers, nBioMarkers))
    for i in range(nBioMarkers):
        pmm[i, :] = np.sum(finalOrders == bestSeq[i], axis=0)

    fig, ax = plt.subplots()

    if option=='standard':
        ax.imshow(pmm, interpolation='nearest', cmap=plt.cm.Blues)
    elif option=='cumulative':
        fig,(ax,ax2)=plt.subplots(1,2)
        ax.imshow(np.cumsum(pmm,axis=1), interpolation='nearest', cmap=plt.cm.Blues)
        ax2.plot(np.cumsum(pmm,axis=1).T)
        ax2.legend(bioHeaders[bestSeq])
    if(bootstrap):
        ax.set_title('Bootstrapped EBM Output')
    else:
        ax.set_title('EBM Output')
    tick_marks = np.arange(nBioMarkers)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(range(1, nBioMarkers+1), rotation=45)

    ax.set_yticks(tick_marks)
    ax.set_yticklabels(np.array(bioHeaders, dtype='object')[bestSeq],
                       rotation=30, ha='right',
                       rotation_mode='anchor')

    ax.set_ylabel('Biomarker Name')
    ax.set_xlabel('Event Order')
    return fig, ax, pmm
