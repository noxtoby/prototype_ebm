import numpy as np
import itertools
from prototype_ebm.plotting import greedyAscentTrace
from prototype_ebm.mixture_model import getProbMatrix
from builtins import range


def createBootstrap(inData, healthyDiseaseLabels):
    """Summary

    Parameters
    ----------
    inData : array-like, shape(nPatients, nBiomarkers)
        Data to create bootstraps from.
    healthyDiseaseLabels : array-like, shape(nPatients)
        Diagnosis labels to create bootstraps from.

    Returns
    -------
    bootstrapData : array-like, shape(nPatients, nBiomarkers)
        A bootstrap of the patient data, i.e. a random sample with
        replacement of the same size of inData.
    bootstrapLabels : array-like, shape(nPatients)
        A bootstrap of the patient diagnoses, i.e. a random sample with
        replacement of the same size of healthyDiseaseLabels.
    """
    bootstrapData = np.empty(inData.shape)
    bootstrapLabels = np.empty(inData.shape[0])

    nLabels = np.bincount(healthyDiseaseLabels).shape[0]
    # Ensure there is at least one of each class in the strap
    for i in range(nLabels):
        idx = np.random.choice(np.where(healthyDiseaseLabels == i)[0], 1)
        bootstrapData[i] = inData[idx]
        bootstrapLabels[i] = i

    for i in range(nLabels, inData.shape[0]):
        idx = np.random.randint(inData.shape[0])
        bootstrapData[i] = inData[idx]
        bootstrapLabels[i] = healthyDiseaseLabels[idx]

    return bootstrapData, bootstrapLabels


def bootstrapMCMC(inData, healthyDiseaseLabels, mixtureModel,
                  nBootstraps=50, **kwargs):
    """A function that wraps the MCMC and the bootstraps together.
    Needs to be updated as the output of the MCMC has been changed to
    gather more data.

    Parameters
    ----------
    inData : array-like, shape(nPatients, nBiomarkers)
        Biomarker data for each of the patients.
    healthyDiseaseLabels : array-like, shape(nPatients)
        Diagnosis label associated with each of the patients.
    mixtureModels : array-like, shape(nBiomarkers)
        The MixtureModel models for each of the biomarkers.
    nBootstraps : int, optional
        Number of bootstaps to compute the maximum likelihood sequence
        for.
    **kwargs
        Arguments to be passed to the MCMC function.
    """
    allSamples = []
    for i in range(nBootstraps):
        bootLab, bootHDLabels = createBootstrap(inData, healthyDiseaseLabels)
        MCSamples = MCMC(bootLab, bootHDLabels, mixtureModel, **kwargs)
        allSamples += MCSamples
    allSamples.sort(reverse=True, key=lambda dum: dum[0])
    return allSamples


def scoreEventOrder(eventOrder, probMatrix):
    """Gives the likelihood of an event ordering based on the probability matrix
    generated from the patient data.

    Parameters
    ----------
    eventOrder : array-like, shape(nBiomarkers,)
        Event ordering to be scored.
    probMatrix : array-like, shape(nPatients, nBiomarkers, 2)
        Probability for a normal/abnormal measurement in all biomarkers
        for all patients (and controls).

    Returns
    -------
    likelihood : float
        likelihood score for the given event ordering. Based on the patients
        probability matrix provided.
    """
    pYes = np.array(probMatrix[:, eventOrder, 1])
    pNo = np.array(probMatrix[:, eventOrder, 0])

    k = probMatrix.shape[1] + 1
    pPermK = np.zeros((probMatrix.shape[0], k))

    for i in range(k):
        pPermK[:, i] = np.prod(pYes[:, :i], 1) * np.prod(pNo[:, i:k - 1], 1)

    likelihood = np.sum(np.log(np.sum((1. / k) * pPermK, 1) + 1e-250))

    return likelihood


def swapTwoEvents(eventOrder):
    """Randomly swaps two events in the event sequence.

    Parameters
    ----------
    eventOrder : array-like, shape(nBiomarkers,)
        Event ordering.

    Returns
    -------
    newEventOrder : array-like, shape(nBiomarkers,)
        Event ordering with two random swapped events wrt to the input,
        eventOrder.
    """
    newEventOrder = eventOrder.copy()
    bmToSwap = np.random.choice(eventOrder.shape[0], 2, replace=False)
    newEventOrder[bmToSwap] = newEventOrder[bmToSwap[::-1]]

    return newEventOrder


def greedyCreation(probMat, nIterations=1000, nStartPoints=10):
    """Greedy descent algorithm for the creation of a likely sequence.
    All event orders created are output, to ensure heterogeneity in the
    genetic algorithm start point.

    Parameters
    ----------
    probMat : array-like, shape(nPatients, nBiomarkers, 2)
        Probability for a normal/abnormal measurement in all biomarkers
        for all patients (and controls).
    nIterations : int, optional
        Number of greedy descent steps to perform.
    nStartPoints : int, optional
        Number of random start points to initiate greedy ascent algorithm.

    Returns
    -------
    samplesMat : array-like, shape(nStartPoints, nIterations, nBiomarkers)
        Array of all the event orderings generated during the greedy ascent
        algorithm. i,j-th element corresponds to the event ordering at the j-th
        iteration of the i-th random start point.
    likelihoodMat : array-like, shape(nStartPoints, nIterations)
        Array of all the event orderings generated during the greedy ascent
        algorithm. i,j-th element corresponds to the likilihood of the event
        ordering at the j-th iteration of the i-th random start point.
    """
    nBiomarkers = probMat.shape[1]
    likelihoodMat = np.empty((nStartPoints, nIterations))
    samplesMat = np.empty((nStartPoints, nIterations, nBiomarkers),
                          dtype=int)
    for nStarts in range(nStartPoints):
        currentOrder = np.arange(nBiomarkers)
        np.random.shuffle(currentOrder)

        likelihoodMat[nStarts, 0] = scoreEventOrder(currentOrder, probMat)
        samplesMat[nStarts, 0] = currentOrder

        for iterNum in range(1, nIterations):
            newOrder = swapTwoEvents(samplesMat[nStarts, iterNum - 1])
            newScore = scoreEventOrder(newOrder, probMat)
            if (newScore > likelihoodMat[nStarts, iterNum - 1]):
                likelihoodMat[nStarts, iterNum] = newScore
                samplesMat[nStarts, iterNum] = newOrder
            else:
                likelihoodMat[nStarts, iterNum] = likelihoodMat[nStarts,
                                                                iterNum - 1]
                samplesMat[nStarts, iterNum] = samplesMat[nStarts, iterNum - 1]
    return samplesMat, likelihoodMat


def MCMC(inData, healthyDiseaseLabels, mixtureModels,
         nIterations=100000, nGreedyIt=1000,
         nStartPoints=10, plot=True, cleanData=False):
    """Summary

    Parameters
    ----------
    inData : array-like, shape(nPatients, nBiomarkers)
        Biomarker data for each of the patients.
    healthyDiseaseLabels : array-like, shape(nPatients)
        Diagnosis label associated with each of the patients.
    mixtureModels : array-like, shape(nBiomarkers)
        The MixtureModel models for each of the biomarkers.
    nIterations : int, optional
        Number of iterations to perform of the MCMC sampling.
    nGreedyIt : int, optional
        Number of iterations to perform of the greedy ascent algorithm.
    nStartPoints : int, optional
        Number of random start points for the greedy ascent algorithm.
    plot : bool, optional
        Description

    Returns
    -------
    finalOrders : array-like, shape(nIterations, 2)
        Sorted array of the the event-orderings and their scores for each of
        the MCMC iterations.
    """
    probMat = getProbMatrix(inData, mixtureModels, healthyDiseaseLabels,
                            cleanData=cleanData)
    nBiomarkers = probMat.shape[1]

    samplesMat, likelihoodMat = greedyCreation(probMat, nGreedyIt,
                                               nStartPoints)
    if (plot):
        fig, ax = greedyAscentTrace(likelihoodMat)
        fig.show()

    maxIdx = np.unravel_index(np.argmax(likelihoodMat), likelihoodMat.shape)
    currentScore = likelihoodMat[maxIdx]
    currentOrder = samplesMat[maxIdx]

    likelihoodMat = np.zeros(nIterations)
    samplesMat = np.zeros((nIterations, nBiomarkers), dtype=int)

    likelihoodMat[0] = currentScore
    samplesMat[0] = currentOrder
    for i in range(1, nIterations):
        samplesMat[i] = swapTwoEvents(samplesMat[i - 1])
        likelihoodMat[i] = scoreEventOrder(samplesMat[i], probMat)
        ratio = np.exp(likelihoodMat[i] - likelihoodMat[i - 1])
        if (ratio < np.random.random()):
            likelihoodMat[i] = likelihoodMat[i - 1]
            samplesMat[i] = samplesMat[i - 1]
    finalOrders = list(reversed(list(zip(likelihoodMat, samplesMat))))
    finalOrders.sort(reverse=True, key=lambda dum: dum[0])
    return finalOrders


def enumerateAll(inData, healthyDiseaseLabels, mixtureModels):
    """Enumerates all possible permutations of event orders, scores them all
    and sorts them based on the scores.

    Parameters
    ----------
    inData : array-like, shape(nPatients, nBiomarkers)
        Raw data of patient biomarker data.
    healthyDiseaseLabels : array-like, shape(nPatients,)
        Clinical diagnosis for patients, 0 for healthy, 1 for diseased,
        and 2 for other.
    mixtureModels : array-like, shape(nBiomarkers,)
        List of mixtureModel objects which have already been fit.

    Returns
    -------
    outPop : array-like, shape(nBiomarkers!, 2)
        Sorted array of all the possible event-orderings and their
        corresponding score.
    """
    nBiomarkers = inData.shape[1]
    currentPopulation = itertools.permutations(range(nBiomarkers))
    probMat = getProbMatrix(inData, mixtureModels, healthyDiseaseLabels)
    outPop = []
    for i in currentPopulation:
        outPop.append((scoreEventOrder(i, probMat), i))
    outPop.sort(reverse=True, key=lambda dum: dum[0])
    return outPop
