from scipy import optimize
import numpy as np
from builtins import range

class MixtureModel():
    """Wraps up two distributions and the mixture parameter.

    Attributes
    ----------
    dModel : distribution
        Distribution object to use for the diseased data.
    hModel : distribution
        Distribution object to use for the healthy data.
    mix : float
        Mixing fraction, as percent of healthy patients.
    """
    def __init__(self, healthyModel=None, diseaseModel=None, mixture=None):
        """Initiate new MixtureModel object

        Parameters
        ----------
        healthyModel : distribution, optional
            Distribution object to use for the healthy data.
        diseaseModel : TYdistributionPE, optional
            Distribution object to use for the diseased data.
        mixture : float, optional
            Mixing fraction, as percent of healthy patients.
        """
        self.hModel = healthyModel
        self.dModel = diseaseModel
        self.mix = mixture

    def calculateLikelihoodMixtureModel(self, theta, inData):
        """"Calculates the likelihood of the data given the model
        parameters scored in theta. theta should contain normal mean,
        normal standard deviation, abnormal mean, abnormal standard
        deviation and the fraction of the data that is normal

        Parameters
        ----------
        theta : array-like, shape(5,)
            List containing the parameters required for a mixture model.
            [hModelMu, hModelSig, dModelMu, dModelSig, mixture]
        inData : array-like, shape(numPatients,)
            Biomarker measurements for patients.

        Returns
        -------
        likelihood : float
            Negative log likelihood of the data given the parameters theta.
        """
        mixture = theta[-1]
        # thetaNums allows us to use other distributions with a varying
        # number of paramters. Not included in this version of the code.
        thetaNums = {'norm': 2}
        if(len(theta[np.isnan(theta)])):
            return 1e100

        hTheta = theta[:thetaNums['norm']]
        dTheta = theta[thetaNums['norm']:-1]

        self.hModel.setTheta(hTheta)
        self.dModel.setTheta(dTheta)

        likeNorm1 = self.hModel.getPDF(inData)*mixture
        likeNorm2 = self.dModel.getPDF(inData)*(1-mixture)

        likeDPoints = likeNorm1 + likeNorm2
        likeDPoints[likeDPoints == 0] = np.finfo(float).eps
        likeDPoints = np.log(likeDPoints)
        return -1*np.sum(likeDPoints)

    def getProb(self, inData):
        """Get the probability of some data based on the mixture model

        Parameters
        ----------
        inData : array-like, shape(numPatients,)
            Biomarker measurements for patients.

        Returns
        -------
        hProb : array-like, shape(numPatients, 2)
            Probability of patients biomarkers being normal according to the
            MixtureModel.
        dProb : array-like, shape(numPatients, 2)
            Probability of patients biomarkers being abnormal according to the
            MixtureModel.
        """
        hLike = self.hModel.getPDF(inData)
        dLike = self.dModel.getPDF(inData)
        # This seems to be a problem when the fit goes completely wrong
        if(hLike == 0 and dLike == 0):
            return 0, 0
        hProb = hLike/(hLike+dLike)
        return hProb, 1-hProb

    def fit(self, inData, healthyDiseaseLabels):
        """This will fit a mixture model to some given data. Labelled data
        is used to derive starting conditions for the optimize function,
        labels are 0 for normal and 1 for abnormal. The model type corresponds
        to the type of distributions used, currently there is normal and
        uniform distributions. Be careful when chosing distributions as the
        optimiser can throw out NaNs.

        Parameters
        ----------
        inData : array-like, shape(numPatients,)
            Biomarker measurements for patients.
        healthyDiseaseLabels : array-like, shape(numPatients,)
            Diagnosis labels for each of the patients.

        Returns
        -------
        mixInfoOutput : array-like, shape(5,)
            List containing the parameters required for a mixture model.
            [hModelMu, hModelSig, dModelMu, dModelSig, mixture]
        """
        healthyData = inData[healthyDiseaseLabels == 0]
        diseaseData = inData[healthyDiseaseLabels == 1]

        eventSign = np.nanmean(healthyData) < np.nanmean(diseaseData)
        np.seterr(all='ignore')
        tempData = np.append(healthyData, diseaseData)
        indivBM = tempData[~np.isnan(tempData)]
        optBounds = []
        optBounds += self.hModel.getBounds(inData, healthyData, eventSign)
        optBounds += self.dModel.getBounds(inData, diseaseData, not eventSign)
        optBounds += [(0.1, 0.99)]
        firstGuess = []
        hStart = self.hModel.getStart(healthyData)
        firstGuess += hStart
        firstGuess += self.dModel.getStart(diseaseData)
        firstGuess += [0.5]
        mixInfoOutput = optimize.minimize(self.calculateLikelihoodMixtureModel,
                                          firstGuess, args=(indivBM, ),
                                          bounds=optBounds,
                                          method = 'SLSQP')
        mixInfoOutput = mixInfoOutput.x
        if(np.isnan(np.sum(mixInfoOutput))):
            mixInfoOutput = optimize.minimize(self.calculateLikelihoodMixtureModel,
                                              firstGuess, args=(indivBM, ),
                                              bounds=optBounds)
            mixInfoOutput = mixInfoOutput.x
        self.hModel.setTheta(mixInfoOutput[:len(hStart)])
        self.dModel.setTheta(mixInfoOutput[len(hStart):-1])
        self.mix = mixInfoOutput[-1]
        return mixInfoOutput


def getProbMatrix(inData, mixtureModels, healthyDiseaseLabels, cleanData=True):
    """Gives the matrix of probabilities that a patient has normal/abnormal
    measurements for each of the biomarkers. Output is number of patients x
    number of biomarkers x 2.

    Parameters
    ----------
    inData : array-like, shape(numPatients, numBiomarkers)
        All patient-all biomarker measurements.
    mixtureModels : array-like, shape(numBiomarkers,)
        List of fit mixture models for each of the biomarkers.
    healthyDiseaseLabels : array-like, shape(numPatients,)
        Diagnosis labels for each of the patients.
    cleanData : bool, optional
        Whether or not to include the non binary diagnoses. True if leave
        out non-binary and False to include all diagnoses.

    Returns
    -------
    outProbs : array-like, shape(numPatients, numBioMarkers, 2)
        Probability for a normal/abnormal measurement in all biomarkers
        for all patients (and controls).
    """

    if(cleanData):
        tData = inData[healthyDiseaseLabels != 2]
    else:
        tData = inData.copy()
    nPatients, nBioMarkers = tData.shape
    outProbs = np.zeros((nPatients, nBioMarkers, 2))
    for i in range(nPatients):
        for j in range(nBioMarkers):
            if(np.isnan(tData[i][j])):
                outProbs[i][j] = .5, .5
            else:
                outProbs[i][j] = mixtureModels[j].getProb(tData[i][j])
    return outProbs


def patientStaging(probMat, sequence):
    """Calculates the stage that a patient is currently at. Uses the maximum
    likelihood for the event sequence

    Parameters
    ----------
    inData : array-like, shape(numPatients, numBiomarkers)
        All patient-all biomarker measurements.
    probMat : array-like, shape(numPatients, numBioMarkers, 2)
        Probability for a normal/abnormal measurement in all biomarkers
        for all patients (and controls).
    sequence : array-like, shape(numBiomarkers,)
        Event ordering for the biomarkers.

    Returns
    -------
    stages : array-like, shape(numPatients,)
        List of the maximum likelihood stage for each patient.
    """
    pYes = np.array(probMat[:, sequence, 1])
    pNo = np.array(probMat[:, sequence, 0])
    nBioMarkers = len(sequence)

    pStage = np.zeros((len(probMat), nBioMarkers+1))
    for i in range(nBioMarkers+1):
        pStage[:, i] = np.prod(pYes[:, :i], 1) * np.prod(pNo[:, i:nBioMarkers], 1)

    stages = np.zeros(len(probMat))
    for i in range(len(probMat)):
        stages[i] = np.argmax(pStage[i])

    return stages, pStage

def patientStaging_new(probMat, sequence, option=2):
    """New version of patientStaging()
    
    Where there is missing data and p(event) = p(not-event) = 0.5, 
    these stages often become the equally-most-likely, so the staging 
    takes the earliest event.
    
    Two options are made available here:
    1. Use only non-missing data to stage
    2. Take the average of these stages

    Parameters
    ----------
    Same as for patientStaging(), plus
    option : 1 or 2 (corresponding to the options above)

    Returns
    -------
    stages : array-like, shape(numPatients,)
        List of the maximum likelihood stage for each patient.
    stages_new: array-like, shape(numPatients,)
        List of the *new* maximum likelihood stage for each patient.
    
    First coded:  13 March 2019
    Last updated: 13 March 2019
    """
    pYes = np.array(probMat[:, sequence, 1])
    pNo = np.array(probMat[:, sequence, 0])
    nBioMarkers = len(sequence)

    pStage = np.zeros((len(probMat), nBioMarkers+1))
    for i in range(nBioMarkers+1):
        pStage[:, i] = np.prod(pYes[:, :i], 1) * np.prod(pNo[:, i:nBioMarkers], 1)

    stages = np.zeros(len(probMat))
    for i in range(len(probMat)):
        stages[i] = np.argmax(pStage[i])

    if option==1:
        #* Ignore stages with missing data: doesn't handle stage 0 elegantly
        npwhere = np.where(pYes != 0.5)
        n = np.max(npwhere[0])+1
        stages_new = np.empty(stages.shape)
        for k in np.arange(0,n):
            x_s = npwhere[0]==k
            notmissing_k = npwhere[1][x_s]
            p_stages_k = pStage[k][notmissing_k]
            stages_new[k] = notmissing_k[np.argmax(p_stages_k)]
    elif option==2:
        #* Average of multiple ML stages, rounding up
        max_probs = np.max(pStage,axis=1)
        pStage_normalised = pStage / max_probs[:,None]
        npwhere = np.where(pStage_normalised == pStage_normalised.max(axis=1)[:,None])
        n = np.max(npwhere[0])+1
        stages_new = np.empty(stages.shape)
        for k in np.arange(0,n):
            x_s = npwhere[0]==k
            stage_k = np.mean(npwhere[1][x_s])
            stages_new[k] = np.ceil(stage_k)
    else:
        print("patientStaging_new(): ERROR. option=1 or option=2 are the only allowable values. Returning stages_new = None \n")
        stages_new = None
    return stages, pStage, stages_new