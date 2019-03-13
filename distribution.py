from scipy import stats
import numpy as np


class Distribution():
    """Wrapper for distributions to be used in the mixture
    modelling. Addition distributions can be added here.

    Attributes
    ----------
    scipDist : scipy.stats.object
        scipy distribution to be used for pdf calculations.
    theta : array-like, shape(2)
        Array of the mean and standard deviation of the normal distribution
    """
    def __init__(self, theta=None):
        """Constructor for the Distribution class.

        Parameters
        ----------
        theta : array-like, shape(2), optional
            An array of the parameters for this distribution.
        """
        self.scipDist = stats.norm
        self.theta = theta

    def getPDF(self, inData):
        """Summary

        Parameters
        ----------
        inData : array-like, shape(nPatients)
            Array of biomarker data for patients. Should not contain
            NaN values.

        Returns
        -------
        name : array-like, shape(nPatients)
            The probability distribution function of each of the values
            from inData.
        """
        return self.scipDist.pdf(inData, loc=self.theta[0],
                                 scale=self.theta[1])

    def setTheta(self, theta):
        """Set's the theta values for this instance of the class.

        Parameters
        ----------
        theta : array-like, shape(2)
            The new value of theta.
        """
        self.theta = theta

    def getBounds(self, inData, modelData, eventSign):
        """Get the bounds be used in the minimisation of the mixture model.

        Parameters
        ----------
        inData : array-like, shape(nPatients)
            All patient data for this biomarker
        modelData : array-like, shape(nPatientsSample)
            Sample of the patient data used to create distribution.
            This is usually either controls or AD diagnosed data.
        eventSign : bool
            1 if this sample mean is greater than the mean of the other
            component in the mixture model, 0 otherwise.

        Returns
        -------
        name : array-like, shape(2, 2)
            (upper-bound, lower-bound) Pairs for each of the parameters in
            theta, i.e. mean and standard deviation.
        """
        if(not eventSign):
            return [(np.nanmean(modelData), np.nanmax(modelData)),
                    (np.finfo(float).eps, np.nanstd(modelData))]
        else:
            return [(np.nanmin(inData), np.nanmean(modelData)),
                    (np.finfo(float).eps, np.nanstd(modelData))]

    def getStart(self, modelData):
        """Gets values for the start point of the optimisation.

        Parameters
        ----------
        modelData : array-like, shape(nPatientsSample)
            Sample of the patient data used to create distribution.
            This is usually either controls or AD diagnosed data.

        Returns
        -------
        name : array-like, shape(2)
            Initial values of parameters in theta for optimisation.
        """
        return [np.nanmean(modelData), np.nanstd(modelData)]
