import os
import numpy as np
import csv


def adniFileReader(input_file):
    """Read the ADNI data which can be found the Data directory.
    Format for data is similar to that of scikit-learn, that is
    there a feature matrix with a corresponding label array.

    Parameters
    ----------
    adni1 : bool, optional
        If True, just use data from ADNI1, else use all cohorts.

    Returns
    -------
    outData : array-like, shape(nPatients, nBiomarkers)
        Matrix containing the biomarker measurements for each of
        the patients.
    outLabels : array-like, shape(nPatients)
        Diagnosis labels for each patient.
    bioHeaders : array-like, shape(nBiomarkers)
        Name of each of the biomarker measurements in outData.
    """
    # Some constants based on the data location/format
    bmStartIdx, bmEndIdx = 3, 13
    dxIdx = 13

    f = open(input_file, 'rU')
    reader = csv.reader(f, dialect=csv.excel_tab)
    bioHeaders = np.array(next(reader)[bmStartIdx:bmEndIdx])

    # *****************************************
    # Gives each diagnosis an integer label
    # It's assumed that 0 is a healthy control
    # 1 is a patient population of interest
    # 2 is all other labels
    # *****************************************
    dxDictionary = {'CN': 0, 'AD': 1, 'LMCI': 2, 'EMCI': 2, 'SMC': 2}

    outData, outLabels = [], []
    for line in reader:
        # Collect all the biomarkers that are being used as events
        patBmData = line[bmStartIdx:bmEndIdx]
        patBmData = np.array(patBmData, dtype=float)

        # Get the diagnosis of the patient
        patDx = dxDictionary[line[dxIdx]]

        outData.append(patBmData)
        outLabels.append(patDx)

    outData = np.array(outData)
    outLabels = np.array(outLabels, dtype=int)

    return outData, outLabels, bioHeaders
